"""Use Harbor to own SWE-Pro event-profile task container lifecycles.

The source controller retains its command/snapshot/scoring semantics. There is
one Harbor-managed main container per trial; scoring snapshots stay isolated.
"""
import asyncio
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid
import warnings
import yaml


def compose_platform():
    """Return the platform Compose will apply, not the daemon architecture."""
    platform = os.environ.get('DOCKER_DEFAULT_PLATFORM', '').strip()
    if re.fullmatch(r'linux/(?:amd64|arm64)', platform):
        return platform
    return None


def drop_wrong_platform_tag(image, platform):
    """Untag an existing image whose architecture Compose will reject.

    Harbor treats a content-addressed name as reusable without inspecting its
    architecture. Removing only the tag lets the requested platform rebuild it.
    """
    wanted = platform.split('/', 1)[-1]
    inspected = subprocess.run(
        ['docker', 'image', 'inspect', '--format', '{{.Os}}/{{.Architecture}}', image],
        capture_output=True, text=True)
    if inspected.returncode:
        return False
    if inspected.stdout.strip().split('/', 1)[-1] == wanted:
        return False
    removed = subprocess.run(['docker', 'rmi', image], capture_output=True, text=True)
    return removed.returncode == 0


def prepare_egress_sidecar(platform):
    """Make the content-addressed sidecar match the platform Compose requests.

    The image name includes the platform, but Harbor reuses that tag after a
    name-only inspect. A tag left by the daemon architecture must be removed
    before the requested platform can rebuild it.
    """
    if not re.fullmatch(r'linux/(?:amd64|arm64)', platform or ''):
        return None
    from harbor.environments.docker.docker import DockerEnvironment
    from harbor.environments.docker.utils import (
        docker_build_context_hash, _compute_image_name)
    image = _compute_image_name(
        DockerEnvironment._EGRESS_CONTROL_SIDECAR_DOCKER_NAME,
        docker_build_context_hash(
            context=DockerEnvironment._EGRESS_CONTROL_SIDECAR_CONTEXT_PATH,
            dockerfile_path=DockerEnvironment._egress_control_sidecar_dockerfile_path(),
            build_args={}, platform=platform))
    drop_wrong_platform_tag(image, platform)
    present = subprocess.run(['docker', 'image', 'inspect', image],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if present.returncode == 0:
        return image
    dockerfile = DockerEnvironment._egress_control_sidecar_dockerfile_path()
    context = DockerEnvironment._EGRESS_CONTROL_SIDECAR_CONTEXT_PATH
    built = subprocess.run(
        ['docker', 'buildx', 'build', '--file=' + str(dockerfile), '--platform=' + platform,
         '--output=type=docker,name=' + image, str(context)],
        capture_output=True, text=True)
    if built.returncode:
        raise RuntimeError('Failed to build sidecar ' + image + ': ' + built.stderr[-800:])
    import harbor.environments.docker.docker as docker_mod

    async def requested_platform():
        return platform

    # Harbor's own builder asks the daemon unless this binding is replaced.
    docker_mod.default_docker_platform = requested_platform
    return image


def align_egress_sidecar_platform():
    """Build Harbor's egress sidecar for the platform Compose will request.

    Compose applies ``DOCKER_DEFAULT_PLATFORM`` to every service, including the
    sidecar. Harbor otherwise builds that image for the daemon architecture and
    reuses the tag without checking it, so an arm64 image keeps that name.
    """
    platform = compose_platform()
    if platform is None:
        return False
    from harbor.environments.docker.docker import DockerEnvironment
    current = DockerEnvironment._ensure_egress_control_sidecar_image_built
    if getattr(current, '_swe_persona_aligned', False):
        return True
    original = current

    async def build_sidecar(self):
        requested = compose_platform()
        if requested is None:
            return await original(self)
        from harbor.environments.docker.utils import (
            docker_build_context_hash, ensure_docker_image_built, _compute_image_name)
        image = _compute_image_name(
            self._EGRESS_CONTROL_SIDECAR_DOCKER_NAME,
            docker_build_context_hash(
                context=self._EGRESS_CONTROL_SIDECAR_CONTEXT_PATH,
                dockerfile_path=self._egress_control_sidecar_dockerfile_path(),
                build_args={}, platform=requested))
        drop_wrong_platform_tag(image, requested)
        self._env_vars.egress_control_sidecar_image_name = await ensure_docker_image_built(
            docker_name=self._EGRESS_CONTROL_SIDECAR_DOCKER_NAME,
            docker_build_context=self._EGRESS_CONTROL_SIDECAR_CONTEXT_PATH,
            dockerfile_path=self._egress_control_sidecar_dockerfile_path(),
            build_args={},
            platform=requested,
            logger=self.logger,
        )

    build_sidecar._swe_persona_aligned = True
    DockerEnvironment._ensure_egress_control_sidecar_image_built = build_sidecar
    return True


def remove_empty_session_network(root, project, timeout_sec=10):
    """Reclaim only this bridge's stopped session, never prune shared Docker."""
    receipt = {"project": project, "removed": False, "errors": []}
    if not re.fullmatch(r"(?:sp-harbor-|ds-)[0-9a-f]{12}", project):
        raise ValueError("Unexpected benchmark compose project")
    network = project + "_default"
    started = time.monotonic()
    deadline = started + timeout_sec
    try:
        inspected = subprocess.run(['docker', 'network', 'inspect', network],
                                   capture_output=True, text=True, timeout=min(5, timeout_sec/2))
        if inspected.returncode:
            if any(text in inspected.stderr.lower() for text in
                   ('no such network', 'not found', 'no such object')):
                receipt['already_absent'] = True
            else:
                receipt['errors'].append('network_inspection_failed')
        else:
            details = json.loads(inspected.stdout)[0]
            labels = details.get('Labels') or {}
            if labels.get('com.docker.compose.project') != project:
                receipt['errors'].append('network_owner_mismatch')
            elif details.get('Containers'):
                receipt['errors'].append('network_has_live_endpoints')
            else:
                # Remove the inspected ID, so a name replaced between inspection
                # and removal cannot cause us to remove a different network.
                removed = subprocess.run(['docker', 'network', 'rm', details['Id']],
                                         capture_output=True, text=True,
                                         timeout=max(0.001, min(5, deadline-time.monotonic())))
                receipt['removed'] = removed.returncode == 0
                if removed.returncode:
                    receipt['errors'].append('network_removal_failed')
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, IndexError) as exc:
        receipt['errors'].append(type(exc).__name__)
    path = Path(root)/'resource-cleanup.json'
    path.write_text(json.dumps(receipt, indent=2) + '\n')
    if receipt['errors']:
        warnings.warn(f'Benchmark session network cleanup incomplete; see {path}', RuntimeWarning)
    return receipt


def cleanup_empty_session_networks(output, timeout_sec=10):
    """Reclaim networks after a worker exits, including watchdog interruption.

    Inspect only known host-controlled bridge directories, never agent artifacts
    or unrelated runs. A killed worker cannot run its own normal teardown.
    """
    output = Path(output).resolve()
    deadline = time.monotonic() + timeout_sec
    receipt = {'sessions': [], 'budget_exhausted': False}
    for cache in (output/'environments', output/'episode-runtime/environments'):
        if cache.is_symlink() or not cache.resolve().is_relative_to(output):
            continue
        for root in sorted([*cache.glob('harbor-*'), *cache.glob('deepswe-harbor-*')]):
            if (root.is_symlink() or not root.is_dir()
                    or not root.resolve().is_relative_to(output)
                    or not (root/'environment/docker-compose.yaml').is_file()
                    or not re.fullmatch(r'(?:harbor-|deepswe-harbor-)[0-9a-f]{12}', root.name)):
                continue
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                receipt['budget_exhausted'] = True
                break
            project = ('ds-'+root.name.removeprefix('deepswe-harbor-')
                       if root.name.startswith('deepswe-harbor-') else 'sp-'+root.name)
            receipt['sessions'].append(remove_empty_session_network(
                root, project, timeout_sec=remaining))
        if receipt['budget_exhausted']:
            break
    receipt['elapsed_sec'] = max(0, time.monotonic()-(deadline-timeout_sec))
    (output/'session-network-cleanup.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt

def install(dockerx, cache, workdir='/app', sample_labels=False):
    from harbor.environments.docker.docker import DockerEnvironment
    from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
    from harbor.models.trial.paths import TrialPaths
    original_remove=dockerx.Container.remove

    def start(self):
        root=Path(cache)/('harbor-'+uuid.uuid4().hex[:12]);root.mkdir(parents=True)
        self._harbor_root=root;self._harbor_project='sp-'+root.name
        envdir=root/'environment';envdir.mkdir()
        # All mounts are explicit source-controller assets. Never mount row.json,
        # model configuration, the entire dataset, or host credentials.
        volumes=[f'{m.source}:{m.target}'+(':ro' if m.read_only else '') for m in self.mounts]
        compose={'services':{'main':{'image':self.image,'container_name':self.name,
            'entrypoint':['/bin/bash','-c'],'command':['sleep infinity'],'working_dir':workdir,'volumes':volumes}}}
        if sample_labels:
            compose['services']['main']['network_mode']='none'
            compose['services']['main']['security_opt']=['no-new-privileges:true']
            owner=os.environ.get('PERSONACODE_SAMPLE_OWNER')
            if owner:
                compose['services']['main']['labels']={'personacode.sample':owner}
                compose['services']['harbor-docker-egress-control-sidecar']={
                    'labels':{'personacode.sample':owner,'personacode.fixture':'true'}}
        (envdir/'docker-compose.yaml').write_text(yaml.safe_dump(compose))
        paths=TrialPaths(root/'trial');paths.mkdir()
        # Harbor enforces egress policy; SWE-Live also declares literal network_mode: none.
        policy=NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
        self._harbor_loop=asyncio.new_event_loop()
        self._harbor_env=DockerEnvironment(environment_dir=envdir,environment_name='sp-'+root.name,
            session_id='sp-'+root.name,trial_paths=paths,task_env_config=EnvironmentConfig(
                docker_image=self.image,cpus=int(self.cpus),memory_mb=_memory_mb(self.memory)),
            network_policy=policy,phase_network_policies=[policy])
        try:
            launch=self._harbor_env.start(force_build=False)
            deadline=getattr(self,'deadline',None)
            if deadline:launch=asyncio.wait_for(launch,timeout=deadline.remaining())
            self._harbor_loop.run_until_complete(launch)
        except BaseException:
            try:self._harbor_loop.run_until_complete(self._harbor_env.stop(delete=False))
            finally:
                self._harbor_loop.close()
                remove_empty_session_network(root,self._harbor_project)
            raise
        return subprocess.check_output(['docker','inspect','--format','{{.Id}}',self.name],text=True).strip()

    def remove(self):
        if not hasattr(self,'_harbor_env'):return original_remove(self)
        if self._harbor_loop.is_closed():return
        try:self._harbor_loop.run_until_complete(self._harbor_env.stop(delete=False))
        finally:
            self._harbor_loop.close()
            remove_empty_session_network(self._harbor_root,self._harbor_project)
    dockerx.Container.start=start;dockerx.Container.remove=remove

def _memory_mb(value):
    value=str(value).lower()
    if value.endswith('g'):return int(float(value[:-1])*1024)
    if value.endswith('m'):return int(float(value[:-1]))
    return max(1,int(value)//(1024*1024))
