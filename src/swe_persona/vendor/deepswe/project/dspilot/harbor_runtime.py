"""Harbor 0.23 lifecycle for the staged native_act controller (new scaffold).

Model calls stay on the host. Each Box gets an isolated, CPU-only container.
Original source snapshots, actions, stage gates and grader commands are retained.
This adapter does not reproduce the historical Pier 0.3.1 execution scaffold.
"""
import asyncio
import importlib.metadata
import os
from pathlib import Path
import tempfile
import uuid
import warnings

VERSION = 'deep-swe'

def metadata():
    version = importlib.metadata.version('harbor')
    if version != '0.23.0': raise ValueError('This scaffold requires harbor==0.23.0')
    return {'name':'harbor','version':VERSION,'harbor_version':version,
            'coding_backend':'json_act','historical_pier_equivalence':False,
            'image_overrides': __import__('json').loads((Path(__file__).parents[1]/'environment-images.json').read_text()) if (Path(__file__).parents[1]/'environment-images.json').exists() else {}}

def start(box, verifier_tests, image, resources):
    metadata()
    import yaml
    # Harbor starts here, after the copied worker has returned. Rebuild the
    # sidecar for the platform this compose file requests before it is reused.
    from swe_persona.harbor_bridge import prepare_egress_sidecar
    prepare_egress_sidecar('linux/amd64')
    from harbor.environments.docker.docker import DockerEnvironment
    from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
    from harbor.models.trial.paths import TrialPaths
    from .coding_budget import bounded_timeout
    parent=Path(os.environ.get('DS_HARBOR_WORK_ROOT') or tempfile.gettempdir())
    root=parent/('deepswe-harbor-'+uuid.uuid4().hex[:12]);root.mkdir()
    box._harbor_root=root
    box._harbor_project='ds-'+root.name.removeprefix('deepswe-harbor-')
    env=root/'environment';env.mkdir()
    service={'image':image,'container_name':box.name,'runtime':'runc','platform':'linux/amd64',
             'entrypoint':['/bin/bash','-c'],'command':['sleep infinity'],'working_dir':'/app',
             'network_mode':'none','environment':{'NVIDIA_VISIBLE_DEVICES':'void','CUDA_VISIBLE_DEVICES':''},
             'pids_limit':int(resources.get('pids_limit',512)), 'security_opt':['no-new-privileges:true']}
    if verifier_tests is not None:
        service.update(volumes=[str(Path(verifier_tests).resolve())+':/tests:ro'],cap_drop=['ALL'])
    if resources.get('nofile_limit'):
        limit=int(resources['nofile_limit']);service['ulimits']={'nofile':{'soft':limit,'hard':limit}}
    (env/'docker-compose.yaml').write_text(yaml.safe_dump({'services':{'main':service}}))
    paths=TrialPaths(root/'trial');paths.mkdir()
    memory=str(resources.get('memory','8g')).lower()
    mb=int(float(memory[:-1])*1024) if memory.endswith('g') else int(memory[:-1]) if memory.endswith('m') else int(memory)//1048576
    policy=NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
    box._harbor_loop=asyncio.new_event_loop()
    box._harbor_env=DockerEnvironment(environment_dir=env,environment_name='ds-'+uuid.uuid4().hex[:12],
        session_id=box._harbor_project,trial_paths=paths,task_env_config=EnvironmentConfig(
            docker_image=image,cpus=int(resources.get('cpus',2)),memory_mb=mb),
        network_policy=policy,phase_network_policies=[policy])
    try:
        box._harbor_loop.run_until_complete(asyncio.wait_for(box._harbor_env.start(force_build=False),bounded_timeout(300)))
    except BaseException:
        try: close(box)
        except Exception as exc:
            warnings.warn(f'DeepSWE startup cleanup failed: {type(exc).__name__}', RuntimeWarning)
        raise

def close(box):
    if box._harbor_loop.is_closed(): return
    from swe_persona.harbor_bridge import remove_empty_session_network
    try: box._harbor_loop.run_until_complete(box._harbor_env.stop(delete=False))
    finally:
        try: remove_empty_session_network(box._harbor_root,box._harbor_project)
        finally: box._harbor_loop.close()
