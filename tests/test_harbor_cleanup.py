"""A stopped benchmark session must not exhaust the shared Docker address pool."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from swe_persona.harbor_bridge import (align_egress_sidecar_platform,
                                       cleanup_empty_session_networks,
                                       drop_wrong_platform_tag,
                                       prepare_egress_sidecar,
                                       remove_empty_session_network)


class SessionNetworkCleanupTests(unittest.TestCase):
    project = 'sp-harbor-123456abcdef'

    def inspect(self, owner=None, endpoints=None):
        return subprocess.CompletedProcess([], 0, json.dumps([{
            'Id': 'recorded-network-id',
            'Labels': {'com.docker.compose.project': owner or self.project},
            'Containers': endpoints or {},
        }]), '')

    def test_empty_owned_network_is_removed_by_id_with_a_receipt(self):
        with tempfile.TemporaryDirectory() as temp, patch('swe_persona.harbor_bridge.subprocess.run',
                side_effect=[self.inspect(), subprocess.CompletedProcess([], 0, '', '')]) as run:
            receipt = remove_empty_session_network(temp, self.project)
            self.assertTrue(receipt['removed'])
            self.assertEqual(receipt['errors'], [])
            self.assertEqual(run.call_args_list[1].args[0],
                             ['docker', 'network', 'rm', 'recorded-network-id'])
            self.assertEqual(json.loads((Path(temp)/'resource-cleanup.json').read_text()), receipt)

    def test_foreign_or_live_network_is_never_removed(self):
        for response, reason in [(self.inspect(owner='another-project'), 'network_owner_mismatch'),
                                 (self.inspect(endpoints={'container-id': {}}), 'network_has_live_endpoints')]:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as temp, \
                    patch('swe_persona.harbor_bridge.subprocess.run', return_value=response) as run, \
                    self.assertWarns(RuntimeWarning):
                receipt = remove_empty_session_network(temp, self.project)
                self.assertFalse(receipt['removed'])
                self.assertEqual(receipt['errors'], [reason])
                self.assertEqual(run.call_count, 1)

    def test_absent_network_is_already_clean_and_unexpected_project_is_rejected(self):
        absent = subprocess.CompletedProcess([], 1, '', 'Error: No such network')
        with tempfile.TemporaryDirectory() as temp, patch('swe_persona.harbor_bridge.subprocess.run',
                                                        return_value=absent) as run:
            self.assertTrue(remove_empty_session_network(temp, self.project)['already_absent'])
            with self.assertRaises(ValueError):
                remove_empty_session_network(temp, 'shared-lab-project')
            self.assertEqual(run.call_count, 1)

    def test_docker_unavailability_is_bounded_audited_and_does_not_prune(self):
        with tempfile.TemporaryDirectory() as temp, patch('swe_persona.harbor_bridge.subprocess.run',
                side_effect=subprocess.TimeoutExpired('docker', 5)) as run, self.assertWarns(RuntimeWarning):
            receipt = remove_empty_session_network(temp, self.project)
            self.assertEqual(receipt['errors'], ['TimeoutExpired'])
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.kwargs['timeout'], 5)

    def test_exited_worker_sessions_are_reclaimed_from_host_metadata(self):
        with tempfile.TemporaryDirectory() as temp, patch('swe_persona.harbor_bridge.subprocess.run',
                side_effect=[self.inspect(), subprocess.CompletedProcess([], 0, '', '')]):
            root = Path(temp)/'episode-runtime/environments/harbor-123456abcdef'
            (root/'environment').mkdir(parents=True)
            (root/'environment/docker-compose.yaml').write_text('services: {}')
            result = cleanup_empty_session_networks(temp)
            self.assertEqual(len(result['sessions']), 1)
            self.assertTrue(result['sessions'][0]['removed'])
            self.assertFalse(result['budget_exhausted'])

    def test_agent_artifacts_and_symlinks_cannot_authorize_network_cleanup(self):
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as outside, \
                patch('swe_persona.harbor_bridge.subprocess.run') as run:
            forged = Path(temp)/'artifacts/harbor-123456abcdef/environment'
            forged.mkdir(parents=True);(forged/'docker-compose.yaml').write_text('services: {}')
            cache = Path(temp)/'environments';cache.mkdir()
            (cache/'harbor-123456abcdef').symlink_to(outside)
            self.assertEqual(cleanup_empty_session_networks(temp)['sessions'], [])
            run.assert_not_called()

    def test_cli_reclaims_empty_networks_after_timeout_or_interruption(self):
        from swe_persona.cli import worker
        for interrupted in (False, True):
            with self.subTest(interrupted=interrupted), tempfile.TemporaryDirectory() as temp, \
                    patch('swe_persona.cli.run_bounded',
                          side_effect=KeyboardInterrupt if interrupted else None,
                          return_value={'process_exit_code': -2, 'timed_out': True}), \
                    patch('swe_persona.harbor_bridge.cleanup_empty_session_networks') as cleanup:
                if interrupted:
                    with self.assertRaises(KeyboardInterrupt):
                        worker(temp, 'run', temp, time_limit={'effective_sec': 30})
                else:
                    self.assertEqual(worker(temp, 'run', temp, time_limit={'effective_sec': 30}), -2)
                cleanup.assert_called_once_with(temp)


class SidecarPlatformTests(unittest.TestCase):
    def setUp(self):
        from harbor.environments.docker.docker import DockerEnvironment
        self.environment = DockerEnvironment
        self.original = DockerEnvironment._ensure_egress_control_sidecar_image_built
        self.addCleanup(setattr, DockerEnvironment,
                        '_ensure_egress_control_sidecar_image_built', self.original)

    def test_compose_platform_is_used_for_the_sidecar_build(self):
        with patch.dict(os.environ, {'DOCKER_DEFAULT_PLATFORM': 'linux/amd64'}):
            self.assertTrue(align_egress_sidecar_platform())
            aligned = self.environment._ensure_egress_control_sidecar_image_built
            self.assertTrue(aligned._swe_persona_aligned)
            self.assertTrue(align_egress_sidecar_platform())
            self.assertIs(self.environment._ensure_egress_control_sidecar_image_built, aligned)
            fake = SimpleNamespace(_env_vars=SimpleNamespace(), logger=None,
                                   _EGRESS_CONTROL_SIDECAR_DOCKER_NAME='sidecar',
                                   _EGRESS_CONTROL_SIDECAR_CONTEXT_PATH=Path('/context'))
            fake._egress_control_sidecar_dockerfile_path = lambda: Path('/context/Dockerfile')
            with patch('harbor.environments.docker.utils.docker_build_context_hash',
                       return_value='hash'), \
                 patch('harbor.environments.docker.utils._compute_image_name',
                       return_value='sidecar:amd64'), \
                 patch('swe_persona.harbor_bridge.drop_wrong_platform_tag',
                       return_value=False), \
                 patch('harbor.environments.docker.utils.ensure_docker_image_built',
                       new=AsyncMock(return_value='sidecar:amd64')) as build:
                asyncio.run(aligned(fake))
            self.assertEqual(build.await_args.kwargs['platform'], 'linux/amd64')
            self.assertEqual(fake._env_vars.egress_control_sidecar_image_name, 'sidecar:amd64')

    def test_wrong_architecture_tag_is_removed_before_rebuild(self):
        with patch.dict(os.environ, {'DOCKER_DEFAULT_PLATFORM': 'linux/amd64'}):
            self.assertTrue(align_egress_sidecar_platform())
            aligned = self.environment._ensure_egress_control_sidecar_image_built
            fake = SimpleNamespace(_env_vars=SimpleNamespace(), logger=None,
                                   _EGRESS_CONTROL_SIDECAR_DOCKER_NAME='sidecar',
                                   _EGRESS_CONTROL_SIDECAR_CONTEXT_PATH=Path('/context'))
            fake._egress_control_sidecar_dockerfile_path = lambda: Path('/context/Dockerfile')
            with patch('harbor.environments.docker.utils.docker_build_context_hash',
                       return_value='hash'), \
                 patch('harbor.environments.docker.utils._compute_image_name',
                       return_value='sidecar--hash'), \
                 patch('swe_persona.harbor_bridge.drop_wrong_platform_tag',
                       return_value=True) as drop, \
                 patch('harbor.environments.docker.utils.ensure_docker_image_built',
                       new=AsyncMock(return_value='sidecar--hash')):
                asyncio.run(aligned(fake))
        drop.assert_called_once_with('sidecar--hash', 'linux/amd64')

    def test_prepare_rebuilds_only_a_wrong_architecture_sidecar(self):
        with patch('swe_persona.harbor_bridge.drop_wrong_platform_tag', return_value=True) as drop, \
             patch('swe_persona.harbor_bridge.subprocess.run', autospec=True, side_effect=[
                 subprocess.CompletedProcess([], 1, '', ''),
                 subprocess.CompletedProcess([], 0, '', '')]) as run:
            image = prepare_egress_sidecar('linux/amd64')
        self.assertTrue(image.endswith('--9af3dc172792cee8'))
        drop.assert_called_once_with(image, 'linux/amd64')
        self.assertEqual(run.call_args_list[1].args[0][1:3], ['buildx', 'build'])
        self.assertIn('--platform=linux/amd64', run.call_args_list[1].args[0])
        from harbor.environments.docker.docker import default_docker_platform
        self.assertEqual(asyncio.run(default_docker_platform()), 'linux/amd64')

    def test_wrong_architecture_tag_is_dropped_and_matching_tag_stays(self):
        with patch('swe_persona.harbor_bridge.subprocess.run', autospec=True, side_effect=[
                subprocess.CompletedProcess([], 0, 'linux/arm64\n', ''),
                subprocess.CompletedProcess([], 0, '', '')]) as run:
            self.assertTrue(drop_wrong_platform_tag('sidecar--hash', 'linux/amd64'))
        self.assertEqual(run.call_args_list[1].args[0][:2], ['docker', 'rmi'])
        with patch('swe_persona.harbor_bridge.subprocess.run', autospec=True, side_effect=[
                subprocess.CompletedProcess([], 0, 'linux/amd64\n', '')]) as run:
            self.assertFalse(drop_wrong_platform_tag('sidecar--hash', 'linux/amd64'))
        self.assertEqual(run.call_count, 1)

    def test_unset_or_unknown_platform_keeps_harbor_default(self):
        for value in (None, 'linux/386'):
            with self.subTest(value=value), patch.dict(os.environ, {}, clear=False):
                os.environ.pop('DOCKER_DEFAULT_PLATFORM', None)
                if value: os.environ['DOCKER_DEFAULT_PLATFORM'] = value
                self.assertFalse(align_egress_sidecar_platform())
                self.assertIs(self.environment._ensure_egress_control_sidecar_image_built, self.original)


if __name__ == '__main__':
    unittest.main()
