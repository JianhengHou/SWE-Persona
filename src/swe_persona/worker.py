"""Isolated profile worker. Import names in inherited runtimes cannot collide."""
import hashlib
import argparse
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

from .api import RoleAPI, load_config, role_name, chat_token_options
from .dataset import write

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def install_tb3(root,config,out):
    sys.path.insert(0,str(root/'runtime'))
    import azure_client
    if config:
        def call(role,prompt,schema,audit_dir,timeout=600,images=None,messages=None):
            chosen=role_name(role)
            if messages is None:
                content=[{'type':'input_text','text':prompt}]
                content += [{'type':'input_image','image_url':uri} for uri in (images or [])]
                messages=[{'role':'user','content':content}] if images else [{'role':'user','content':prompt}]
            # Agent calls use the shared effort budget and bounded growth policy.
            # Keep the existing per-call fallback for simulator/judge roles.
            limit=None if chosen=='agent' else 16384
            return RoleAPI(config,chosen,audit_dir).json(messages,schema,max_tokens=limit,timeout=timeout)
        azure_client.call=call
    # Regenerate executable bindings after stripping construction-only fields.
    import requirement_loop
    budget=json.loads((root/'row.json').read_text())['budget']
    requirement_loop.POLICY.update(max_user_turns=budget['max_user_turns'],max_simulator_steps=budget['max_simulator_decisions'])
    for p in (root/'tasks').glob('*/tests/persona/binding.json'):
        requirement_loop.write_bundle(json.loads(p.read_text()),p.parent/'closed_loop')

def install_v6(root,config,out):
    sys.path.insert(0,str(root/'runtime'))
    import pcb_v6.llm as llm
    if not config:return
    old_init=llm.Llm.__init__
    def init(self,model=None,role='unspecified',*args,**kwargs):
        chosen=role_name(role)
        old_init(self,config[chosen]['model'],role,*args,**kwargs)
        self._role_api_config=config;self._role_api_name=chosen
    def responses(self,messages,schema,name,max_output_tokens):
        if self._transport is not None:
            return self._transport(messages=messages,schema=schema,name=name,max_output_tokens=max_output_tokens)
        return RoleAPI(config,self._role_api_name,out/'api').raw(messages,schema,name,max_output_tokens)
    llm.Llm.__init__=init;llm.Llm._responses=responses
    os.environ['PCB_MODEL']=config['judge']['model']
    os.environ['PCB_AGENT_MODEL']=config['agent']['model']
    os.environ['OPENAI_BASE_URL']=config['judge']['base_url']
    os.environ['OPENAI_API_KEY']=os.environ.get(config['judge']['api_key_env'],'')
    os.environ['SWE_PERSONA_AGENT_REQUEST_OPTIONS']=json.dumps(chat_token_options(config['agent']))
    os.environ['SWE_PERSONA_AGENT_HTTP_TIMEOUT']=str(config['agent'].get('timeout_sec',120))
    os.environ['PCB_AGENT_BASE_URL']=config['agent']['base_url']
    os.environ['PCB_AGENT_API_KEY']=os.environ.get(config['agent']['api_key_env'],'')

def install_native_reference_control(row,destination,fixtures):
    controls=json.loads((Path(fixtures)/'native-reference-controls.json').read_text())
    if controls.get('protocol')!='swepro-native-reference-controls-v1':
        raise ValueError('Unknown native reference control protocol')
    matches=[c for c in controls['controls'] if c['instance_id']==row['source']['instance_id']]
    if len(matches)!=1:raise ValueError('Missing or duplicate native reference control')
    control=matches[0]
    if control['base_commit']!=row['environment']['expected_base_commit']:
        raise ValueError('Native reference control base mismatch')
    for name,key in [('test.patch','test_patch_sha256'),('run_script.sh','runner_sha256'),('parser.py','parser_sha256')]:
        if hashlib.sha256((destination/name).read_bytes()).hexdigest()!=control[key]:
            raise ValueError('Native reference control asset mismatch: '+name)
    reference=control['reference_patch'].encode()
    if hashlib.sha256(reference).hexdigest()!=control['reference_patch_sha256']:
        raise ValueError('Native reference patch checksum mismatch')
    (destination/'gold.patch').write_bytes(reference)

def install_v7(root,row,config,out,calibration=False):
    runner=root/'portable/runtime-v3/runner'
    sys.path.insert(0,str(runner));sys.path.insert(1,str(runner.parent))
    os.environ['PCB_PERSONA_FILE']=str(root/'portable/personas.json')
    os.environ['PCB_COMPLETE_NATIVE_OPENING']='1'
    import tb2_persona_pilot.llm as base
    if config:
        aliases={'swe-persona-agent':'agent','swe-persona-simulator':'user_simulator',config['judge']['model']:'judge'}
        old_resolve=base.resolve_model
        def resolve(name):
            if name not in aliases:return old_resolve(name)
            c=config[aliases[name]]
            return base.ModelSpec(c['model'],'azure_responses',c['base_url'],c['api_key_env'],
                                  c.get('strict_json_schema',True),{}, {})
        base.resolve_model=resolve
        old_init=base.Llm.__init__
        def init(self,model=base.DEFAULT_MODEL,*args,**kwargs):
            self._sp_role=aliases.get(model,'judge')
            old_init(self,model,*args,**kwargs)
        def response(self,messages,max_output_tokens,schema,schema_name):
            data=RoleAPI(config,self._sp_role,out/'api').raw(messages,schema,schema_name,max_output_tokens)
            return data['text'],(data['input_tokens'],data['output_tokens']),{}
        base.Llm.__init__=init;base.Llm._azure_responses=response
    import dockerx
    from .harbor_bridge import install
    install(dockerx,out/'environments')
    if calibration:return
    configure_verifier_parallelism(row,dockerx,out)
    # Ordinary evaluation consumes assets from the selected Parquet row; no HF
    # dataset download, author's cache or source checkout is needed. Reference
    # controls are packaged separately and mounted only for native diagnosis.
    import task_adapter
    short=row['source']['task_id'];tests=root/'tasks'/short/'tests'
    spec=json.loads((tests/'task_spec.json').read_text())
    public=json.loads(row['native_task']['config_json'])
    native=dict(spec,**public,instance_id=row['source']['instance_id'],base_commit=row['environment']['expected_base_commit'],
                repo=row['source']['repository'] or spec['repo'],patch='',test_patch=(tests/'test.patch').read_text(),
                before_repo_set_cmd='',dockerhub_tag='parquet-pinned',repo_language=spec.get('repo_language',''))
    original_from_dataset=task_adapter.TaskManifest.from_dataset.__func__
    def from_dataset(cls,*,instance_id=None,dataset_index=None,index_base=0):
        if instance_id != row['source']['instance_id']:
            return original_from_dataset(cls,instance_id=instance_id,dataset_index=dataset_index,index_base=index_base)
        manifest=cls.from_row(native,row['source']['row_index'])
        resources=row['environment']['agent']['resources']
        manifest.limits.update(cpus=resources['cpus'],memory=str(resources['memory_mb'])+'m',network='none')
        return manifest
    task_adapter.TaskManifest.from_dataset=classmethod(from_dataset)
    source_commit=json.loads(row['interaction']['config_json']).get('source_commit')
    task_adapter.TaskManifest.source_commit=property(lambda self: source_commit)
    def materialize(self,destination,fixtures,**kwargs):
        destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
        for p in tests.iterdir():
            if p.is_file():shutil.copy2(p,destination/p.name)
        # Frozen row assets hold test scripts/data. Executable checker logic
        # comes from this runtime, including failure classification repairs.
        shutil.copy2(Path(fixtures)/'checker_driver.py',destination/'checker_driver.py')
        install_native_reference_control(row,destination,fixtures)
        hashes={name:hashlib.sha256((destination/name).read_bytes()).hexdigest()
                for name in ('run_script.sh','parser.py','checker_driver.py','task_spec.json')}
        write(destination/'asset_hashes.json',hashes)
        return destination
    task_adapter.TaskManifest.materialize_checker_assets=materialize

def configure_verifier_parallelism(row,dockerx,out):
    # Old Go toolchains size build parallelism from host CPUs rather than the
    # Docker CPU quota. Teleport's EC2 dependency can exceed the 4 GiB task cap.
    if row['source']['repository']!='gravitational/teleport':return
    cpus=max(1,int(row['environment']['verifier']['resources']['cpus']))
    original=dockerx.run_ephemeral
    def run(image,script,**kwargs):
        return original(image,f'export GOMAXPROCS={cpus} GOFLAGS=-p=1; '+script,**kwargs)
    dockerx.run_ephemeral=run
    write(out/'verifier-environment.json',{
        'variant':'teleport-verifier-go-parallelism-v1',
        'environment':{'GOMAXPROCS':str(cpus),'GOFLAGS':'-p=1'},
        'reason':'Bound Go build concurrency to avoid compiler OOM under the original container memory cap.',
        'test_selection_unchanged':True,'resource_limits_unchanged':True})

def prepare_public_submodules(row,out):
    sources={'gravitational/teleport':('teleport-webassets.git','https://github.com/gravitational/webassets.git'),
             'future-architect/vuls':('vuls-integration.git','https://github.com/vulsio/integration')}
    source=sources.get(row['source']['repository'])
    if not source:return
    import fcntl
    import tempfile
    cache=Path(os.environ.get('PCB_SUBMODULE_CACHE',Path.home()/'.cache/swe-persona/submodules')).resolve()
    cache.mkdir(parents=True,exist_ok=True)
    os.environ['PCB_SUBMODULE_CACHE']=str(cache)
    name,url=source;target=cache/name
    def git(*args):
        return subprocess.run(['git',*args],check=True,capture_output=True,text=True,timeout=900).stdout.strip()
    with (cache/(name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not target.exists():
            with tempfile.TemporaryDirectory(prefix='mirror-',dir=cache) as temp:
                staged=Path(temp)/name
                git('clone','--mirror',url,str(staged))
                staged.rename(target)
        if git('-C',str(target),'remote','get-url','origin')!=url:
            raise ValueError('Unexpected public submodule mirror origin: '+name)
        git('-C',str(target),'fsck','--full')
    write(out/'public-submodules.json',{'mirror':name,'origin':url,'fsck':'pass',
          'task_gitlinks':'Verified by clean-base preparation before the coding episode.'})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--config',type=Path);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--mode',choices=['dry-run','run','calibrate','live-episode'],required=True)
    ap.add_argument('--gate-dir',type=Path)
    ap.add_argument('--gold-dir',type=Path)
    ap.add_argument('--sample-seconds',type=float)
    ap.add_argument('--deadline-monotonic',type=float)
    ap.add_argument('--output-dir',type=Path)
    args=ap.parse_args();root=args.root.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    row=json.loads((root/'row.json').read_text());profile=row['interaction']['profile_id']
    config=load_config(args.config,require_keys=args.mode!='dry-run') if args.config else None
    if args.mode!='dry-run' and config is None:raise ValueError('API configuration is required')
    os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
    from .harbor_bridge import align_egress_sidecar_platform, prepare_egress_sidecar
    align_egress_sidecar_platform()
    if os.environ.get('DOCKER_DEFAULT_PLATFORM'):
        prepare_egress_sidecar(os.environ['DOCKER_DEFAULT_PLATFORM'])
    if config:os.environ['PCB_MODEL']=config['judge']['model']
    if config:
        # Legacy prerequisite checks read these variables, but model calls are
        # routed by role through RoleAPI. No key is written into task files.
        for prefix in ['OPENAI','AZURE_OPENAI']:
            os.environ[prefix+'_BASE_URL']=config['judge']['base_url']
            os.environ[prefix+'_API_KEY']=os.environ.get(config['judge']['api_key_env'],'')
    if profile=='deep-swe':
        from .deepswe import execute
        return execute(root,row,config,out,args.mode,args.gate_dir)
    if profile=='swe-together':
        from .swe_together import execute
        return execute(root,row,config,out,args.mode,args.gate_dir)
    if profile=='swe-bench-live':
        from .swe_live import install_live,execute
        install_live(root,config,out)
        return execute(root,row,config,out,args.mode,args.config,args.gate_dir,args)
    if profile=='terminal-bench-3':install_tb3(root,config,out)
    elif profile=='swe-bench-pro-mini':install_v6(root,config,out)
    else:install_v7(root,row,config,out,args.mode=='calibrate')
    if args.mode=='calibrate':
        if profile=='swe-bench-pro-shell':
            import validate_rubric
            sys.argv=['validate_rubric.py','--judge-model',config['judge']['model'],'--samples','3','--out',str(out)]
            return validate_rubric.main()
        sys.argv=[str(root/'calibrate_judge.py'),'--output',str(out)]
        runpy.run_path(str(root/'calibrate_judge.py'),run_name='__main__');return 0
    if profile=='swe-bench-pro-shell':
        entry=module('sp_run_case',root/'portable/run_case.py')
        case_id=json.loads(row['interaction']['config_json'])['case_id']
        if args.mode=='dry-run':
            case,runtime,binding,candidates=entry.prepare(case_id)
            print(json.dumps({'task_id':row['task_id'],'status':'pass','requirements':len(candidates),
                              'requirements_expected':len(row['requirements']),
                              'loaded_requirement_ids':[c.requirement_id for c in candidates],
                              'requirement_loader':'registry.build_candidates + applicable_candidates',
                              'backend':'harbor-docker','profile':profile}));return 0
        prepare_public_submodules(row,out)
        if args.gate_dir:os.environ['PCB_RUBRIC_GATE_DIR']=str(args.gate_dir.resolve())
        # ShellAgent uses the shared role API on both chat and Responses services.
        os.environ['PCB_AGENT_ADAPTER']='portable_shell:ShellAgent'
        case,runtime,binding,candidates=entry.prepare(case_id)
        import run_episode
        inventory=json.loads((root/'inventory.json').read_text())['tasks'][0]
        sys.argv=['run_episode.py','--instance-id',row['source']['instance_id'],
            '--persona',row['persona']['persona_id'],'--data-version',binding['data_version'],
            '--max-turns',str(row['budget']['max_user_turns']),
            '--max-steps',str(row['budget']['max_agent_queries_per_turn']),
            '--max-evolved',str(len(row['requirements'])),
            '--agent-model','swe-persona-agent','--adapter-model','swe-persona-agent',
            '--simulator-model','swe-persona-simulator','--rubric-model',config['judge']['model'],
            '--judge-model',config['judge']['model'],'--rubric-feedback','online',
            '--runs-root',str(out),'--run-id',case_id+'-a1',
            '--first-message-file',str(root/'portable'/case['opening_path']),
            '--image-ref',inventory['pinned_oci_reference']]
        return run_episode.main()
    short=row['source']['task_id']
    sys.argv=[str(root/'run.py'),'--task',short,'--output',str(out)]
    if args.mode=='dry-run':sys.argv+=['--dry-run']
    elif profile=='swe-bench-pro-mini':
        if config['agent']['api']!='chat_completions':
            raise ValueError('swe-bench-pro-mini mini agent requires a Chat Completions endpoint')
        sys.argv+=['--agent-model',config['agent']['model'],'--mini-venv',sys.prefix]
    runpy.run_path(str(root/'run.py'),run_name='__main__')
    return 0

if __name__=='__main__':raise SystemExit(main())
