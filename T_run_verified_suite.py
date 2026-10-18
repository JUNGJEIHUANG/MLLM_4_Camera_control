"""Staged, resumable execution. Smoke results are not publication measurements."""
import argparse, collections, gc, hashlib, importlib.metadata, json, pathlib, random, sys, time, traceback
import torch
import server_vlm_experiment as e

def write_json(path, obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False));tmp.replace(path)

def balanced(rows, limit):
    buckets=collections.defaultdict(list)
    for row in rows:buckets[(row['task_id'],row['aperture'])].append(row)
    out=[]
    while len(out)<min(limit,len(rows)):
        # Interleave tasks/apertures so a four-row training smoke covers both.
        tasks=sorted({k[0] for k in buckets});aps=sorted({k[1] for k in buckets})
        order=[(tasks[i],aps[(i+j)%len(aps)]) for j in range(len(aps)) for i in range(len(tasks))]
        for key in order:
            if buckets[key] and len(out)<limit:out.append(buckets[key].pop(0))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','full'],default='smoke')
    p.add_argument('--sizes',nargs='+',default=['3B','7B']);p.add_argument('--wait-models',action='store_true')
    a=p.parse_args();root=e.ROOT;out=root/'audit_20261018'/a.mode;out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42);random.seed(42)
    train=[json.loads(x) for x in (root/'train_contracts.jsonl').read_text().splitlines() if x.strip()]
    test=[json.loads(x) for x in (root/'test_contracts.jsonl').read_text().splitlines() if x.strip()]
    if a.mode=='smoke':train=balanced(train,4);test=balanced(test,16)
    ledger=json.loads((root/'sCMOS_physical_ledger.json').read_text())
    manifest={'mode':a.mode,'train_samples':len(train),'test_samples':len(test),
        'train_ids':[r['sample_id'] for r in train],'test_ids':[r['sample_id'] for r in test],
        'gpu':torch.cuda.get_device_name(),'versions':{k:importlib.metadata.version(k) for k in ['torch','transformers','peft','accelerate']},
        'source_sha256':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [root/'server_vlm_experiment.py',root/'iscas_control/vlm_contract_compiler.py',pathlib.Path(__file__),root/'train_contracts.jsonl',root/'test_contracts.jsonl',root/'sCMOS_physical_ledger.json']},
        'limitations':['Same-scene and shared-image train/test; held-out instruction paraphrases only.',
            'Offline real-curve interpolation, not live camera execution.',
            'P1 uses oracle task ROI telemetry and nearest captured image.',
            'Confirmation times combine measured model generation with assumed 240ms/frame and 12ms/command.',
            'Task pass criteria are brightness/clipping proxies, not measured localization/OCR/SNR task accuracy.']}
    previous=out/'manifest.json'
    if previous.exists() and json.loads(previous.read_text())['source_sha256']!=manifest['source_sha256']:
        raise RuntimeError('Source/data changed since prior run; preserve and move the old output directory before rerunning.')
    write_json(previous,manifest)
    def stage(name,fn):
        path=out/f'{name}.json'
        if path.exists():print('RESUME',name,flush=True);return json.loads(path.read_text())
        write_json(out/'status.json',{'state':'running','stage':name,'updated_unix':time.time()})
        print('STAGE_START',name,flush=True);t=time.time();res=fn();res['stage_wall_seconds']=round(time.time()-t,3)
        write_json(path,res);print('STAGE_PASS',name,res['stage_wall_seconds'],flush=True);return res
    stage('fixed_center',lambda:e.evaluate_fixed_center_baseline(test,ledger))
    for size in a.sizes:
        model_path=None
        for attempt in range(720 if a.wait_models else 1):
            mf=root/'T_model_paths.json'
            paths=json.loads(mf.read_text()) if mf.exists() else {}
            if size in paths:model_path=paths[size];break
            if attempt%20==0:print('WAIT_MODEL',size,flush=True)
            time.sleep(10)
        if model_path is None:raise RuntimeError(f'Model {size} not available')
        model,processor=e.load_vlm(model_path)
        # Unscored warm-up; excludes the first CUDA compilation from timing.
        print('WARMUP',size,e.vlm_generate(model,processor,root/test[0]['images']['dual_offset_grid'],e.SYSTEM_CONTRACT_PROMPT,'Return a valid exposure contract.',max_new_tokens=8)[0],flush=True)
        stage(f'{size}_zeroshot',lambda:e.evaluate_contract_compilation(model,processor,test,ledger,model_tag=f'{size}-ZeroShot'))
        stage(f'{size}_direct',lambda:e.evaluate_direct_vlm_in_loop(model,processor,test,ledger,model_tag=f'{size}-P1'))
        if size=='3B':
            for repr_mode in ['linear','minmax','dual_offset_grid']:
                stage(f'3B_repr_{repr_mode}',lambda m=repr_mode:e.evaluate_contract_compilation(model,processor,test,ledger,repr_mode=m,include_telemetry=False,model_tag=f'3B-{m}-visual-only'))
        del model,processor;gc.collect();torch.cuda.empty_cache()
        if size=='3B':
            checkpoint=out/'adapter'
            if not (out/'lora_training.json').exists():
                write_json(out/'status.json',{'state':'running','stage':'lora_training','updated_unix':time.time()})
                model,processor,stats=e.train_contract_lora(model_path,train,checkpoint,epochs=1 if a.mode=='smoke' else 2)
                write_json(out/'lora_training.json',stats)
                del model,processor;gc.collect();torch.cuda.empty_cache()
            from peft import PeftModel
            base,processor=e.load_vlm(model_path)
            model=PeftModel.from_pretrained(base,str(checkpoint));model.eval()
            print('ADAPTER_RELOAD_PASS',str(checkpoint),flush=True)
            post=stage('3B_lora_telemetry',lambda:e.evaluate_contract_compilation(model,processor,test,ledger,model_tag='3B-LoRA'))
            stage('3B_lora_visual',lambda:e.evaluate_contract_compilation(model,processor,test,ledger,include_telemetry=False,model_tag='3B-LoRA-visual-only'))
            per_task={}
            for task in sorted({r['task_id'] for r in test}):
                records=[r for r in post['records'] if r['task_id']==task]
                per_task[task]={'episodes':len(records),'roi_accuracy':sum(r['roi_match'] for r in records)/len(records),
                    'policies':{pol:sum(r['policy_outcomes'][pol]['task_qualified'] for r in records)/len(records) for pol in records[0]['policy_outcomes']}}
            write_json(out/'experiment_2_by_task.json',per_task)
            del model,base,processor;gc.collect();torch.cuda.empty_cache()
    write_json(out/'status.json',{'state':'complete','sizes':a.sizes,'mode':a.mode,'updated_unix':time.time()})
    print('SUITE_PASS',a.mode,a.sizes,flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc()
        mode='full' if '--mode' in sys.argv and sys.argv[sys.argv.index('--mode')+1]=='full' else 'smoke'
        out=e.ROOT/'audit_20261018'/mode;out.mkdir(parents=True,exist_ok=True)
        old=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
        old.update(state='failed',traceback=traceback.format_exc(),updated_unix=time.time())
        write_json(out/'status.json',old);sys.exit(1)
