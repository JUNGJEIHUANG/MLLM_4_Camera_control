"""Compile one contract with the released adapter. Does not actuate camera hardware."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base-model',default='Qwen/Qwen2.5-VL-3B-Instruct',help='Official Hub ID or downloaded local base-model directory.')
    p.add_argument('--adapter',type=Path,default=ROOT/'checkpoints/qwen25_vl_3b_contract_lora_full')
    p.add_argument('--image',required=True,type=Path)
    p.add_argument('--instruction',required=True)
    p.add_argument('--telemetry',default='')
    a=p.parse_args()
    import torch
    if not torch.cuda.is_available():p.error('This audited BF16 inference path requires a CUDA GPU.')
    from peft import PeftModel
    import server_vlm_experiment as e
    base,processor=e.load_vlm(a.base_model)
    # An explicit base model avoids the training-machine path in adapter_config.json.
    model=PeftModel.from_pretrained(base,str(a.adapter));model.eval()
    prompt=f'Task Instruction: {a.instruction}'
    if a.telemetry:prompt+=f'\nScene Telemetry: {a.telemetry}'
    else:prompt+='\nReturn one compact exposure contract. No telemetry is available.'
    raw,ms=e.vlm_generate(model,processor,a.image,e.SYSTEM_CONTRACT_PROMPT,prompt,max_new_tokens=192)
    try:
        import re
        match=re.search(r'\{.*\}',raw,flags=re.S)
        payload=json.loads(match.group(0) if match else raw)
        e.validate_raw_contract(payload);valid=True;error=None
    except Exception as exc:valid=False;error=str(exc)
    print(json.dumps({'result_scope':'same-scene research adapter; new scenes are not validated',
        'camera_actuation':False,'raw_output':raw,'schema_valid':valid,'validation_error':error,
        'generation_ms':round(ms,2)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
