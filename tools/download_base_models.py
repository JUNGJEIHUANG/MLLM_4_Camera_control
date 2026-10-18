"""Download official base models into a user-selected cache; no weights are mirrored here."""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sizes',nargs='+',choices=['3B','7B'],default=['3B'])
    p.add_argument('--cache-dir',type=Path,default=Path('models/base'))
    a=p.parse_args()
    from modelscope import snapshot_download
    root=Path(__file__).resolve().parents[1]
    manifest=root/'T_model_paths.json'
    paths=json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else {}
    for size in a.sizes:
        paths[size]=str(Path(snapshot_download(f'Qwen/Qwen2.5-VL-{size}-Instruct',cache_dir=str(a.cache_dir.resolve()))).resolve())
    manifest.write_text(json.dumps(paths,indent=2),encoding='utf-8')
    print('Saved local model paths to',manifest)

if __name__=='__main__':main()
