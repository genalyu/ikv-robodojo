"""Accept mirrored pi05 weights only when each file matches official GCS metadata."""
import json
import shutil
from pathlib import Path
from download_pi05_base import PREFIX, ROOT, list_objects, valid
MIRROR = Path('/mnt/cfs/9wt59p/genalyu/robodojo-posttrain/pi05-hf-mirror/pi05_base')
def main():
    items = list_objects()
    if len(items) != 29:
        raise RuntimeError(f'unexpected official object count: {len(items)}')
    for item in items:
        rel = item['name'].removeprefix(PREFIX)
        target = ROOT / rel
        if valid(target, item):
            print('official', rel, flush=True)
            continue
        mirror = MIRROR / rel
        if not valid(mirror, item):
            raise RuntimeError(f'mirror does not match official MD5/size: {rel}')
        target.parent.mkdir(parents=True, exist_ok=True)
        target_tmp = target.with_name(target.name + '.verified_mirror_tmp')
        shutil.copyfile(mirror, target_tmp)
        if not valid(target_tmp, item):
            target_tmp.unlink(missing_ok=True)
            raise RuntimeError(f'copy changed bytes: {rel}')
        target_tmp.replace(target)
        print('verified mirror', rel, flush=True)
    manifest = [{key: item.get(key) for key in ('name','size','generation','md5Hash')} for item in items]
    (ROOT / 'source_manifest.json').write_text(json.dumps(manifest, indent=2))
    print('complete', flush=True)
if __name__ == '__main__':
    main()
