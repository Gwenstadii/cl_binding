"""Publish compact evidence in docs; retain full traces in the local/remote run archive."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

root = Path(__file__).resolve().parent
archive = Path(sys.argv[1])
report = json.loads((root / 'summary.json').read_text())
assert report['episodes'] == 340
target = root.parents[1] / 'docs/results/2026-11-07/k13_r1_front_20'
target.mkdir(parents=True, exist_ok=False)
code = target / 'code'
code.mkdir()
for name in ('summary.json',):
    shutil.copyfile(root / name, target / name)
for name in ('protocol.json', 'oracle_supplement.json', 'geometry.json', 'dependency_sha256.json'):
    shutil.copyfile(archive / name, target / name)
for name in ('r1_front_start.py', 'oracle_start.py', 'k11_reference.py'):
    shutil.copyfile(archive / name, code / name)
for name in ('summarize.py', 'export_results.py'):
    shutil.copyfile(root / name, code / name)
shutil.copytree(archive / 'dependencies', code / 'dependencies')
hashes, records = {}, []
for group in (archive, archive / 'oracle'):
    for init in range(20, 40):
        path = group / f'i{init}.jsonl'
        records.extend(json.loads(line) for line in path.read_text().splitlines())
        for filename in (f'i{init}.jsonl', f'i{init}_traces.jsonl', f'i{init}_COMPLETE.json'):
            path = group / filename
            hashes[str(path.relative_to(archive)).replace('\\', '/')] = hashlib.sha256(path.read_bytes()).hexdigest()
assert len(records) == 340
with (target / 'episodes.jsonl').open('w', encoding='utf-8') as f:
    for record in sorted(records, key=lambda r: (r['init'], r['condition'], r['goal'])):
        f.write(json.dumps(record, allow_nan=False) + '\n')
(target / 'artifacts.json').write_text(json.dumps({
    'remote_archive': '/home/lzx/cl_binding/.server/k13_r1_front_20_20261107_01',
    'local_archive': str(archive.resolve()), 'model_sha256': report['model_sha256'],
    'raw_file_sha256': hashes, 'episodes': 340, 'trace_steps': 136000,
}, indent=2), encoding='utf-8')
print(target)
