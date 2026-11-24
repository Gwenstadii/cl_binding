import argparse
from datetime import datetime,timezone
import importlib.metadata
from pathlib import Path
import sys
import whole4_new as c

run=Path(__file__).resolve().parent
a=argparse.ArgumentParser();a.add_argument('stage',choices=['start','final']);v=a.parse_args()
p=c.read(run/'preparation.json');old=c.read(c.REFERENCE/'four_protocol.json');prior=c.read(c.REFERENCE/'runtime.json')
files=dict(old['files'])
for step,m in p['candidates'].items():files[m['weights']]=m['sha256']
for n,h in p['sources'].items():files[str(run/n)]=h
for n,h in p['reference_sources'].items():files[str(c.REFERENCE/n)]=h
for rel,h in prior['dependency_sha256'].items():files[str(Path('/home/lzx')/rel)]=h
mods=[c,c.full,c.full.reference,c.ref,c.ref.old,c.ref.old.original,c.ref.old.original.k21,c.ref.old.timing,c.ref.old.spec,c.ref.old.spec.scoring]
files.update({str(Path(m.__file__).resolve()):c.digest(m.__file__) for m in mods})
for path,value in files.items():assert c.digest(path)==value,path
packages={n:importlib.metadata.version(n) for n in prior['packages']};assert packages==prior['packages']
out=dict(captured_at_utc=datetime.now(timezone.utc).isoformat(),files=files,packages=packages,python=sys.version,executable=sys.executable,
    source_sha256=c.digest(Path(__file__)),preparation_sha256=c.digest(run/'preparation.json'),stage=v.stage)
if v.stage=='final':
    original=c.read(run/'start_identity.json');assert original['files']==files and original['packages']==packages
    assert c.read(run/'COMPLETE.json')['episodes']==400
dest=run/(v.stage+'_identity.json');assert not dest.exists();c.write(dest,out)
print('NEW_IDENTITY_VERIFIED',v.stage,len(files),flush=True)
