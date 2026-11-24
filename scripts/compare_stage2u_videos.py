"""Create all 12 paired videos; no simulation or policy calls."""
import argparse
import hashlib
import json
from pathlib import Path


def main(root,output):
    import av
    import numpy as np
    assert not output.exists();output.mkdir(parents=True)
    eps=[json.loads(x) for x in (root/'evaluation/episodes.jsonl').read_text().splitlines()]
    lookup={(e['mode'],e['requested_goal'],e['init_index'],e['noise_seed']):e for e in eps}
    assert json.loads((root/'audit.json').read_text())['status']=='passed'
    results=[]
    for goal in ('left','front','right'):
        for index in (0,1):
            for seed in (2026102801,2026102802):
                a,b=[root/'evaluation/videos'/lookup[m,goal,index,seed]['video'] for m in ('native','handoff60')]
                name=f'{goal.upper()}_FT4_NATIVE_vs_R0_60_FT4_init{index:02d}_seed{seed}.mp4'
                count=0
                with av.open(str(a)) as va,av.open(str(b)) as vb,av.open(str(output/name),'w') as out:
                    stream=out.add_stream('libx264',rate=20);stream.width=1536;stream.height=492;stream.pix_fmt='yuv420p'
                    stream.options={'crf':'22','preset':'veryfast'}
                    for fa,fb in zip(va.decode(video=0),vb.decode(video=0),strict=True):
                        pixels=np.concatenate([fa.to_ndarray(format='rgb24'),fb.to_ndarray(format='rgb24')],axis=1)
                        for packet in stream.encode(av.VideoFrame.from_ndarray(pixels,format='rgb24')):out.mux(packet)
                        count+=1
                    for packet in stream.encode():out.mux(packet)
                assert count==400
                with av.open(str(output/name)) as v:decoded=sum(1 for _ in v.decode(video=0))
                assert decoded==400
                row={'file':name,'requested_goal':goal,'init_index':index,'noise_seed':seed,'frames':count,
                     'decoded_frames':decoded,'left_native_source':a.name,'right_handoff_source':b.name,
                     'sha256':hashlib.sha256((output/name).read_bytes()).hexdigest(),'bytes':(output/name).stat().st_size}
                results.append(row);print(json.dumps(row),flush=True)
    (output/'comparison_manifest.json').write_text(json.dumps({'status':'passed','videos':results},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.root,a.output)
