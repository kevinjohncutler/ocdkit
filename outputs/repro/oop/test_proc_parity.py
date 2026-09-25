import socket, time, urllib.request
import numpy as np
from ocdkit.tileserve import _proc

s=socket.socket(); s.bind(("127.0.0.1",0)); port=s.getsockname()[1]; s.close()
arr=(np.random.rand(64,48)*100).astype(np.float32)
ref=np.ascontiguousarray(arr).tobytes()

client=_proc.spawn(port, [])
try:
    sid,label=client.call("register_array",(arr,),{"single_level":True,"n_levels":1})
    url=f"http://127.0.0.1:{port}/tile/{sid}/{label}/99?fmt=raw"
    body=None
    for _ in range(30):
        try:
            with urllib.request.urlopen(url,timeout=2) as r:
                if r.status==200: body=r.read(); break
        except Exception: pass
        time.sleep(0.1)
    assert body is not None, "no tile body from child"
    assert body==ref, f"BYTE MISMATCH child={len(body)} ref={len(ref)}"
    # /info too
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/info/{sid}",timeout=2) as r:
        import json; info=json.loads(r.read())
    print("PARITY OK | tile bytes:",len(body),"==",len(ref),"| info layers:",list(info.get("layers",{})))
    print("child alive:",client.alive())
finally:
    client.close()
