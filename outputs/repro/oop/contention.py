import os, sys, time, threading, statistics, urllib.request
import numpy as np
mode=sys.argv[1]
if mode=='oop': os.environ['OCDKIT_TILESERVE_OOP']='1'
# default 5ms GIL switch interval to show the real in-process starvation
os.environ.setdefault('OCDKIT_TILESERVE_SWITCHINTERVAL','0')
from ocdkit.tileserve import server
url=server.ensure_server()
sid=server.register_pending(64,64,['x'],single_level=['x']); server.fill(sid,'x',np.zeros((64,64),np.float32))
iu=f"{url}/info/{sid}"
def ttfb(n):
    ts=[]
    for _ in range(n):
        t=time.perf_counter()
        try:
            with urllib.request.urlopen(iu,timeout=5) as r: r.read()
        except Exception: continue
        ts.append((time.perf_counter()-t)*1000)
    return statistics.median(ts) if ts else float('nan')
ttfb(5)
base=ttfb(60)
stop=threading.Event()
def busy():
    x=0
    while not stop.is_set():
        for _ in range(500000): x+=1
threading.Thread(target=busy,daemon=True).start(); time.sleep(0.3)
loaded=ttfb(60); stop.set()
print(f"{mode}: baseline={base:.2f}ms loaded={loaded:.2f}ms ratio={loaded/base:.1f}x")
server.reset_server()
