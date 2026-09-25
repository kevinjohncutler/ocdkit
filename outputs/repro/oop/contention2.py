import os, sys, time, threading, subprocess
import numpy as np
mode=sys.argv[1]
if mode=='oop': os.environ['OCDKIT_TILESERVE_OOP']='1'
os.environ.setdefault('OCDKIT_TILESERVE_SWITCHINTERVAL','0')  # default 5ms: full starvation
from ocdkit.tileserve import server
url=server.ensure_server()
sid=server.register_pending(64,64,['x'],single_level=['x']); server.fill(sid,'x',np.zeros((64,64),np.float32))
HERE=os.path.dirname(__file__); CLI=[sys.executable, os.path.join(HERE,'_client.py'), url, sid, '60']
def client(): return float(subprocess.run(CLI,capture_output=True,text=True).stdout.strip() or 'nan')
client()  # warm
base=client()
stop=threading.Event()
def busy():
    x=0
    while not stop.is_set():
        for _ in range(500000): x+=1
threading.Thread(target=busy,daemon=True).start(); time.sleep(0.3)
loaded=client(); stop.set()
print(f"{mode}: baseline={base:.2f}ms loaded={loaded:.2f}ms ratio={loaded/base:.1f}x  (client = separate process, like the browser)")
server.reset_server()
