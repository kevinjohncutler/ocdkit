import sys, time, statistics, urllib.request
url, sid, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
iu=f"{url}/info/{sid}"; ts=[]
for _ in range(n):
    t=time.perf_counter()
    try:
        with urllib.request.urlopen(iu,timeout=5) as r: r.read()
    except Exception: continue
    ts.append((time.perf_counter()-t)*1000)
print(f"{statistics.median(ts):.2f}" if ts else "nan")
