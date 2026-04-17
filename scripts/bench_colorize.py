"""Benchmark colorize: matmul vs opt_einsum across backends and shapes."""

import time, string
import numpy as np
import torch
from opt_einsum import contract


def make_colors(C):
    angle = np.linspace(0, 1, C, endpoint=False) * 2 * np.pi
    angles = np.stack((angle, angle + 2*np.pi/3, angle + 4*np.pi/3), axis=-1)
    return ((np.cos(angles) + 1) / 2).astype(np.float32)


def bench(fn, n_warmup=3, n_iter=10):
    for _ in range(n_warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1000


configs = [
    (126, (2000, 2000), [126],        '126ch 2kx2k single'),
    (126, (2000, 2000), [42,42,42],   '126ch 2kx2k 3-int'),
    (126, (10000, 100), [126],        '126ch 10kx100 single'),
    (256, (5000, 50),   [64]*4,       '256ch 5kx50 4-int'),
    (512, (1000, 1000), [512],        '512ch 1kx1k single'),
    (512, (1000, 1000), [128]*4,      '512ch 1kx1k 4-int'),
    (1024,(500, 500),   [1024],       '1024ch 500x500 single'),
    (126, (100, 100, 100),[126],      '126ch 100^3 single'),
    (256, (32, 32),     [256],        '256ch 32x32 single'),
    (3,   (4000, 4000), [3],          '3ch 4kx4k single'),
    (8,   (4000, 4000), [4,4],        '8ch 4kx4k 2-int'),
]

devices = ['cpu']
if torch.cuda.is_available():
    devices.append('cuda')
if torch.backends.mps.is_available():
    devices.append('mps')

for device_name in devices:
    device = torch.device(device_name)
    print(f"\n{'='*95}")
    print(f"  Device: {device_name.upper()}")
    print(f"{'='*95}")
    print(f"{'Config':<35} {'np mm':>8} {'np ein':>8} {'t mm':>8} {'t ein':>8}  np    torch")
    print('-' * 95)

    for C, spatial, intervals, label in configs:
        N = len(intervals)
        im_np = np.random.rand(C, *spatial).astype(np.float32)
        colors = make_colors(C)
        agg = np.zeros((C, N), dtype=np.float32)
        s = 0
        for i, sz in enumerate(intervals):
            agg[s:s+sz, i] = 1.0/sz
            s += sz

        idx = ''.join(c for c in string.ascii_lowercase if c not in 'cl')
        sp = idx[:len(spatial)]
        eq = f'c{sp},cN,cl->N{sp}l'

        # Numpy (always CPU)
        def np_mm():
            w = (agg[...,None]*colors[:,None,:]).reshape(C,N*3)
            return w.T @ im_np.reshape(C,-1)
        def np_ein():
            return contract(eq, im_np, agg, colors)

        t1 = bench(np_mm)
        t2 = bench(np_ein)

        # Torch on target device
        im_t = torch.from_numpy(im_np).to(device)
        agg_t = torch.from_numpy(agg).to(device)
        col_t = torch.from_numpy(colors).to(device)

        def t_mm():
            w = (agg_t[...,None]*col_t[:,None,:]).reshape(C,N*3).float()
            return w.T @ im_t.reshape(C,-1).float()
        def t_ein():
            return contract(eq, im_t.float(), agg_t, col_t)

        t3 = bench(t_mm)
        t4 = bench(t_ein)

        np_w = 'ein' if t2 < t1 else 'mm'
        t_w = 'ein' if t4 < t3 else 'mm'
        np_r = max(t1,t2)/min(t1,t2)
        t_r = max(t3,t4)/min(t3,t4)

        print(f'{label:<35} {t1:>6.1f}ms {t2:>6.1f}ms {t3:>6.1f}ms {t4:>6.1f}ms  {np_w}({np_r:.1f}x) {t_w}({t_r:.1f}x)')

        # Free GPU memory
        del im_t, agg_t, col_t
        if device_name != 'cpu':
            torch.cuda.empty_cache() if device_name == 'cuda' else torch.mps.empty_cache()
