"""Tests for ocdkit.array — verify numpy/torch/dask parity."""

import numpy as np
import pytest
import torch
import dask.array as da

from ocdkit.array import (
    get_module,
    safe_divide,
    rescale,
    is_integer,
    to_torch,
    parallel_reduce,
    parallel_copy,
    unique_nonzero,
    torch_norm,
    normalize99,
    normalize_field,
    meshgrid,
    generate_flat_coordinates,
    divergence,
    searchsorted,
    quantile_rescale,
    normalize99_hist,
    qnorm,
    localnormalize,
    pnormalize,
    normalize_image,
    adjust_contrast_masked,
    gamma_normalize,
    to_8_bit,
    to_16_bit,
    get_size,
    random_int,
    ravel_index,
    unravel_index,
    border_indices,
    split_array,
    reconstruct_array,
    enumerate_nested,
)


# ---------------------------------------------------------------------------
# get_module
# ---------------------------------------------------------------------------

class TestGetModule:
    def test_numpy(self):
        assert get_module(np.array([1.0])) is np

    def test_torch(self):
        assert get_module(torch.tensor([1.0])) is torch

    def test_dask(self):
        assert get_module(da.from_array(np.array([1.0]))) is np

    def test_scalar(self):
        assert get_module(1.0) is np
        assert get_module(3) is np

    def test_tuple(self):
        assert get_module((1, 2)) is np


# ---------------------------------------------------------------------------
# safe_divide — parity across backends
# ---------------------------------------------------------------------------

class TestSafeDivide:
    def test_numpy_basic(self):
        num = np.array([1.0, 2.0, 3.0])
        den = np.array([2.0, 0.0, 1.0])
        r = safe_divide(num, den)
        assert np.allclose(r, [0.5, 0.0, 3.0])

    def test_torch_basic(self):
        num = torch.tensor([1.0, 2.0, 3.0])
        den = torch.tensor([2.0, 0.0, 1.0])
        r = safe_divide(num, den)
        np.testing.assert_allclose(r.numpy(), [0.5, 0.0, 3.0], atol=1e-6)

    def test_dask_basic(self):
        num = da.from_array(np.array([1.0, 2.0, 3.0]))
        den = da.from_array(np.array([2.0, 0.0, 1.0]))
        r = safe_divide(num, den).compute()
        np.testing.assert_allclose(r, [0.5, 0.0, 3.0], atol=1e-6)

    def test_parity(self):
        np.random.seed(0)
        num_np = np.random.rand(10).astype(np.float32)
        den_np = np.random.rand(10).astype(np.float32)
        den_np[3] = 0.0  # zero
        den_np[7] = np.nan  # nan

        r_np = safe_divide(num_np, den_np)
        r_torch = safe_divide(torch.from_numpy(num_np), torch.from_numpy(den_np)).numpy()
        r_dask = safe_divide(da.from_array(num_np), da.from_array(den_np)).compute()

        # All should agree on zero outputs where den is invalid
        assert r_np[3] == 0.0
        assert r_torch[3] == 0.0
        assert r_dask[3] == 0.0
        assert r_np[7] == 0.0
        assert r_dask[7] == 0.0

    def test_cutoff(self):
        num = np.array([1.0, 1.0])
        den = np.array([0.5, 2.0])
        r = safe_divide(num, den, cutoff=1.0)
        assert r[0] == 0.0  # den=0.5 < cutoff=1.0
        assert np.isclose(r[1], 0.5)


# ---------------------------------------------------------------------------
# rescale — parity
# ---------------------------------------------------------------------------

class TestRescale:
    def test_numpy(self):
        x = np.array([0.0, 5.0, 10.0])
        r = rescale(x)
        np.testing.assert_allclose(r, [0.0, 0.5, 1.0])

    def test_torch(self):
        x = torch.tensor([0.0, 5.0, 10.0])
        r = rescale(x)
        np.testing.assert_allclose(r.numpy(), [0.0, 0.5, 1.0], atol=1e-6)

    def test_parity(self):
        np.random.seed(1)
        x = np.random.rand(4, 4).astype(np.float32)
        r_np = rescale(x)
        r_torch = rescale(torch.from_numpy(x)).numpy()
        np.testing.assert_allclose(r_np, r_torch, atol=1e-5)

    def test_exclude_dims(self):
        x = np.array([[0.0, 10.0], [0.0, 5.0]])
        r = rescale(x, exclude_dims=0)
        # Each row rescaled independently
        np.testing.assert_allclose(r[0], [0.0, 1.0])
        np.testing.assert_allclose(r[1], [0.0, 1.0])


# ---------------------------------------------------------------------------
# torch_norm — parity
# ---------------------------------------------------------------------------

class TestTorchNorm:
    def test_numpy(self):
        x = np.array([[3.0, 0.0], [4.0, 1.0]])
        mag = torch_norm(x, dim=0)
        np.testing.assert_allclose(mag, [5.0, 1.0])

    def test_torch(self):
        x = torch.tensor([[3.0, 0.0], [4.0, 1.0]])
        mag = torch_norm(x, dim=0)
        np.testing.assert_allclose(mag.numpy(), [5.0, 1.0])

    def test_parity(self):
        np.random.seed(2)
        x = np.random.rand(3, 8, 8).astype(np.float32)
        mag_np = torch_norm(x, dim=0)
        mag_torch = torch_norm(torch.from_numpy(x), dim=0).numpy()
        np.testing.assert_allclose(mag_np, mag_torch, atol=1e-6)

    def test_keepdim(self):
        x = torch.rand(2, 4, 4)
        mag = torch_norm(x, dim=0, keepdim=True)
        assert mag.shape == (1, 4, 4)


# ---------------------------------------------------------------------------
# normalize99 — parity
# ---------------------------------------------------------------------------

class TestNormalize99:
    def test_numpy_basic(self):
        x = np.arange(100, dtype=np.float32)
        r = normalize99(x)
        assert r.min() >= 0.0
        assert r.max() <= 1.0

    def test_torch_basic(self):
        x = torch.arange(100, dtype=torch.float32)
        r = normalize99(x)
        assert r.min() >= 0.0
        assert r.max() <= 1.0

    def test_parity(self):
        np.random.seed(3)
        x = np.random.rand(64).astype(np.float32)
        r_np = normalize99(x)
        r_torch = normalize99(torch.from_numpy(x)).numpy()
        np.testing.assert_allclose(r_np, r_torch, atol=1e-5)

    def test_contrast_limits(self):
        x = np.arange(10, dtype=np.float32)
        r = normalize99(x, contrast_limits=(2.0, 8.0))
        assert np.isclose(r[2], 0.0)
        assert np.isclose(r[8], 1.0)

    def test_dim(self):
        x = np.random.rand(3, 10).astype(np.float32)
        r = normalize99(x, dim=0)
        assert r.shape == x.shape
        # Each row should have values spanning [0, 1]
        for i in range(3):
            assert r[i].min() >= 0.0
            assert r[i].max() <= 1.0

    def test_kwargs_ignored(self):
        """Extra kwargs (like omni=True) should not raise."""
        x = np.arange(10, dtype=np.float32)
        r = normalize99(x, omni=True)
        assert r.shape == x.shape


# ---------------------------------------------------------------------------
# normalize_field — parity
# ---------------------------------------------------------------------------

class TestNormalizeField:
    def test_numpy(self):
        mu = np.array([[3.0, 0.0], [4.0, 0.0]])  # (2, 2) flow field
        r = normalize_field(mu)
        mag = np.sqrt(np.sum(r**2, axis=0))
        # First column had magnitude 5, should be 1 now
        np.testing.assert_allclose(mag[0], 1.0, atol=1e-6)
        # Second column was zero, should stay zero
        assert r[0, 1] == 0.0 and r[1, 1] == 0.0

    def test_torch(self):
        mu = torch.tensor([[3.0, 0.0], [4.0, 0.0]])
        r = normalize_field(mu)
        mag = torch.sqrt((r**2).sum(dim=0))
        np.testing.assert_allclose(mag[0].item(), 1.0, atol=1e-6)
        assert r[0, 1] == 0.0 and r[1, 1] == 0.0

    def test_parity(self):
        np.random.seed(4)
        mu = np.random.rand(2, 8, 8).astype(np.float32)
        r_np = normalize_field(mu)
        r_torch = normalize_field(torch.from_numpy(mu)).numpy()
        np.testing.assert_allclose(r_np, r_torch, atol=1e-5)

    def test_cutoff(self):
        mu = np.array([[0.01, 3.0], [0.01, 4.0]])
        r = normalize_field(mu, cutoff=0.1)
        # First column magnitude ~0.014 < cutoff, should be unchanged
        np.testing.assert_allclose(r[:, 0], mu[:, 0], atol=1e-6)
        # Second column should be normalized
        mag = np.sqrt(np.sum(r[:, 1]**2))
        np.testing.assert_allclose(mag, 1.0, atol=1e-6)

    def test_use_torch_kwarg_accepted(self):
        """Backward compat: use_torch= should not raise."""
        mu = torch.rand(2, 4, 4)
        r = normalize_field(mu, use_torch=True, cutoff=0.0)
        assert r.shape == mu.shape


# ---------------------------------------------------------------------------
# is_integer
# ---------------------------------------------------------------------------

class TestIsInteger:
    def test_python_int(self):
        assert is_integer(5) is True

    def test_numpy_int(self):
        assert is_integer(np.int32(5)) is True

    def test_numpy_array_int(self):
        assert is_integer(np.array([1, 2], dtype=np.int64)) is True

    def test_numpy_array_float(self):
        assert is_integer(np.array([1.0])) is False

    def test_torch_int(self):
        assert is_integer(torch.tensor([1], dtype=torch.int32)) is True

    def test_torch_float(self):
        assert is_integer(torch.tensor([1.0])) is False

    def test_dask_int(self):
        assert is_integer(da.from_array(np.array([1], dtype=np.int32))) is True

    def test_float(self):
        assert is_integer(1.5) is False


# ---------------------------------------------------------------------------
# to_torch — uniform numpy/dask/torch/scalar conversion
# ---------------------------------------------------------------------------

class TestToTorch:
    def test_numpy_roundtrip(self):
        x = np.arange(12, dtype=np.float32).reshape(3, 4)
        t = to_torch(x)
        assert torch.is_tensor(t)
        np.testing.assert_array_equal(t.numpy(), x)

    def test_numpy_dtype_cast(self):
        x = np.arange(6, dtype=np.uint16).reshape(2, 3)
        t = to_torch(x, dtype=torch.float32)
        assert t.dtype == torch.float32
        np.testing.assert_allclose(t.numpy(), x.astype(np.float32))

    def test_numpy_non_contiguous(self):
        # Non-contiguous slice — torch.from_numpy alone would still work but
        # to_torch should also accept it without raising.
        x = np.arange(20, dtype=np.float32).reshape(4, 5)[:, ::2]
        assert not x.flags["C_CONTIGUOUS"]
        t = to_torch(x)
        np.testing.assert_array_equal(t.numpy(), x)

    def test_dask_materialized(self):
        x_np = np.random.randint(0, 100, size=(5, 6), dtype=np.uint16)
        x_dk = da.from_array(x_np, chunks=(5, 3))
        t = to_torch(x_dk, dtype=torch.float32)
        assert torch.is_tensor(t)
        assert t.dtype == torch.float32
        np.testing.assert_allclose(t.numpy(), x_np.astype(np.float32))

    def test_torch_passthrough(self):
        x = torch.arange(10, dtype=torch.float32)
        t = to_torch(x)
        assert t is x or torch.equal(t, x)

    def test_torch_dtype_change(self):
        x = torch.arange(4, dtype=torch.int64)
        t = to_torch(x, dtype=torch.float32)
        assert t.dtype == torch.float32

    def test_python_list(self):
        t = to_torch([1.0, 2.0, 3.0], dtype=torch.float32)
        np.testing.assert_array_equal(t.numpy(), [1.0, 2.0, 3.0])

    def test_python_scalar(self):
        t = to_torch(3.14, dtype=torch.float32)
        assert torch.is_tensor(t)
        assert t.ndim == 0

    def test_memmap(self, tmp_path):
        path = tmp_path / "x.npy"
        arr = np.arange(20, dtype=np.uint16).reshape(4, 5)
        np.save(path, arr)
        mm = np.load(path, mmap_mode="r")
        assert isinstance(mm, np.memmap)
        t = to_torch(mm, dtype=torch.float32)
        np.testing.assert_allclose(t.numpy(), arr.astype(np.float32))


# ---------------------------------------------------------------------------
# parallel_reduce — threaded numpy reductions
# ---------------------------------------------------------------------------

class TestParallelReduce:
    @pytest.mark.parametrize("op", ["sum", "mean", "max", "min"])
    @pytest.mark.parametrize("axis", [0, 1, 2, -1])
    def test_parity_with_numpy(self, op, axis):
        np.random.seed(0)
        arr = np.random.randint(0, 1000, size=(40, 60, 80), dtype=np.uint16)
        kw = {"dtype": np.float32} if op in ("sum", "mean") else {}
        ref = getattr(np, op)(arr, axis=axis, **kw)
        got = parallel_reduce(arr, op, axis=axis, dtype=kw.get("dtype"))
        np.testing.assert_allclose(got, ref, rtol=1e-5)
        assert got.shape == ref.shape

    def test_memmap_input(self, tmp_path):
        arr = np.random.randint(0, 65535, size=(20, 50, 50), dtype=np.uint16)
        p = tmp_path / "x.npy"
        np.save(p, arr)
        mm = np.load(p, mmap_mode="r")
        ref = np.sum(arr, axis=0, dtype=np.float32)
        got = parallel_reduce(mm, "sum", axis=0, dtype=np.float32)
        np.testing.assert_allclose(got, ref, rtol=1e-5)

    def test_callable_op(self):
        arr = np.random.rand(10, 20, 30).astype(np.float32)
        ref = np.std(arr, axis=0)
        got = parallel_reduce(arr, lambda a, axis, dtype=None: np.std(a, axis=axis), axis=0)
        np.testing.assert_allclose(got, ref, rtol=1e-5)

    def test_unknown_op(self):
        arr = np.zeros((4, 4))
        with pytest.raises(ValueError, match="unsupported op"):
            parallel_reduce(arr, "argmax", axis=0)

    def test_tiny_array_uses_singlethread_path(self):
        # Below 1M elements should still work and match np exactly.
        arr = np.random.randint(0, 100, size=(5, 5), dtype=np.uint16)
        got = parallel_reduce(arr, "sum", axis=0, dtype=np.float32)
        np.testing.assert_allclose(got, arr.sum(axis=0, dtype=np.float32))

    def test_n_threads_capped(self):
        arr = np.ones((100, 100, 100), dtype=np.uint16)
        # Asking for 1000 threads on a 100-row split-axis should be fine.
        got = parallel_reduce(arr, "sum", axis=0, n_threads=1000, dtype=np.float32)
        np.testing.assert_allclose(got, arr.sum(axis=0, dtype=np.float32))


# ---------------------------------------------------------------------------
# parallel_copy
# ---------------------------------------------------------------------------

class TestParallelCopy:
    def test_parity_with_np_array(self):
        arr = np.random.randint(0, 65535, size=(40, 200, 200), dtype=np.uint16)
        out = parallel_copy(arr)
        np.testing.assert_array_equal(out, arr)
        assert out.dtype == arr.dtype
        assert out.shape == arr.shape

    def test_returns_fresh_buffer(self):
        arr = np.random.rand(50, 100, 100).astype(np.float32)
        out = parallel_copy(arr)
        assert out is not arr
        assert out.flags["C_CONTIGUOUS"]
        # Mutating out must not affect arr.
        out[0, 0, 0] = -999
        assert arr[0, 0, 0] != -999

    def test_memmap_materialized(self, tmp_path):
        arr = np.random.randint(0, 65535, size=(50, 200, 200), dtype=np.uint16)
        p = tmp_path / "x.npy"
        np.save(p, arr)
        mm = np.load(p, mmap_mode="r")
        out = parallel_copy(mm)
        # Output is a regular ndarray, not a memmap.
        assert not isinstance(out, np.memmap)
        np.testing.assert_array_equal(out, arr)

    def test_tiny_array_uses_np_array(self):
        arr = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        out = parallel_copy(arr)
        np.testing.assert_array_equal(out, arr)

    def test_1d_array(self):
        arr = np.random.rand(2_000_000).astype(np.float32)
        out = parallel_copy(arr)
        np.testing.assert_array_equal(out, arr)


# ---------------------------------------------------------------------------
# unique_nonzero
# ---------------------------------------------------------------------------

class TestUniqueNonzero:
    def test_basic(self):
        arr = np.array([0, 3, 1, 0, 3, 2])
        np.testing.assert_array_equal(unique_nonzero(arr), [1, 2, 3])

    def test_all_zero(self):
        assert len(unique_nonzero(np.array([0, 0, 0]))) == 0


# ---------------------------------------------------------------------------
# meshgrid / generate_flat_coordinates
# ---------------------------------------------------------------------------

class TestMeshgrid:
    def test_2d(self):
        yy, xx = meshgrid((3, 4))
        assert yy.shape == (3, 4) and xx.shape == (3, 4)
        # ij indexing — yy varies along axis 0
        np.testing.assert_array_equal(yy[:, 0], [0, 1, 2])
        np.testing.assert_array_equal(xx[0, :], [0, 1, 2, 3])

    def test_3d(self):
        zz, yy, xx = meshgrid((2, 3, 4))
        assert zz.shape == yy.shape == xx.shape == (2, 3, 4)
        assert zz[1, 0, 0] == 1
        assert yy[0, 2, 0] == 2
        assert xx[0, 0, 3] == 3

    def test_matches_np_meshgrid(self):
        ours = meshgrid((5, 7))
        theirs = np.meshgrid(np.arange(5), np.arange(7), indexing='ij')
        for a, b in zip(ours, theirs):
            np.testing.assert_array_equal(a, b)


class TestGenerateFlatCoordinates:
    def test_2d(self):
        flat_y, flat_x = generate_flat_coordinates((2, 3))
        # ij order: (0,0),(0,1),(0,2),(1,0),(1,1),(1,2)
        np.testing.assert_array_equal(flat_y, [0, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(flat_x, [0, 1, 2, 0, 1, 2])

    def test_consistent_with_meshgrid(self):
        shape = (4, 5)
        grids = meshgrid(shape)
        flats = generate_flat_coordinates(shape)
        for g, f in zip(grids, flats):
            np.testing.assert_array_equal(g.ravel(), f)


# ---------------------------------------------------------------------------
# divergence
# ---------------------------------------------------------------------------

class TestDivergence:
    def test_numpy_2d_uniform(self):
        # Constant field has zero divergence
        f = np.ones((2, 5, 5), dtype=np.float32)
        div = divergence(f)
        assert div.shape == (5, 5)
        np.testing.assert_allclose(div, 0.0, atol=1e-6)

    def test_numpy_2d_radial(self):
        # f = (y, x) → divergence = du/dy + dv/dx = 1 + 1 = 2 (interior)
        ys, xs = np.meshgrid(np.arange(8), np.arange(8), indexing='ij')
        f = np.stack([ys.astype(np.float32), xs.astype(np.float32)])
        div = divergence(f)
        # Interior: du/dy = 1, dv/dx = 1 → div = 2
        np.testing.assert_allclose(div[3:5, 3:5], 2.0, atol=1e-6)

    def test_numpy_degenerate_axis(self):
        f = np.ones((2, 1, 5), dtype=np.float32)  # spatial dim 0 has size < 2
        div = divergence(f)
        np.testing.assert_array_equal(div, np.zeros((1, 5)))

    def test_torch_2d(self):
        # Batched torch input: (B=1, D=2, H, W)
        ys, xs = np.meshgrid(np.arange(8), np.arange(8), indexing='ij')
        f_np = np.stack([ys.astype(np.float32), xs.astype(np.float32)])
        f = torch.from_numpy(f_np).unsqueeze(0)
        div = divergence(f)
        assert div.shape == (1, 8, 8)
        np.testing.assert_allclose(div[0, 3:5, 3:5].numpy(), 2.0, atol=1e-6)

    def test_torch_degenerate(self):
        f = torch.ones((1, 2, 1, 5))
        div = divergence(f)
        assert div.shape == (1, 1, 5)
        assert torch.all(div == 0)

    def test_torch_3d(self):
        f = torch.zeros((2, 3, 4, 4, 4))  # B=2, D=3, 4x4x4
        div = divergence(f)
        assert div.shape == (2, 4, 4, 4)


# ---------------------------------------------------------------------------
# searchsorted
# ---------------------------------------------------------------------------

class TestSearchsorted:
    def test_numpy(self):
        arr = np.array([0.1, 0.3, 0.5, 0.7])
        assert searchsorted(arr, 0.4) == 2
        assert searchsorted(arr, 0.0) == 0
        assert searchsorted(arr, 1.0) == 4

    def test_torch(self):
        arr = torch.tensor([0.1, 0.3, 0.5, 0.7])
        assert int(searchsorted(arr, 0.4)) == 2


# ---------------------------------------------------------------------------
# quantile_rescale
# ---------------------------------------------------------------------------

class TestQuantileRescale:
    def test_basic(self):
        Y = np.linspace(0, 100, 1000, dtype=np.float32)
        r = quantile_rescale(Y, lower=0.0, upper=1.0)
        assert r.min() == 0.0
        assert r.max() == 1.0

    def test_clipping(self):
        Y = np.linspace(0, 100, 100, dtype=np.float32)
        r = quantile_rescale(Y, lower=0.1, upper=0.9)
        # Tail values should be clipped to 0/1
        assert (r == 0).any()
        assert (r == 1).any()


# ---------------------------------------------------------------------------
# normalize99_hist
# ---------------------------------------------------------------------------

class TestNormalize99Hist:
    def test_numpy_bounds(self):
        x = np.linspace(0, 100, 101, dtype=np.float32)
        r = normalize99_hist(x, lower=0, upper=100)
        assert r.min() == 0.0
        assert r.max() == 1.0

    def test_contrast_limits_numpy(self):
        x = np.arange(10, dtype=np.float32)
        r = normalize99_hist(x, contrast_limits=(0.0, 9.0))
        assert r.min() >= 0.0
        assert r.max() <= 1.0

    def test_contrast_limits_torch(self):
        t = torch.linspace(0, 1, 10)
        r = normalize99_hist(t, contrast_limits=(0.0, 1.0))
        assert r.min() >= 0.0
        assert r.max() <= 1.0


# ---------------------------------------------------------------------------
# qnorm
# ---------------------------------------------------------------------------

class TestQnorm:
    def test_debug_branch(self):
        rng = np.random.RandomState(0)
        Y = rng.rand(4, 8, 8).astype(np.float32)
        r, x, y, d, imin, imax, vmin, vmax = qnorm(
            Y,
            nbins=16,
            dx=2,
            log=True,
            debug=True,
            density_quantile=0.5,
            density_cutoff=0.5,
        )
        assert r.shape == (4, 4, 4)
        assert imin <= imax


# ---------------------------------------------------------------------------
# localnormalize
# ---------------------------------------------------------------------------

class TestLocalnormalize:
    def test_numpy(self):
        img = np.random.RandomState(0).rand(16, 16).astype(np.float32)
        out = localnormalize(img, sigma1=1, sigma2=2)
        assert out.shape == img.shape
        assert np.isfinite(out).all()

    def test_torch(self):
        pytest.importorskip("torchvision")
        img = torch.rand(1, 16, 16)
        out = localnormalize(img, sigma1=1, sigma2=2)
        assert out.shape == img.shape
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# pnormalize
# ---------------------------------------------------------------------------

class TestPnormalize:
    def test_numpy_range(self):
        arr = np.linspace(0, 5, 6, dtype=np.float32)
        out = pnormalize(arr, p_min=-1, p_max=2)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_torch(self):
        arr = torch.linspace(1, 5, 5)
        out = pnormalize(arr, p_min=-1, p_max=2)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


# ---------------------------------------------------------------------------
# normalize_image
# ---------------------------------------------------------------------------

class TestNormalizeImage:
    def test_basic(self):
        rng = np.random.RandomState(1)
        img = rng.uniform(0.1, 2.0, size=(8, 8)).astype(np.float32)
        mask = np.ones_like(img, dtype=np.uint8)
        out = normalize_image(img, mask, target=0.5, foreground=True, iterations=0)
        assert out.shape == img.shape
        assert np.isfinite(out).all()

    def test_with_erosion(self):
        img = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
        mask = np.ones((3, 3), dtype=np.uint8)
        out = normalize_image(img, mask, iterations=1, channel_axis=0)
        assert out.shape == img.shape


# ---------------------------------------------------------------------------
# adjust_contrast_masked
# ---------------------------------------------------------------------------

class TestAdjustContrastMasked:
    def test_basic(self):
        img = np.zeros((8, 8), dtype=np.float32)
        img[2:6, 2:6] = 1.0
        masks = np.zeros_like(img, dtype=np.uint8)
        masks[2:6, 2:6] = 1
        out, gamma, limits = adjust_contrast_masked(img, masks)
        assert out.shape == img.shape
        assert 0.2 <= gamma <= 5.0
        assert np.isfinite(out).all()
        assert limits[0] <= limits[1]

    def test_empty_mask(self):
        img = np.zeros((4, 4), dtype=np.float32)
        masks = np.zeros_like(img, dtype=np.uint8)
        out, gamma, _ = adjust_contrast_masked(img, masks)
        assert gamma == 1.0

    def test_gamma_path(self):
        img = np.ones((4, 4), dtype=np.float32)
        masks = np.zeros_like(img, dtype=np.uint8)
        masks[0:2, 0:2] = 1
        out, gamma, _ = adjust_contrast_masked(img, masks, r_target=0.5)
        assert 0.2 <= gamma <= 5.0
        assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# gamma_normalize
# ---------------------------------------------------------------------------

class TestGammaNormalize:
    def test_basic(self):
        img = np.zeros((1, 8, 8), dtype=np.float32)
        vals = np.linspace(0.2, 1.0, 16, dtype=np.float32).reshape(4, 4)
        img[:, 2:6, 2:6] = vals
        mask = (img > 0).astype(np.uint8)
        out = gamma_normalize(
            img,
            mask,
            target=torch.tensor(0.5),
            foreground=True,
            iterations=0,
            channel_axis=0,
        )
        assert out.shape == img.shape[1:]
        assert np.isfinite(out[mask[0] > 0]).all()

    def test_with_erosion(self):
        img = np.linspace(0.1, 1.0, 9, dtype=np.float32).reshape(3, 3)
        mask = (img > 0.5).astype(np.uint8)
        out = gamma_normalize(
            img, mask, iterations=1, channel_axis=0, target=torch.tensor(1.0)
        )
        assert out.shape == img.shape


# ---------------------------------------------------------------------------
# to_8_bit / to_16_bit
# ---------------------------------------------------------------------------

class TestBitConversions:
    def test_ranges(self):
        im = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        out8 = to_8_bit(im)
        out16 = to_16_bit(im)
        assert out8.dtype == np.uint8
        assert out16.dtype == np.uint16
        assert out8.min() == 0 and out8.max() == 255
        assert out16.min() == 0 and out16.max() == 65535


# ---------------------------------------------------------------------------
# is_integer — all backends
# ---------------------------------------------------------------------------

class TestIsIntegerAllBackends:
    def test_memmap(self, tmp_path):
        mmap_path = tmp_path / "ints.dat"
        mmap = np.memmap(mmap_path, dtype=np.int16, mode="w+", shape=(4,))
        mmap[:] = 1
        assert is_integer(mmap)


# ---------------------------------------------------------------------------
# get_size / random_int
# ---------------------------------------------------------------------------

class TestGetSize:
    def test_units(self):
        arr = np.zeros((10,), dtype=np.uint8)
        assert get_size(arr, unit="B") == arr.nbytes
        assert np.isclose(get_size(arr, unit="KB"), arr.nbytes / 1024)
        assert np.isclose(get_size(arr, unit="MB"), arr.nbytes / (1024 ** 2))


class TestRandomInt:
    def test_seeded(self):
        N = 10
        M = 5
        seed = 123
        rng = np.random.RandomState(seed)
        expected = rng.randint(0, N, M)
        out = random_int(N, M, seed=seed)
        assert np.array_equal(out, expected)

        rng = np.random.RandomState(seed)
        expected_scalar = rng.randint(0, N)
        out_scalar = random_int(N, seed=seed)
        assert out_scalar == expected_scalar


# ---------------------------------------------------------------------------
# ravel_index / unravel_index / border_indices
# ---------------------------------------------------------------------------

class TestRavelUnravelIndex:
    def test_roundtrip(self):
        shape = (3, 4, 5)
        coords = np.array([1, 2, 3])
        idx = ravel_index(coords, shape)
        out = unravel_index(idx, shape)
        assert out == tuple(coords)


class TestBorderIndices:
    def test_basic(self):
        arr = np.zeros((3, 4), dtype=np.int32)
        border = border_indices(arr.shape)
        assert border.size > 0
        flat = arr.ravel()
        flat[border] = 1
        assert flat.sum() == np.unique(border).size


# ---------------------------------------------------------------------------
# split_array / reconstruct_array
# ---------------------------------------------------------------------------

class TestSplitReconstruct:
    def test_round_trip(self):
        array = np.arange(20).reshape(5, 4)
        parts = split_array(array, parts=2)
        rebuilt = reconstruct_array(parts)
        assert np.array_equal(rebuilt, array)

    def test_warns_on_uneven(self, capsys):
        array = np.arange(20).reshape(5, 4)
        _ = split_array(array, parts=(2,), axes=0)
        captured = capsys.readouterr()
        assert "Warning: Axis 0" in captured.out

    def test_axes_length_mismatch(self):
        array = np.arange(24).reshape(6, 4)
        with pytest.raises(ValueError):
            split_array(array, parts=(2, 2), axes=(0,))


# ---------------------------------------------------------------------------
# enumerate_nested
# ---------------------------------------------------------------------------

class TestEnumerateNested:
    def test_pairs(self):
        a = [[1, 2], [3, 4]]
        b = [[10, 20], [30, 40]]
        results = list(enumerate_nested(a, b))
        assert results[0] == ([0, 0], 1, 10)
        assert results[1] == ([0, 1], 2, 20)
        assert results[2] == ([1, 0], 3, 30)
        assert results[3] == ([1, 1], 4, 40)


# ---------------------------------------------------------------------------
# move_axis / move_min_dim
# ---------------------------------------------------------------------------

class TestMoveAxis:
    def test_first_last(self):
        x = np.zeros((2, 3, 4))
        from ocdkit.array import move_axis
        assert move_axis(x, axis=-1, pos="first").shape == (4, 2, 3)
        assert move_axis(x, axis=0, pos="last").shape == (3, 4, 2)

    def test_pos_int_variants(self):
        from ocdkit.array import move_axis
        x = np.zeros((2, 3, 4, 5))
        assert move_axis(x, axis=2, pos=0).shape == (4, 2, 3, 5)
        assert move_axis(x, axis=1, pos=-1).shape == (2, 4, 5, 3)


class TestMoveMinDim:
    def test_channel_last_stays(self):
        from ocdkit.array import move_min_dim
        x = np.zeros((4, 5, 2))
        assert move_min_dim(x).shape == (4, 5, 2)

    def test_force(self):
        from ocdkit.array import move_min_dim
        x = np.zeros((10, 2, 10))
        assert move_min_dim(x, force=True).shape == (10, 10, 2)

    def test_2d_unchanged(self):
        from ocdkit.array import move_min_dim
        x = np.zeros((3, 4))
        assert move_min_dim(x).shape == (3, 4)


# ---------------------------------------------------------------------------
# moving_average / add_poisson_noise / correct_illumination
# ---------------------------------------------------------------------------

class TestMovingAverage:
    def test_simple(self):
        from ocdkit.array import moving_average
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        out = moving_average(x, 3)
        assert out.shape == x.shape
        assert np.isfinite(out).all()


class TestAddPoissonNoise:
    def test_bounds(self):
        from ocdkit.array import add_poisson_noise
        img = np.ones((8, 8), dtype=np.float32) * 0.5
        noisy = add_poisson_noise(img)
        assert noisy.shape == img.shape
        assert noisy.min() >= 0.0
        assert noisy.max() <= 1.0


class TestCorrectIllumination:
    def test_finite(self):
        from ocdkit.array import correct_illumination
        img = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
        out = correct_illumination(img, sigma=1)
        assert out.shape == img.shape
        assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# resize_image
# ---------------------------------------------------------------------------

class TestResizeImage:
    def test_errors_without_size(self):
        from ocdkit.array import resize_image
        with pytest.raises(ValueError):
            resize_image(np.zeros((4, 4), dtype=np.float32))

    def test_no_channels_stack(self):
        from ocdkit.array import resize_image
        img = np.zeros((2, 4, 4), dtype=np.float32)
        out = resize_image(img, rsz=0.5, no_channels=True)
        assert out.shape == (2, 2, 2)

    def test_with_channels(self):
        from ocdkit.array import resize_image
        img = np.zeros((2, 4, 4, 2), dtype=np.float32)
        out = resize_image(img, rsz=[0.5, 0.5], no_channels=False)
        assert out.shape == (2, 2, 2, 2)

    def test_2d(self):
        from ocdkit.array import resize_image
        img = np.zeros((4, 6), dtype=np.float32)
        out = resize_image(img, rsz=0.5, no_channels=True)
        assert out.shape == (2, 3)
