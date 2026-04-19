"""Tests for ocdkit.gpu — device management, tensor utilities, torch_zoom.

No monkeypatching — tests exercise real hardware paths.
Mac hits MPS, Threadripper hits CUDA, both hit CPU.
"""

import numpy as np
import torch
import pytest

from ocdkit.utils.gpu import (
    to_device, from_device, torch_zoom,
    ensure_torch, torch_and, empty_cache,
    get_device, resolve_device, seed_all,
    torch_CPU, torch_GPU, ARM,
)


# ---------------------------------------------------------------------------
# ARM / resolve_device / get_device
# ---------------------------------------------------------------------------

class TestResolveDevice:
    def test_none_returns_device(self):
        dev = resolve_device(None)
        assert isinstance(dev, torch.device)

    def test_string_passthrough(self):
        assert resolve_device("cpu") == torch.device("cpu")

    def test_device_passthrough(self):
        assert resolve_device(torch.device("cpu")) == torch.device("cpu")

    def test_auto_matches_torch_GPU(self):
        assert resolve_device(None) == torch_GPU


class TestGetDevice:
    def test_returns_tuple(self):
        dev, ok = get_device(0)
        assert isinstance(dev, torch.device)
        assert isinstance(ok, bool)

    def test_none_defaults_to_zero(self):
        dev, ok = get_device(None)
        assert isinstance(dev, torch.device)

    def test_gpu_available_matches_ARM_or_cuda(self):
        dev, ok = get_device(0)
        if ok:
            assert dev.type in ("mps", "cuda")
        else:
            assert dev.type == "cpu"

    def test_invalid_index_falls_back(self):
        dev, ok = get_device(999)
        # Either succeeds (unlikely) or falls back to CPU
        assert isinstance(dev, torch.device)


class TestConstants:
    def test_torch_CPU(self):
        assert torch_CPU == torch.device("cpu")

    def test_torch_GPU_is_device(self):
        assert isinstance(torch_GPU, torch.device)

    def test_ARM_is_bool(self):
        assert isinstance(ARM, bool)


# ---------------------------------------------------------------------------
# empty_cache / seed_all
# ---------------------------------------------------------------------------

class TestEmptyCache:
    def test_callable(self):
        empty_cache()


class TestSeedAll:
    def test_deterministic(self):
        seed_all(42)
        a = torch.randn(5)
        seed_all(42)
        b = torch.randn(5)
        torch.testing.assert_close(a, b)

    def test_different_seeds_differ(self):
        seed_all(1)
        a = torch.randn(100)
        seed_all(2)
        b = torch.randn(100)
        assert not torch.allclose(a, b)


# ---------------------------------------------------------------------------
# to_device / from_device
# ---------------------------------------------------------------------------

class TestToDevice:
    def test_numpy_to_cpu(self):
        arr = np.ones((3, 4), dtype=np.float32)
        t = to_device(arr, torch.device("cpu"))
        assert isinstance(t, torch.Tensor)
        assert t.shape == (3, 4)
        assert t.dtype == torch.float32

    def test_numpy_to_gpu(self):
        arr = np.ones((3, 4), dtype=np.float32)
        t = to_device(arr, torch_GPU)
        assert t.device.type == torch_GPU.type

    def test_tensor_same_device_passthrough(self):
        t = torch.ones(3, 4)
        out = to_device(t, torch.device("cpu"))
        assert out is t

    def test_tensor_to_gpu(self):
        t = torch.ones(3, 4)
        out = to_device(t, torch_GPU)
        assert out.device.type == torch_GPU.type

    def test_tensor_gpu_to_cpu(self):
        t = torch.ones(3, 4, device=torch_GPU)
        out = to_device(t, torch.device("cpu"))
        assert out.device == torch.device("cpu")


class TestFromDevice:
    def test_cpu(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        arr = from_device(t)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_gpu(self):
        t = torch.tensor([1.0, 2.0], device=torch_GPU)
        arr = from_device(t)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, [1.0, 2.0])

    def test_grad_detach(self):
        t = torch.tensor([1.0], requires_grad=True) * 2
        arr = from_device(t)
        assert isinstance(arr, np.ndarray)


# ---------------------------------------------------------------------------
# ensure_torch
# ---------------------------------------------------------------------------

class TestEnsureTorch:
    def test_numpy_adds_batch_dim(self):
        arr = np.ones((3, 4), dtype=np.float32)
        (t,) = ensure_torch(arr, device=torch.device("cpu"))
        assert t.shape == (1, 3, 4)

    def test_tensor_2d_adds_batch(self):
        t = torch.ones(4, 5)
        (out,) = ensure_torch(t)
        assert out.shape == (1, 4, 5)

    def test_tensor_3d_vector_adds_batch(self):
        t = torch.ones(2, 4, 5)
        (out,) = ensure_torch(t)
        assert out.shape == (1, 2, 4, 5)

    def test_already_batched_passthrough(self):
        t = torch.ones(1, 2, 4, 5)
        (out,) = ensure_torch(t)
        assert out.shape == (1, 2, 4, 5)

    def test_non_array_passthrough(self):
        (out,) = ensure_torch("hello")
        assert out == "hello"

    def test_to_gpu(self):
        arr = np.ones((3, 4), dtype=np.float32)
        (t,) = ensure_torch(arr, device=torch_GPU)
        assert t.device.type == torch_GPU.type

    def test_multiple_arrays(self):
        a = np.ones((3, 4), dtype=np.float32)
        b = torch.ones(2, 4, 5)
        ta, tb = ensure_torch(a, b, device=torch.device("cpu"))
        assert ta.shape == (1, 3, 4)
        assert tb.shape == (1, 2, 4, 5)


# ---------------------------------------------------------------------------
# torch_and
# ---------------------------------------------------------------------------

class TestTorchAnd:
    def test_cpu(self):
        a = torch.tensor([True, False, True])
        b = torch.tensor([True, True, False])
        result = torch_and([a, b])
        expected = torch.tensor([True, False, False])
        torch.testing.assert_close(result, expected)

    def test_gpu(self):
        a = torch.tensor([True, False, True], device=torch_GPU)
        b = torch.tensor([True, True, False], device=torch_GPU)
        result = torch_and([a, b])
        assert result.device.type == torch_GPU.type
        expected = torch.tensor([True, False, False], device=torch_GPU)
        torch.testing.assert_close(result, expected)

    def test_broadcast(self):
        a = torch.ones(3, 4, dtype=torch.bool)
        b = torch.tensor([True, False, True, True])
        result = torch_and([a, b])
        assert result.shape == (3, 4)
        assert not result[:, 1].any()


# ---------------------------------------------------------------------------
# torch_zoom
# ---------------------------------------------------------------------------

class TestTorchZoom:
    def test_scale_factor(self):
        x = torch.zeros((1, 1, 4, 6))
        y = torch_zoom(x, scale_factor=0.5, dim=2)
        assert y.shape[-2:] == (2, 3)

    def test_explicit_size(self):
        x = torch.zeros((1, 1, 4, 6))
        z = torch_zoom(x, size=(5, 5))
        assert z.shape[-2:] == (5, 5)

    def test_upscale(self):
        x = torch.zeros((1, 1, 4, 4))
        y = torch_zoom(x, scale_factor=2.0, dim=2)
        assert y.shape[-2:] == (8, 8)

    def test_on_gpu(self):
        x = torch.zeros((1, 1, 4, 6), device=torch_GPU)
        y = torch_zoom(x, scale_factor=0.5, dim=2)
        assert y.device.type == torch_GPU.type
        assert y.shape[-2:] == (2, 3)
