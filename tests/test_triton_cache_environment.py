import torch

from ninetoothed.compiler import cache


def test_native_launcher_cache_changes_with_runtime_versions(monkeypatch, tmp_path):
    versions = {"torch": "2.9", "torch-npu": "2.9", "triton": "3.2"}
    monkeypatch.setattr(cache, "_compiler_versions", lambda: dict(versions))
    original = cache.triton_cache_directory(tmp_path)
    assert original.parent == tmp_path
    assert cache.triton_cache_directory(tmp_path) == original

    for package in versions:
        previous = versions[package]
        versions[package] = "different-build"
        assert cache.triton_cache_directory(tmp_path) != original
        versions[package] = previous


def test_native_launcher_cache_changes_with_torch_abi(monkeypatch, tmp_path):
    original = cache.triton_cache_directory(tmp_path)
    monkeypatch.setattr(
        torch._C, "_GLIBCXX_USE_CXX11_ABI", not torch._C._GLIBCXX_USE_CXX11_ABI
    )
    assert cache.triton_cache_directory(tmp_path) != original


def test_native_launcher_cache_changes_with_python_environment(monkeypatch, tmp_path):
    original = cache.triton_cache_directory(tmp_path)
    monkeypatch.setattr(cache.sys, "prefix", "/another/environment")
    assert cache.triton_cache_directory(tmp_path) != original
