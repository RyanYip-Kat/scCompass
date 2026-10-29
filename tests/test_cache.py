import json
from pathlib import Path

from scCompass.core.compass import cache


def test_cache_reads_bundled_data_but_writes_to_user_directory(tmp_path, monkeypatch):
    bundled = tmp_path / "bundled"
    writable = tmp_path / "writable"
    bundled_file = bundled / "model" / "media" / "preprocess.json"
    bundled_file.parent.mkdir(parents=True)
    bundled_file.write_text(json.dumps({"cached": 1.0}))

    monkeypatch.setattr(cache, "BUNDLED_CACHE_DIR", str(bundled))
    monkeypatch.setattr(cache, "PREPROCESS_CACHE_DIR", str(writable))
    cache._cache.clear()
    cache._new_cache.clear()

    assert cache.load("model", "media") == {"cached": 1.0}
    cache._cache[("model", "media")]["saved"] = 2.0
    cache.save("model", "media")

    written = writable / "model" / "media" / "preprocess.json"
    assert json.loads(written.read_text()) == {"cached": 1.0, "saved": 2.0}
    assert json.loads(bundled_file.read_text()) == {"cached": 1.0}


def test_cache_directory_can_be_configured_by_environment(tmp_path, monkeypatch):
    target = tmp_path / "custom-cache"
    monkeypatch.setenv("SCCOMPASS_CACHE_DIR", str(target))

    assert cache.resolve_cache_dir() == target
