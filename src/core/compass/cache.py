import os
import json
import tempfile
from pathlib import Path

from ..globals import RESOURCE_DIR
from ..models import init_model


BUNDLED_CACHE_DIR = os.path.join(RESOURCE_DIR, 'COMPASS')


def resolve_cache_dir(environ=None):
    """Choose a writable cache location, honoring SCCOMPASS_CACHE_DIR."""

    environ = os.environ if environ is None else environ
    configured = environ.get("SCCOMPASS_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    xdg_cache = environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache).expanduser() / "scCompass"
    try:
        return Path.home() / ".cache" / "scCompass"
    except RuntimeError:
        return Path(tempfile.gettempdir()) / "scCompass-cache"


def _ensure_cache_dir(path):
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "scCompass-cache"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    return path


PREPROCESS_CACHE_DIR = str(_ensure_cache_dir(resolve_cache_dir()))

# Keys are tuple of (model, media)
_cache = {}
_new_cache = {}

def load(model, media=None):
    global _cache

    if media is None:
        media = model.media

    if not isinstance(model, str):
        model = model.name

    if (model, media) not in _cache:

        relative = os.path.join(model, media, "preprocess.json")
        writable_file = os.path.join(PREPROCESS_CACHE_DIR, relative)
        bundled_file = os.path.join(BUNDLED_CACHE_DIR, relative)
        cache_file = writable_file if os.path.exists(writable_file) else bundled_file

        if os.path.exists(cache_file):
            with open(cache_file) as fin:
                out = json.load(fin)

            _cache[(model, media)] = out

        else:
            _cache[(model, media)] = {}
            _new_cache[(model, media)] = True

    return _cache[(model, media)]


def save(model, media=None):
    global _cache

    if media is None:
        media = model.media

    if not isinstance(model, str):
        model = model.name

    cache_data = _cache[(model, media)]

    cache_dir = Path(PREPROCESS_CACHE_DIR) / model / media
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / 'preprocess.json'
    temporary = cache_file.with_suffix(".json.tmp")

    with temporary.open('w') as fout:
        json.dump(cache_data, fout, indent=1)
        fout.flush()
        os.fsync(fout.fileno())
    os.replace(temporary, cache_file)

def is_new_cache(model, media=None):
    if media is None:
        media = model.media

    if not isinstance(model, str):
        model = model.name

    if (model, media) in _new_cache:
        return _new_cache[(model, media)]
    else:
        return False

def clear(model, media=None):
    global _cache

    if media is None:
        media = model.media

    if not isinstance(model, str):
        model = model.name

    _cache[(model, media)] = {}

    save(model, media)
