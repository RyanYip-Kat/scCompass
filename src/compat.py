"""Compatibility helpers for the vendored Compass implementation."""


def iteritems(data):
    """Return key/value iteration across supported pandas versions."""

    method = getattr(data, "items", None)
    if method is None:
        method = data.iteritems
    return method()
