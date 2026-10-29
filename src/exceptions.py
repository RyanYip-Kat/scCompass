class ScCompassError(Exception):
    """Base exception for scCompass."""


class SolverUnavailableError(ScCompassError):
    """Raised when the configured COMAPSS solver backend is unavailable."""


class CheckpointMismatchError(ScCompassError):
    """Raised when an existing checkpoint belongs to different input or settings."""
