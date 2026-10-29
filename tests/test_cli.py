import subprocess
import sys


def test_cli_help_is_available():
    completed = subprocess.run(
        [sys.executable, "-m", "scCompass.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--input" in completed.stdout
    assert "--output" in completed.stdout
