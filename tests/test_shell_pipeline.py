import subprocess


def test_run_pbmc5k_shell_script_has_valid_syntax():
    completed = subprocess.run(
        ["bash", "-n", "scripts/run_pbmc5k.sh"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
