import textwrap
import subprocess
from typing import Text
from pathlib import Path
from yarl import URL


def latest_request(mocked, request_type, path):
    return mocked.requests.get((request_type, URL(path)))


def json_of_latest_request(r):
    return r[-1].kwargs["json"]


def fingerprint_consistency_test(python_script: Text, tmp_path: Path):
    """Tests whether a provided fingerprinting script returns the same output twice.

    Unfortunately PYTHONHASHSEED is not
    manipulatable in an already started python process:
    https://stackoverflow.com/questions/30585108/disable-hash-randomization-from-within-python-program
    As a result, we need to invoke new processes to make sure fingerprints do not
    use builtin hashing functions which would lead to differing fingerprints
    across runs.
    """
    dedented_script = textwrap.dedent(python_script)
    python_script_path = str(tmp_path / "test_fingerprint.py")
    with open(python_script_path, "w", encoding="utf-8") as script_file:
        script_file.write(dedented_script)

    fp1 = subprocess.getoutput(f"python {python_script_path}")
    fp2 = subprocess.getoutput(f"python {python_script_path}")
    print(fp1)
    print(fp2)
    assert len(fp1) == 32
    assert fp1 == fp2
