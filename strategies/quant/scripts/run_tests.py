import subprocess
import sys
from pathlib import Path

"""
- This script is a command line wrapper to run tests
- The script:
1) locates strategies/quant/tests/test.py,
2) executes it using the current Python interpreter,
3) lets the existing test runner handle everything else.

- How to run using command line:
python strategies/quant/scripts/run_tests.py
"""


def main():
    test_script = (
        Path(__file__).resolve().parent.parent
        / "tests"
        / "test.py"
    )

    subprocess.run(
        [sys.executable, str(test_script)],
        check=True,
    )


if __name__ == "__main__":
    main()
