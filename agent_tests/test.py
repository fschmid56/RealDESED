import subprocess
import sys
from pathlib import Path


def main():
    test_dir = Path(__file__).resolve().parent
    test_files = sorted(
        path
        for path in test_dir.glob("test_*.py")
        if path.name != Path(__file__).name
    )

    if not test_files:
        print("No component test files found.")
        return 0

    failures = []
    for test_file in test_files:
        print(f"\n=== Running {test_file.name} ===")
        result = subprocess.run([sys.executable, str(test_file)], cwd=test_dir.parent)
        if result.returncode != 0:
            failures.append((test_file.name, result.returncode))

    if failures:
        print("\nFailed component tests:")
        for name, returncode in failures:
            print(f"- {name}: exit code {returncode}")
        return 1

    print("\nAll component tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
