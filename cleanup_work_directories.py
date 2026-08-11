from pathlib import Path
import shutil
import sys

def delete_ultrastar_work(root: Path) -> None:
    for path in root.rglob(".ultrastar_work"):
        if path.is_dir():
            print(f"Deleting: {path}")
            shutil.rmtree(path)

if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    if not root.is_dir():
        print(f"Error: directory does not exist: {root}")
        sys.exit(1)

    delete_ultrastar_work(root)