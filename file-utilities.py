from pathlib import Path
import argparse
import shutil


def delete_ultrastar_work(root: Path) -> None:
    for path in root.rglob(".ultrastar_work"):
        if path.is_dir():
            print(f"Deleting: {path}")
            shutil.rmtree(path)


def delete_bak_files(root: Path) -> None:
    for path in root.rglob("*.bak"):
        if path.is_file():
            print(f"Deleting: {path}")
            path.unlink()


def delete_usdb_files(root: Path) -> None:
    for path in root.rglob("*.usdb"):
        if path.is_file():
            print(f"Deleting: {path}")
            path.unlink()


def process_folder(root: Path, reverse: bool = False) -> None:
    if reverse:
        # Reverse the replacement:
        # File.txt -> File [REALIGNED].txt
        # File.bak -> File.txt

        for txt_file in root.rglob("*.txt"):
            bak_file = txt_file.with_suffix(".bak")

            if not bak_file.is_file():
                continue

            realigned_file = txt_file.with_name(
                txt_file.stem + " [REALIGNED].txt"
            )

            if realigned_file.exists():
                print(f"SKIPPED (REALIGNED file already exists): {realigned_file}")
                continue

            print(f"Reversing: {txt_file}")
            print(f"  Backup:  {bak_file}")
            print(f"  New:     {realigned_file}")

            txt_file.rename(realigned_file)
            bak_file.rename(txt_file)

    else:
        # Normal replacement:
        # File.txt -> File.bak
        # File [REALIGNED].txt -> File.txt

        suffix = " [REALIGNED].txt"

        for realigned_file in root.rglob("*.txt"):
            name = realigned_file.name

            if not name.endswith(suffix):
                continue

            original_name = name[:-len(suffix)] + ".txt"
            original_file = realigned_file.with_name(original_name)

            if not original_file.is_file():
                continue

            backup_file = original_file.with_suffix(".bak")

            if backup_file.exists():
                print(f"SKIPPED (backup already exists): {original_file}")
                continue

            print(f"Replacing: {original_file}")
            print(f"  Backup:  {backup_file}")
            print(f"  New:     {realigned_file}")

            original_file.rename(backup_file)
            realigned_file.rename(original_file)


def main() -> None:
    parser = argparse.ArgumentParser()

    mode_group = parser.add_mutually_exclusive_group(required=True)

    mode_group.add_argument(
        "--realign-replace",
        action="store_true",
        help="Replace original files with [REALIGNED] files",
    )

    mode_group.add_argument(
        "--realign-reverse",
        action="store_true",
        help="Reverse the realignment replacement",
    )

    mode_group.add_argument(
        "--clean-work",
        action="store_true",
        help="Delete all .ultrastar_work directories",
    )

    mode_group.add_argument(
        "--clean-bak",
        action="store_true",
        help="Delete all .bak files",
    )

    mode_group.add_argument(
        "--clean-usdb",
        action="store_true",
        help="Delete all .usdb files",
    )

    mode_group.add_argument(
        "--clean-all",
        action="store_true",
        help="Delete all .bak, .usdb, and .ultrastar_work files/directories",
    )

    parser.add_argument(
        "directory",
        type=Path,
        help="Root directory to process",
    )

    args = parser.parse_args()

    root = args.directory.expanduser().resolve()

    if not root.is_dir():
        print(f"Error: directory does not exist: {root}")
        raise SystemExit(1)

    if args.realign_replace:
        process_folder(root, reverse=False)

    elif args.realign_reverse:
        process_folder(root, reverse=True)

    elif args.clean_work:
        delete_ultrastar_work(root)

    elif args.clean_bak:
        delete_bak_files(root)

    elif args.clean_usdb:
        delete_usdb_files(root)

    elif args.clean_all:
        delete_bak_files(root)
        delete_usdb_files(root)
        delete_ultrastar_work(root)

    print("Done.")


if __name__ == "__main__":
    main()