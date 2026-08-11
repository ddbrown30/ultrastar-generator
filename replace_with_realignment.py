from pathlib import Path
import argparse


def process_folder(root: Path, reverse: bool = False):
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

            # Don't overwrite an existing REALIGNED file
            if realigned_file.exists():
                print(f"SKIPPED (REALIGNED file already exists): {realigned_file}")
                continue

            print(f"Reversing: {txt_file}")
            print(f"  Backup:  {bak_file}")
            print(f"  New:     {realigned_file}")

            # Current .txt -> [REALIGNED].txt
            txt_file.rename(realigned_file)

            # .bak -> .txt
            bak_file.rename(txt_file)

    else:
        # Normal replacement:
        # File.txt -> File.bak
        # File [REALIGNED].txt -> File.txt

        for realigned_file in root.rglob("*.txt"):
            name = realigned_file.name

            # Must end exactly with " [REALIGNED].txt"
            suffix = " [REALIGNED].txt"
            if not name.endswith(suffix):
                continue

            # Remove " [REALIGNED]" to get the original filename
            original_name = name[:-len(suffix)] + ".txt"
            original_file = realigned_file.with_name(original_name)

            # Only proceed if the original file exists
            if not original_file.is_file():
                continue

            backup_file = original_file.with_suffix(".bak")

            # Don't overwrite an existing backup
            if backup_file.exists():
                print(f"SKIPPED (backup already exists): {original_file}")
                continue

            print(f"Replacing: {original_file}")
            print(f"  Backup:  {backup_file}")
            print(f"  New:     {realigned_file}")

            # Original -> .bak
            original_file.rename(backup_file)

            # [REALIGNED] -> original name
            realigned_file.rename(original_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory",
        type=str,
        help="Root directory to process"
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse the replacement: restore .bak and add [REALIGNED] to the current .txt"
    )

    args = parser.parse_args()
    root = Path(args.directory).expanduser().resolve()

    if not root.is_dir():
        print(f"Folder does not exist: {root}")
        raise SystemExit(1)

    process_folder(root, reverse=args.reverse)
    print("Done.")