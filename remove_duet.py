import argparse
from pathlib import Path


OLD = " [DUET]"
NEW = ""


def remove_duet(text):
    return text.replace(OLD, NEW)


def update_txt_file(path, apply):
    """
    Remove [DUET] from references inside a USDX text file.

    Returns True if any changes were made.
    """

    try:
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    except UnicodeDecodeError:
        try:
            lines = path.read_text(encoding="cp1252").splitlines(keepends=True)
        except (UnicodeDecodeError, OSError) as e:
            print(f"ERROR reading {path}: {e}")
            return False
    except OSError as e:
        print(f"ERROR reading {path}: {e}")
        return False

    changed = False
    new_lines = []

    for line_number, line in enumerate(lines, start=1):
        new_line = remove_duet(line)

        if new_line != line:
            changed = True

            print(f"UPDATE CONTENT: {path}")
            print(f"  Line {line_number}:")
            print(f"    - {line.rstrip()}")
            print(f"    + {new_line.rstrip()}")

        new_lines.append(new_line)

    if changed and apply:
        try:
            path.write_text("".join(new_lines), encoding="utf-8", newline="")
        except OSError as e:
            print(f"ERROR writing {path}: {e}")
            return False

    return changed


def update_associated_txt_files(renamed_file, apply):
    """
    If a file containing [DUET] is renamed, update any .txt files in the
    same directory that reference that filename.
    """

    old_name = renamed_file.name
    new_name = old_name.replace(OLD, NEW)

    changed = False

    for txt_path in renamed_file.parent.glob("*.txt"):
        try:
            lines = txt_path.read_text(
                encoding="utf-8"
            ).splitlines(keepends=True)
        except UnicodeDecodeError:
            try:
                lines = txt_path.read_text(
                    encoding="cp1252"
                ).splitlines(keepends=True)
            except (UnicodeDecodeError, OSError):
                continue
        except OSError:
            continue

        file_changed = False
        new_lines = []

        for line_number, line in enumerate(lines, start=1):
            new_line = line.replace(old_name, new_name)

            if new_line != line:
                file_changed = True
                changed = True

                print(f"UPDATE CONTENT: {txt_path}")
                print(f"  Line {line_number}:")
                print(f"    - {line.rstrip()}")
                print(f"    + {new_line.rstrip()}")

            new_lines.append(new_line)

        if file_changed and apply:
            txt_path.write_text(
                "".join(new_lines),
                encoding="utf-8",
                newline=""
            )

    return changed


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Remove [DUET] from filenames, directory names, "
            "and references inside USDX text files."
        )
    )

    parser.add_argument(
        "root",
        type=Path,
        help="Root directory containing the USDX folders",
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually perform the changes",
    )

    args = parser.parse_args()
    root = args.root

    if not root.is_dir():
        parser.error(f"Directory does not exist: {root}")

    if args.apply:
        print(f"APPLYING CHANGES under: {root}")
    else:
        print(f"DRY RUN under: {root}")
        print("Use --apply to actually make the changes.")

    print()

    found = False

    # ------------------------------------------------------------
    # 1. Update all existing .txt files
    # ------------------------------------------------------------

    for path in root.rglob("*.txt"):
        if update_txt_file(path, args.apply):
            found = True

    # ------------------------------------------------------------
    # 2. Rename files and directories
    # ------------------------------------------------------------

    entries = sorted(
        root.rglob("*"),
        key=lambda p: len(p.parts),
        reverse=True,
    )

    for path in entries:
        if OLD not in path.name:
            continue

        found = True

        new_name = path.name.replace(OLD, NEW)
        new_path = path.with_name(new_name)

        if new_path.exists():
            print("WARNING: Target already exists:")
            print(f"  {path}")
            print(f"  -> {new_path}")
            continue

        kind = "DIRECTORY" if path.is_dir() else "FILE"

        print(f"RENAME {kind}:")
        print(f"  {path}")
        print(f"  -> {new_path}")

        # Update references to renamed files inside local .txt files.
        if path.is_file():
            update_associated_txt_files(path, args.apply)

        if args.apply:
            path.rename(new_path)

    if not found:
        print("No [DUET] references or names were found.")

    print()
    print("Done.")


if __name__ == "__main__":
    main()