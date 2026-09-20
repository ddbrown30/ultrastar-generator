"""
backup_song_files.py

Backs up every UltraStar song .txt file under a directory tree into a
single zip archive (paths stored relative to the backup root, so restore
can put each file back exactly where it came from), and restores that
zip back to disk.

Usage:
    python backup_song_files.py backup "Z:\\Songs"
        -> Z:\\Songs\\songs_backup_<timestamp>.zip

    python backup_song_files.py backup "Z:\\Songs" --output D:\\backups\\songs.zip

    python backup_song_files.py restore songs_backup_20260920_120000.zip
        -> preview only (dry run); restores to the original backup root
           by default, recorded inside the archive

    python backup_song_files.py restore songs_backup_20260920_120000.zip --apply
        -> actually restore

    python backup_song_files.py restore songs.zip --dest "Z:\\Songs" --apply
        -> restore to an explicit location instead of the recorded root
"""

import argparse
import json
import time
import zipfile
from pathlib import Path
from typing import List, Optional

DEFAULT_PATTERN = "*.txt"
EXCLUDE_DIR_NAMES = (".ultrastar_work",)


def find_txt_files(root: Path, pattern: str = DEFAULT_PATTERN) -> List[Path]:
    results = []
    for path in root.rglob(pattern):
        if not path.is_file():
            continue
        if any(part in EXCLUDE_DIR_NAMES for part in path.parts):
            continue
        results.append(path)
    return sorted(results)


def backup(root: Path, output: Path, pattern: str, overwrite: bool) -> int:
    root = root.resolve()
    if output.exists() and not overwrite:
        print(f"ERROR: {output} already exists. Pass --overwrite to replace it, or choose a "
              f"different --output path (never overwrites a previous backup by default).")
        return 1

    files = find_txt_files(root, pattern)
    if not files:
        print(f"No files found under {root} (pattern: {pattern}).")
        return 1

    print(f"Found {len(files)} file(s) under {root}.")
    output.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "root": str(root),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "count": len(files),
        "pattern": pattern,
    }

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, path in enumerate(files, 1):
            arcname = path.relative_to(root).as_posix()
            zf.write(path, arcname)
            if i % 50 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] added")
        zf.comment = json.dumps(manifest).encode("utf-8")

    size_mb = output.stat().st_size / (1024 * 1024)
    print()
    print("=" * 70)
    print(f"Backed up {len(files)} file(s) -> {output} ({size_mb:.2f} MB)")
    print(f"Original root recorded in the archive: {root}")
    print("=" * 70)
    return 0


def _load_manifest(zf: zipfile.ZipFile) -> Optional[dict]:
    if not zf.comment:
        return None
    try:
        return json.loads(zf.comment.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None


def _resolve_zip_path(zip_path: Path) -> Optional[Path]:
    """Accepts either a zip file directly, or a folder containing exactly one
    -- auto-detects, same fail-closed-on-ambiguity convention as the rest of
    this project's "give me a folder" tools (e.g. find_existing_txt_in_folder)."""
    if zip_path.is_file():
        return zip_path
    if zip_path.is_dir():
        candidates = sorted(zip_path.glob("*.zip"))
        if len(candidates) == 1:
            print(f"{zip_path} is a folder; using the only .zip file in it: {candidates[0].name}")
            return candidates[0]
        if len(candidates) > 1:
            names = ", ".join(p.name for p in candidates)
            print(f"ERROR: {zip_path} contains multiple .zip files ({names}) -- "
                  f"pass the specific one you want to restore.")
            return None
        print(f"ERROR: {zip_path} is a folder with no .zip file in it.")
        return None
    print(f"ERROR: path not found: {zip_path}")
    return None


def restore(zip_path: Path, dest: Optional[Path], apply_changes: bool) -> int:
    resolved = _resolve_zip_path(zip_path)
    if resolved is None:
        return 1
    zip_path = resolved

    with zipfile.ZipFile(zip_path, "r") as zf:
        manifest = _load_manifest(zf)
        members = [n for n in zf.namelist() if not n.endswith("/")]
        if not members:
            print("Archive has no files to restore.")
            return 1

        if dest is None:
            if manifest and manifest.get("root"):
                dest = Path(manifest["root"])
                print(f"No --dest given; using the original backup root recorded in the archive: {dest}")
            else:
                print("ERROR: no --dest given and the archive has no recorded original root "
                      "(pass --dest explicitly).")
                return 1
        dest = dest.resolve() if dest.is_absolute() else Path.cwd() / dest

        if manifest:
            print(f"Backup info: {manifest.get('count', '?')} file(s), "
                  f"created {manifest.get('created', 'unknown time')}, "
                  f"original root: {manifest.get('root', 'unknown')}")
        print(f"Restore destination: {dest}")
        print()

        plan = []
        new_count = overwrite_count = identical_count = 0
        for name in members:
            target = dest / name
            if target.exists():
                try:
                    incoming = zf.read(name)
                    identical = target.read_bytes() == incoming
                except OSError:
                    identical = False
                if identical:
                    identical_count += 1
                    status = "unchanged"
                else:
                    overwrite_count += 1
                    status = "OVERWRITE"
            else:
                new_count += 1
                status = "new"
            plan.append((name, target, status))

        for name, target, status in plan:
            if status != "unchanged":
                print(f"  [{status}] {target}")

        print()
        print(f"New files:          {new_count}")
        print(f"Files to overwrite: {overwrite_count}")
        print(f"Already identical:  {identical_count}")

        if not apply_changes:
            print()
            print("Mode: DRY RUN -- no file was written. Pass --apply to actually restore.")
            return 0

        restored = failed = 0
        for name, target, status in plan:
            if status == "unchanged":
                continue
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(name))
                restored += 1
            except OSError as e:
                print(f"  ERROR restoring {target}: {e}")
                failed += 1

        print()
        print("=" * 70)
        print(f"Restored: {restored}")
        print(f"Failed:   {failed}")
        print("=" * 70)
        return 0 if failed == 0 else 1


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Back up UltraStar song .txt files into a zip archive, and restore them back."
    )
    sub = p.add_subparsers(dest="action", required=True)

    b = sub.add_parser("backup", help="Back up all matching files under a directory into a zip archive.")
    b.add_argument("root", help="Root directory to scan recursively.")
    b.add_argument("--output", default=None,
                   help="Output zip path (default: <root>/songs_backup_<timestamp>.zip).")
    b.add_argument("--pattern", default=DEFAULT_PATTERN,
                   help=f"Glob pattern for files to back up (default: {DEFAULT_PATTERN}).")
    b.add_argument("--overwrite", action="store_true",
                   help="Allow replacing an existing file at --output (never overwrites by default).")

    r = sub.add_parser("restore", help="Restore files from a zip archive made by 'backup'.")
    r.add_argument("zip_file", help="Path to the backup zip file.")
    r.add_argument("--dest", default=None,
                   help="Directory to restore into (default: the original backup root, recorded "
                        "inside the archive at backup time).")
    r.add_argument("--apply", action="store_true",
                   help="Actually write the restored files. Without this, only a preview of what "
                        "would change is printed -- no file is touched.")

    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.action == "backup":
        root = Path(args.root).expanduser()
        if not root.is_dir():
            print(f"ERROR: not a directory: {root}")
            return 1
        if args.output:
            output = Path(args.output).expanduser()
        else:
            output = root / f"songs_backup_{time.strftime('%Y%m%d_%H%M%S')}.zip"
        return backup(root, output, args.pattern, args.overwrite)

    if args.action == "restore":
        zip_path = Path(args.zip_file).expanduser()
        dest = Path(args.dest).expanduser() if args.dest else None
        return restore(zip_path, dest, apply_changes=args.apply)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
