#!/usr/bin/env python3
"""Package a built Bayesite CLI binary for GitHub Releases."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parent.parent
ArchiveFormat = Literal["tar.gz", "zip"]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_release_payload(binary: Path, stage: Path, repo_root: Path) -> None:
    stage.mkdir(parents=True, exist_ok=False)
    shutil.copy2(binary, stage / binary.name)
    for name in ["README.md", "LICENSE", "NOTICE"]:
        shutil.copy2(repo_root / name, stage / name)


def _write_tar_gz(stage: Path, archive: Path, name: str) -> None:
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(stage, arcname=name)


def _write_zip(stage: Path, archive: Path, name: str) -> None:
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for path in sorted(stage.rglob("*")):
            zip_file.write(path, Path(name) / path.relative_to(stage))


def package_release(
    *,
    binary: Path,
    name: str,
    archive_format: ArchiveFormat,
    out_dir: Path,
    repo_root: Path = REPO_ROOT,
) -> tuple[Path, Path]:
    """Create a release archive and adjacent .sha256 checksum file."""
    binary = binary.resolve()
    repo_root = repo_root.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not binary.is_file():
        raise FileNotFoundError(f"release binary not found: {binary}")
    if archive_format not in ("tar.gz", "zip"):
        raise ValueError("archive_format must be tar.gz or zip")
    stage_parent = out_dir / ".staging"
    if stage_parent.exists():
        shutil.rmtree(stage_parent)
    stage = stage_parent / name
    archive = out_dir / f"{name}.{archive_format}"
    checksum = Path(f"{archive}.sha256")
    try:
        _copy_release_payload(binary, stage, repo_root)
        if archive_format == "tar.gz":
            _write_tar_gz(stage, archive, name)
        else:
            _write_zip(stage, archive, name)
        digest = _sha256_file(archive)
        checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
        return archive, checksum
    finally:
        if stage_parent.exists():
            shutil.rmtree(stage_parent)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--format", choices=["tar.gz", "zip"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    archive, checksum = package_release(
        binary=args.binary,
        name=args.name,
        archive_format=args.format,
        out_dir=args.out,
    )
    print(archive)
    print(checksum)


if __name__ == "__main__":
    main()
