#!/usr/bin/env python3
"""Vendor the bayeswire spec and conformance corpus into this repository.

Bayesite stays zero-dependency and offline-capable: it consumes the wire
format through byte-identical vendored files, never through package
management. Run this against a bayeswire checkout at the pinned ref:

    python3 scripts/vendor_bayeswire.py --bayeswire-path ../bayeswire

It copies the normative spec docs and the golden corpus, records the source
commit in ``BAYESWIRE_TAG``, and writes ``bayeswire-vendor.json`` with a
sha256 per vendored file. The validation ladder verifies the vendored bytes
against that manifest on every run; bumping the pin is a PR whose diff is
the compatibility review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

VENDORED_SPEC = {
    "spec/ir-format-v1.md": "docs/ir-format-v1.md",
    "spec/ir-v1-tags.md": "docs/ir-v1-tags.md",
}
CORPUS_SOURCE = "src/bayeswire/corpus"
CORPUS_DEST = "tests/golden_ir"


def _source_commit(bayeswire_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(bayeswire_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        sys.exit(f"cannot resolve bayeswire commit: {result.stderr.strip()}")
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bayeswire-path", type=Path, required=True)
    args = parser.parse_args()

    bayeswire_root = args.bayeswire_path.resolve()
    corpus_root = bayeswire_root / CORPUS_SOURCE
    if not corpus_root.is_dir():
        sys.exit(f"not a bayeswire checkout (no corpus): {bayeswire_root}")

    commit = _source_commit(bayeswire_root)
    vendored: dict[str, str] = {}

    for source_rel, dest_rel in VENDORED_SPEC.items():
        source = bayeswire_root / source_rel
        dest = REPO_ROOT / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        vendored[dest_rel] = _sha256(dest)
        print(f"vendored {dest_rel}")

    dest_corpus = REPO_ROOT / CORPUS_DEST
    if dest_corpus.exists():
        shutil.rmtree(dest_corpus)
    for source in sorted(corpus_root.rglob("*.json")):
        rel = source.relative_to(corpus_root)
        dest = dest_corpus / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        vendored[f"{CORPUS_DEST}/{rel}"] = _sha256(dest)
        print(f"vendored {CORPUS_DEST}/{rel}")

    (REPO_ROOT / "BAYESWIRE_TAG").write_text(commit + "\n", encoding="utf-8")
    manifest = {"bayeswire_commit": commit, "files": dict(sorted(vendored.items()))}
    (REPO_ROOT / "bayeswire-vendor.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"pinned bayeswire commit {commit}")


if __name__ == "__main__":
    main()
