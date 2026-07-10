#!/usr/bin/env python3
"""Self-tests for release packaging helpers."""

from __future__ import annotations

import importlib.util
import os
import tarfile
import tempfile
import tomllib
import unittest
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_script(name: str):
    path = REPO_ROOT / "scripts" / name
    if not path.exists():
        raise AssertionError(f"missing release helper {path}")
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseToolingTests(unittest.TestCase):
    def test_release_docs_match_crate_version(self) -> None:
        cargo = tomllib.loads((REPO_ROOT / "crates/core/Cargo.toml").read_text(encoding="utf-8"))
        version = cargo["package"]["version"]
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        capabilities = (REPO_ROOT / "docs/capabilities-v0.md").read_text(encoding="utf-8")

        self.assertIn(f"VERSION=v{version}", readme)
        self.assertIn(f"--tag v{version}", readme)
        self.assertIn(f'"version": "{version}"', capabilities)

    def test_package_release_creates_archive_and_checksum(self) -> None:
        package_release = load_script("package_release.py")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            binary = tmp_path / "bayesite"
            binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            binary.chmod(binary.stat().st_mode | 0o755)
            out_dir = tmp_path / "dist"

            archive, checksum = package_release.package_release(
                binary=binary,
                name="bayesite-v0.1.0-test",
                archive_format="tar.gz",
                out_dir=out_dir,
                repo_root=REPO_ROOT,
            )

            self.assertTrue(archive.exists())
            self.assertTrue(checksum.exists())
            self.assertIn(archive.name, checksum.read_text(encoding="utf-8"))
            with tarfile.open(archive, "r:gz") as tar:
                names = set(tar.getnames())
            self.assertIn("bayesite-v0.1.0-test/bayesite", names)
            self.assertIn("bayesite-v0.1.0-test/README.md", names)
            self.assertIn("bayesite-v0.1.0-test/LICENSE", names)
            self.assertIn("bayesite-v0.1.0-test/NOTICE", names)

    def test_package_release_creates_zip_for_windows(self) -> None:
        package_release = load_script("package_release.py")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            binary = tmp_path / "bayesite.exe"
            binary.write_bytes(b"fake exe")
            out_dir = tmp_path / "dist"

            archive, checksum = package_release.package_release(
                binary=binary,
                name="bayesite-v0.1.0-test-windows",
                archive_format="zip",
                out_dir=out_dir,
                repo_root=REPO_ROOT,
            )

            self.assertTrue(archive.exists())
            self.assertTrue(checksum.exists())
            with zipfile.ZipFile(archive) as zip_file:
                names = set(zip_file.namelist())
            self.assertIn("bayesite-v0.1.0-test-windows/bayesite.exe", names)
            self.assertIn("bayesite-v0.1.0-test-windows/README.md", names)

    def test_smoke_release_binary_accepts_cli_json_error_surface(self) -> None:
        smoke = load_script("smoke_release_binary.py")
        with tempfile.TemporaryDirectory() as tmp:
            if os.name == "nt":
                binary = Path(tmp) / "bayesite.cmd"
                binary.write_text(
                    "@echo off\r\n"
                    '>&2 echo {"error_format":"v0-provisional","error":"InvalidSettings","message":"missing command: use bayesite sample"}\r\n'
                    "exit /b 1\r\n",
                    encoding="utf-8",
                )
            else:
                binary = Path(tmp) / "bayesite"
                binary.write_text(
                    "#!/bin/sh\n"
                    "printf '%s\\n' '{\"error_format\":\"v0-provisional\",\"error\":\"InvalidSettings\",\"message\":\"missing command: use bayesite sample\"}' >&2\n"
                    "exit 1\n",
                    encoding="utf-8",
                )
                binary.chmod(binary.stat().st_mode | 0o755)
            smoke.smoke_binary(binary)


if __name__ == "__main__":
    unittest.main()
