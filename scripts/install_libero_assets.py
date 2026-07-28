#!/usr/bin/env python3

# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Install the verified LIBERO simulator assets omitted from its PyPI package."""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import tempfile
import urllib.request
from importlib.metadata import distribution
from pathlib import Path

LIBERO_ASSETS_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
LIBERO_ASSETS_SHA256 = "05ffcf8349b2e7ef31b038451253d76ca757debbf88c3a0c1de569ca38a80b14"
LIBERO_ASSETS_URL = f"https://codeload.github.com/Lifelong-Robot-Learning/LIBERO/tar.gz/{LIBERO_ASSETS_COMMIT}"
ARCHIVE_ASSETS_ROOT = f"LIBERO-{LIBERO_ASSETS_COMMIT}/libero/libero/assets"
EXPECTED_SCENE = Path("scenes/libero_tabletop_base_style.xml")


def main() -> None:
    libero_package_dir = Path(distribution("libero").locate_file("libero/libero")).resolve()
    assets_dir = libero_package_dir / "assets"

    with tempfile.TemporaryDirectory(prefix="libero-assets-", dir=libero_package_dir.parent) as temp_dir:
        temp_root = Path(temp_dir)
        archive_path = temp_root / "libero-source.tar.gz"
        extracted_assets_dir = temp_root / "assets"

        print(f"Downloading LIBERO assets from {LIBERO_ASSETS_URL}")
        urllib.request.urlretrieve(LIBERO_ASSETS_URL, archive_path)

        archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        if archive_sha256 != LIBERO_ASSETS_SHA256:
            raise RuntimeError(
                f"LIBERO assets checksum mismatch: expected {LIBERO_ASSETS_SHA256}, got {archive_sha256}"
            )

        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive.getmembers():
                member_path = Path(member.name)
                try:
                    relative_path = member_path.relative_to(ARCHIVE_ASSETS_ROOT)
                except ValueError:
                    continue

                if relative_path == Path("."):
                    continue

                destination = extracted_assets_dir / relative_path
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise RuntimeError(f"Unsupported entry in LIBERO assets archive: {member.name}")

                destination.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Unable to read LIBERO asset from archive: {member.name}")
                with source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)

        if not (extracted_assets_dir / EXPECTED_SCENE).is_file():
            raise RuntimeError(f"LIBERO assets archive is missing {EXPECTED_SCENE}")

        backup_dir = temp_root / "previous-assets"
        if assets_dir.exists():
            assets_dir.rename(backup_dir)
        try:
            extracted_assets_dir.rename(assets_dir)
        except BaseException:
            if backup_dir.exists():
                backup_dir.rename(assets_dir)
            raise

    print(f"Installed LIBERO assets in {assets_dir}")


if __name__ == "__main__":
    main()
