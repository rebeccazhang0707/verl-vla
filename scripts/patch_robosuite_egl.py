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

"""Patch robosuite 1.4.0 to map CUDA-visible devices to EGL devices correctly."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

SUPPORTED_ROBOSUITE_VERSION = "1.4.0"
PATCH_MARKER = "# verl-vla: select EGL devices in CUDA-visible order."

EGL_CONTEXT_ORIGINAL = '''from mujoco.egl import egl_ext as EGL
from OpenGL import error


def create_initialized_egl_device_display(device_id=0):
    """Creates an initialized EGL display directly on a device."""
    all_devices = EGL.eglQueryDevicesEXT()
    selected_device = (
        os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if os.environ.get("MUJOCO_EGL_DEVICE_ID", None) is None
        else os.environ.get("MUJOCO_EGL_DEVICE_ID", None)
    )
    if selected_device is None:
        candidates = all_devices
        if device_id == -1:
            device_idx = 0
        else:
            device_idx = device_id
    else:
        if not selected_device.isdigit():
            device_inds = [int(x) for x in selected_device.split(",")]
            if device_id == -1:
                device_idx = device_inds[0]
            else:
                assert device_id in device_inds, "specified device id is not made visible in environment variables."
                device_idx = device_id
        else:
            device_idx = int(selected_device)
        if not 0 <= device_idx < len(all_devices):
            raise RuntimeError(
                f"The MUJOCO_EGL_DEVICE_ID environment variable must be an integer "
                f"between 0 and {len(all_devices)-1} (inclusive), got {device_idx}."
            )
    candidates = all_devices[device_idx : device_idx + 1]
'''

EGL_CONTEXT_PATCHED = (
    """from mujoco.egl import egl_ext as EGL
from OpenGL import error


"""
    + PATCH_MARKER
    + '''
EGL_CUDA_DEVICE_NV = 0x323A
EGLAttrib = ctypes.c_ssize_t
PFNEGLQUERYDEVICEATTRIBEXTPROC = ctypes.CFUNCTYPE(
    EGL.EGLBoolean,
    EGL.EGLDeviceEXT,
    EGL.EGLint,
    ctypes.POINTER(EGLAttrib),
)
_egl_query_device_attrib = PFNEGLQUERYDEVICEATTRIBEXTPROC(EGL.eglGetProcAddress("eglQueryDeviceAttribEXT"))


def _cuda_device_ordinal(device):
    value = EGLAttrib()
    success = _egl_query_device_attrib(device, EGL_CUDA_DEVICE_NV, ctypes.byref(value))
    return value.value if success == EGL.EGL_TRUE else None


def _select_egl_device(all_devices, device_id):
    explicit_egl_device = os.environ.get("MUJOCO_EGL_DEVICE_ID")
    if explicit_egl_device is not None:
        if not explicit_egl_device.isdigit():
            raise RuntimeError(
                "The MUJOCO_EGL_DEVICE_ID environment variable must be a non-negative integer, "
                f"got {explicit_egl_device!r}."
            )
        egl_device_idx = int(explicit_egl_device)
        if not 0 <= egl_device_idx < len(all_devices):
            raise RuntimeError(
                "The MUJOCO_EGL_DEVICE_ID environment variable must be an integer "
                f"between 0 and {len(all_devices) - 1} (inclusive), got {egl_device_idx}."
            )
        return all_devices[egl_device_idx]

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        cuda_device_ordinal = 0 if device_id == -1 else device_id
        candidates = [
            device for device in all_devices if _cuda_device_ordinal(device) == cuda_device_ordinal
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                "Could not map the requested CUDA-visible device to exactly one EGL device: "
                f"CUDA ordinal {cuda_device_ordinal}, matches {len(candidates)}."
            )
        return candidates[0]

    egl_device_idx = 0 if device_id == -1 else device_id
    if not 0 <= egl_device_idx < len(all_devices):
        raise RuntimeError(
            f"EGL device id must be between 0 and {len(all_devices) - 1} (inclusive), got {egl_device_idx}."
        )
    return all_devices[egl_device_idx]


def create_initialized_egl_device_display(device_id=0):
    """Creates an initialized EGL display directly on a device."""
    all_devices = EGL.eglQueryDevicesEXT()
    candidates = [_select_egl_device(all_devices, device_id)]
'''
)

BINDING_ASSERTION_ORIGINAL = """CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if CUDA_VISIBLE_DEVICES != "":
    MUJOCO_EGL_DEVICE_ID = os.environ.get("MUJOCO_EGL_DEVICE_ID", None)
    if MUJOCO_EGL_DEVICE_ID is not None:
        assert MUJOCO_EGL_DEVICE_ID.isdigit() and (
            MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES
        ), "MUJOCO_EGL_DEVICE_ID needs to be set to one of the device id specified in CUDA_VISIBLE_DEVICES"

"""

BINDING_ASSERTION_PATCHED = (
    PATCH_MARKER
    + "\n"
    + "# MUJOCO_EGL_DEVICE_ID is an EGL index, while CUDA_VISIBLE_DEVICES uses CUDA identifiers.\n\n"
)


def _robosuite_package_dir() -> Path:
    try:
        installed_version = version("robosuite")
        package_dir = Path(distribution("robosuite").locate_file("robosuite")).resolve()
    except PackageNotFoundError as error:
        raise RuntimeError("robosuite is not installed in the current Python environment.") from error

    if installed_version != SUPPORTED_ROBOSUITE_VERSION:
        raise RuntimeError(f"This patch supports robosuite {SUPPORTED_ROBOSUITE_VERSION}, found {installed_version}.")
    return package_dir


def _patched_text(path: Path, original: str, patched: str) -> tuple[str, bool]:
    text = path.read_text(encoding="utf-8")
    if patched in text:
        return text, False
    if original not in text:
        raise RuntimeError(
            f"{path} does not match the supported robosuite {SUPPORTED_ROBOSUITE_VERSION} source. "
            "Refusing to modify it."
        )
    return text.replace(original, patched, 1), True


def patch_robosuite(*, check: bool) -> None:
    package_dir = _robosuite_package_dir()
    targets = (
        (
            package_dir / "renderers/context/egl_context.py",
            EGL_CONTEXT_ORIGINAL,
            EGL_CONTEXT_PATCHED,
        ),
        (
            package_dir / "utils/binding_utils.py",
            BINDING_ASSERTION_ORIGINAL,
            BINDING_ASSERTION_PATCHED,
        ),
    )
    updates = [(path, *_patched_text(path, original, patched)) for path, original, patched in targets]
    pending = [path for path, _, changed in updates if changed]

    if check:
        if pending:
            paths = ", ".join(str(path) for path in pending)
            raise RuntimeError(f"robosuite EGL patch is not applied to: {paths}")
        print(f"robosuite {SUPPORTED_ROBOSUITE_VERSION} EGL patch is applied.")
        return

    for path, text, changed in updates:
        if changed:
            path.write_text(text, encoding="utf-8")
            print(f"Patched {path}")

    if not pending:
        print(f"robosuite {SUPPORTED_ROBOSUITE_VERSION} EGL patch is already applied.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that the patch is already applied without modifying robosuite.",
    )
    args = parser.parse_args()
    patch_robosuite(check=args.check)


if __name__ == "__main__":
    main()
