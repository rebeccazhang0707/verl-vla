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

"""Verify a LIBERO installation with an OSMesa-rendered CPU rollout."""

from __future__ import annotations

import os
from importlib.metadata import distribution, version
from pathlib import Path

# These variables must be set before importing MuJoCo, PyOpenGL, robosuite, or
# LIBERO. Override any inherited GPU renderer so this check always exercises
# the CPU-only OSMesa path.
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"


def initialize_libero_config() -> None:
    """Create LIBERO's default path config without its interactive first-run prompt."""
    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", "~/.libero")).expanduser()
    config_file = config_dir / "config.yaml"
    if config_file.exists():
        return

    benchmark_root = Path(distribution("libero").locate_file("libero/libero")).resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        "\n".join(
            (
                f"benchmark_root: {benchmark_root}",
                f"bddl_files: {benchmark_root / 'bddl_files'}",
                f"init_states: {benchmark_root / 'init_files'}",
                f"datasets: {benchmark_root.parent / 'datasets'}",
                f"assets: {benchmark_root / 'assets'}",
                "",
            )
        ),
        encoding="utf-8",
    )


initialize_libero_config()

import numpy as np  # noqa: E402
from libero.libero import get_libero_path  # noqa: E402
from libero.libero.benchmark import get_benchmark  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

EXPECTED_VERSIONS = {
    "libero": "0.1.1",
    "robosuite": "1.4.0",
}
CAMERA_NAMES = ("agentview", "robot0_eye_in_hand")
IMAGE_KEYS = ("agentview_image", "robot0_eye_in_hand_image")


def check_versions() -> None:
    for package, expected in EXPECTED_VERSIONS.items():
        installed = version(package)
        if installed != expected:
            raise RuntimeError(f"Unsupported {package} version {installed!r}; expected {expected!r}.")


def check_assets() -> None:
    for asset_name in ("bddl_files", "init_states"):
        asset_path = Path(get_libero_path(asset_name))
        if not asset_path.is_dir():
            raise RuntimeError(f"LIBERO {asset_name} directory does not exist: {asset_path}")

    scene_path = Path(get_libero_path("assets")) / "scenes" / "libero_tabletop_base_style.xml"
    if not scene_path.is_file():
        raise RuntimeError(f"LIBERO scene asset does not exist: {scene_path}")


def check_cpu_rendering() -> None:
    benchmark = get_benchmark("libero_spatial")()
    task_id = 0
    task = benchmark.get_task(task_id)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    if not bddl_file.is_file():
        raise RuntimeError(f"LIBERO task definition does not exist: {bddl_file}")

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_names=list(CAMERA_NAMES),
        camera_heights=64,
        camera_widths=64,
        camera_depths=False,
    )
    try:
        env.seed(0)
        env.reset()
        init_states = benchmark.get_task_init_states(task_id)
        env.set_init_state(init_states[0])
        observation, _reward, _done, _info = env.step(np.zeros(7, dtype=np.float32))

        for key in IMAGE_KEYS:
            image = observation.get(key)
            if image is None:
                raise RuntimeError(f"LIBERO observation is missing rendered image {key!r}.")
            if image.shape != (64, 64, 3):
                raise RuntimeError(f"Unexpected {key} shape {image.shape}; expected (64, 64, 3).")
            if not np.isfinite(image).all():
                raise RuntimeError(f"Rendered image {key!r} contains non-finite values.")
    finally:
        env.close()


def main() -> None:
    check_versions()
    check_assets()
    check_cpu_rendering()
    print(
        "LIBERO CPU rendering verified with "
        f"libero {version('libero')}, robosuite {version('robosuite')}, "
        f"and mujoco {version('mujoco')} (MUJOCO_GL=osmesa)."
    )


if __name__ == "__main__":
    main()
