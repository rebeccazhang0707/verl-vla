# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Check that the opt-in GR00T package came from verl-vla's pinned commit."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

from verl_vla.models.gr00t_n1d6 import GR00T_N1D6_COMMIT

_EAGLE_ASSETS = (
    "added_tokens.json",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
)


def direct_url() -> dict:
    try:
        dist = distribution("gr00t")
    except PackageNotFoundError as exc:
        raise ModuleNotFoundError(
            "GR00T N1.6 is not installed. Build docker/Dockerfile.gr00t or install "
            f"the pinned source commit {GR00T_N1D6_COMMIT}."
        ) from exc
    direct_url_file = next((file for file in (dist.files or []) if file.name == "direct_url.json"), None)
    if direct_url_file is None:
        raise RuntimeError("The installed gr00t distribution has no direct_url.json; install it from pinned source.")
    with dist.locate_file(direct_url_file).open(encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    metadata = direct_url()
    commit = metadata.get("vcs_info", {}).get("commit_id")
    if commit is None and GR00T_N1D6_COMMIT in metadata.get("url", ""):
        commit = GR00T_N1D6_COMMIT
    if commit != GR00T_N1D6_COMMIT:
        raise RuntimeError(
            f"Unsupported GR00T source commit {commit!r}; expected {GR00T_N1D6_COMMIT}. "
            "Reinstall with the command documented in examples/gr00t_sft/README.md."
        )
    # The upstream VCS wheel does not currently include Eagle's non-Python
    # package data.  Fail during image construction instead of allowing
    # Transformers to silently infer OPTConfig from the /opt/... path.
    import gr00t
    from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
    from transformers import AutoConfig

    if Gr00tN1d6.config_class is not Gr00tN1d6Config:
        raise RuntimeError("The installed GR00T package has an incompatible Gr00tN1d6 config class.")

    eagle_dir = Path(gr00t.__file__).parent / "model" / "modules" / "nvidia" / "Eagle-Block2A-2B-v2"
    missing_assets = [name for name in _EAGLE_ASSETS if not (eagle_dir / name).is_file()]
    if missing_assets:
        raise RuntimeError(
            f"The installed GR00T package is missing Eagle assets: {missing_assets}. "
            "Use docker/Dockerfile.gr00t, which restores them from the pinned source commit."
        )

    eagle_config = AutoConfig.from_pretrained(eagle_dir, trust_remote_code=True)
    if eagle_config.model_type != "eagle_3_vl":
        raise RuntimeError(f"Expected Eagle config model_type 'eagle_3_vl', got {eagle_config.model_type!r}.")

    print(f"GR00T N1.6 source package and Eagle assets verified at {commit}")


if __name__ == "__main__":
    main()
