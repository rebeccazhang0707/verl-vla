# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import re
from pathlib import Path

CONFIG_ROOT = Path("src/verl_vla/workflows/config")


def validate_yaml_format(yaml_lines: list[str]) -> list[str]:
    """Validate verl's documentation and spacing conventions for YAML fields."""
    errors = []

    for index, line in enumerate(yaml_lines):
        stripped = line.strip()
        if not stripped:
            continue

        key_match = re.match(r"^(\s*)([a-zA-Z0-9_]+):", line)
        if not key_match:
            continue

        if index == 0 or not yaml_lines[index - 1].strip().startswith("#"):
            errors.append(f"Missing comment above line {index + 1}: {stripped}")

        if "#" in line and not stripped.startswith("#"):
            comment_index = line.index("#")
            colon_index = line.index(":")
            if comment_index > colon_index:
                errors.append(f"Inline comment found on line {index + 1}: {stripped}")

        if index + 1 < len(yaml_lines) and yaml_lines[index + 1].strip():
            errors.append(f"Missing blank line after line {index + 1}: {stripped}")

    return errors


def check_config_docs() -> list[str]:
    errors = []
    for config_path in sorted(CONFIG_ROOT.rglob("*.yaml")):
        config_errors = validate_yaml_format(config_path.read_text().splitlines(keepends=True))
        errors.extend(f"{config_path}: {error}" for error in config_errors)
    return errors


def test_config_docs() -> None:
    errors = check_config_docs()
    assert not errors, "YAML documentation format check failed:\n" + "\n".join(errors)


if __name__ == "__main__":
    validation_errors = check_config_docs()
    if validation_errors:
        raise SystemExit("YAML documentation format check failed:\n" + "\n".join(validation_errors))
    print("YAML documentation format check passed")
