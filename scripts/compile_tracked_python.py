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

import py_compile
import subprocess
from pathlib import Path


def main() -> None:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.py"],
        check=True,
        capture_output=True,
    )
    for filename in result.stdout.decode().split("\0"):
        if filename and Path(filename).is_file():
            py_compile.compile(filename, doraise=True)


if __name__ == "__main__":
    main()
