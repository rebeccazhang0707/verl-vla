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

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import wraps

_RATE_WARNING_INTERVAL_S = 5.0
_RATE_WARNING_TOLERANCE = 0.02


@dataclass
class _CallRateState:
    target_hz: float
    last_started_at: float
    window_started_at: float
    interval_count: int = 0
    interval_total_s: float = 0.0
    interval_max_s: float = 0.0
    overrun_count: int = 0


def pace_calls(target_hz_attribute: str):
    """Pace decorated method starts using a target frequency stored on its instance."""

    def decorate(method):
        state_attribute = f"_call_rate_state_{method.__name__}"
        logger = logging.getLogger(method.__module__)

        @wraps(method)
        def wrapped(self, *args, **kwargs):
            target_hz = getattr(self, target_hz_attribute, None)
            if target_hz is None:
                if hasattr(self, state_attribute):
                    delattr(self, state_attribute)
                return method(self, *args, **kwargs)

            target_hz = float(target_hz)
            if target_hz <= 0:
                raise ValueError(f"{target_hz_attribute} must be positive when set, got {target_hz}")

            period_s = 1.0 / target_hz
            arrived_at = time.monotonic()
            state = getattr(self, state_attribute, None)
            if state is None or state.target_hz != target_hz:
                started_at = arrived_at
                setattr(
                    self,
                    state_attribute,
                    _CallRateState(
                        target_hz=target_hz,
                        last_started_at=started_at,
                        window_started_at=started_at,
                    ),
                )
                return method(self, *args, **kwargs)

            wait_s = state.last_started_at + period_s - arrived_at
            if wait_s > 0:
                time.sleep(wait_s)

            started_at = time.monotonic()
            interval_s = started_at - state.last_started_at
            state.last_started_at = started_at
            state.interval_count += 1
            state.interval_total_s += interval_s
            state.interval_max_s = max(state.interval_max_s, interval_s)
            if interval_s > period_s * (1.0 + _RATE_WARNING_TOLERANCE):
                state.overrun_count += 1

            window_s = started_at - state.window_started_at
            if window_s >= _RATE_WARNING_INTERVAL_S:
                actual_hz = state.interval_count / state.interval_total_s
                if actual_hz < target_hz * (1.0 - _RATE_WARNING_TOLERANCE):
                    logger.warning(
                        "%s cannot sustain target rate: target=%.2fHz actual=%.2fHz "
                        "avg_interval=%.2fms max_interval=%.2fms overruns=%d/%d",
                        method.__name__,
                        target_hz,
                        actual_hz,
                        state.interval_total_s / state.interval_count * 1000.0,
                        state.interval_max_s * 1000.0,
                        state.overrun_count,
                        state.interval_count,
                    )
                state.window_started_at = started_at
                state.interval_count = 0
                state.interval_total_s = 0.0
                state.interval_max_s = 0.0
                state.overrun_count = 0

            return method(self, *args, **kwargs)

        return wrapped

    return decorate
