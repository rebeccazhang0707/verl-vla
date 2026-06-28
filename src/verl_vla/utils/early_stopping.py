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

from collections import deque
from dataclasses import dataclass
from typing import Optional

from verl.base_config import BaseConfig


@dataclass
class TrendEarlyStoppingConfig(BaseConfig):
    """Configuration for sliding-window trend early stopping."""

    _target_: str = "verl_vla.utils.early_stopping.TrendEarlyStoppingConfig"

    enable: bool = False
    metric: str = "sft/loss"
    window_size: int = 100
    min_improvement_ratio: float = 0.001
    patience_windows: int = 3
    warmup_steps: int = 200

    def __post_init__(self):
        if self.window_size < 2:
            raise ValueError(f"window_size must be at least 2, got {self.window_size}")
        if self.min_improvement_ratio < 0:
            raise ValueError(f"min_improvement_ratio must be non-negative, got {self.min_improvement_ratio}")
        if self.patience_windows <= 0:
            raise ValueError(f"patience_windows must be positive, got {self.patience_windows}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")


class TrendEarlyStopper:
    """Early stopper based on the fitted trend inside one sliding metric window."""

    def __init__(self, config: TrendEarlyStoppingConfig) -> None:
        self.config = config

        self._values: deque[float] = deque(maxlen=config.window_size)
        self._steps = 0
        self._bad_windows = 0
        self._max_improvement_ratio: Optional[float] = None
        self._peak_improvement_step: Optional[int] = None
        self._ready_steps = 0
        self._improvement_ratios_since_peak: list[float] = []
        self._threshold_progress: Optional[float] = None
        self._ready = False
        self._should_stop = False

    @property
    def should_stop(self) -> bool:
        return self._should_stop

    @property
    def threshold_progress(self) -> Optional[float]:
        return self._threshold_progress

    def update(self, value: float) -> dict[str, float]:
        self._steps += 1
        self._values.append(float(value))

        if self._steps <= self.config.warmup_steps or len(self._values) < self.config.window_size:
            self._ready = False
            self._should_stop = False
            return {
                "ready": 0.0,
                "window_size": float(self.config.window_size),
                "num_points": float(len(self._values)),
                "bad_windows": float(self._bad_windows),
            }

        values = list(self._values)
        slope, intercept = self._fit_line(values)
        fitted_start = intercept
        fitted_end = intercept + slope * (len(values) - 1)
        window_mean = sum(values) / len(values)
        improvement_ratio = (fitted_start - fitted_end) / max(abs(fitted_start), 1e-12)
        self._ready_steps += 1
        if self._max_improvement_ratio is None or improvement_ratio > self._max_improvement_ratio:
            self._max_improvement_ratio = improvement_ratio
            self._peak_improvement_step = self._ready_steps
            self._improvement_ratios_since_peak = [improvement_ratio]
        else:
            self._improvement_ratios_since_peak.append(improvement_ratio)
        average_acceleration = self._compute_average_acceleration()
        threshold_progress, projected_steps_to_threshold = self._compute_threshold_progress(
            improvement_ratio,
            average_acceleration,
        )
        self._threshold_progress = threshold_progress

        if improvement_ratio < self.config.min_improvement_ratio:
            self._bad_windows += 1
        else:
            self._bad_windows = 0

        self._ready = True
        self._should_stop = self._bad_windows >= self.config.patience_windows

        metrics = {
            "ready": 1.0,
            "window_size": float(self.config.window_size),
            "num_points": float(len(values)),
            "bad_windows": float(self._bad_windows),
            "window_mean": window_mean,
            "fitted_start": fitted_start,
            "fitted_end": fitted_end,
            "improvement_ratio": improvement_ratio,
            "threshold_progress": threshold_progress,
        }
        if average_acceleration is not None:
            metrics["average_acceleration"] = average_acceleration
        if projected_steps_to_threshold is not None:
            metrics["projected_steps_to_threshold"] = projected_steps_to_threshold
        return metrics

    def _compute_average_acceleration(self) -> Optional[float]:
        if len(self._improvement_ratios_since_peak) < 2:
            return None
        ratios = self._improvement_ratios_since_peak
        accelerations = [current - prev for prev, current in zip(ratios, ratios[1:], strict=False)]
        return sum(accelerations) / len(accelerations)

    def _compute_threshold_progress(
        self,
        improvement_ratio: float,
        average_acceleration: Optional[float],
    ) -> tuple[float, Optional[float]]:
        if improvement_ratio <= self.config.min_improvement_ratio:
            return 1.0, 0.0
        if average_acceleration is None or average_acceleration >= 0:
            return 0.0, None
        if self._peak_improvement_step is None:
            return 0.0, None

        elapsed_steps = self._ready_steps - self._peak_improvement_step
        projected_steps_to_threshold = (improvement_ratio - self.config.min_improvement_ratio) / (-average_acceleration)
        denominator = elapsed_steps + projected_steps_to_threshold
        if denominator <= 0:
            return 1.0, projected_steps_to_threshold
        progress = elapsed_steps / denominator
        return min(1.0, max(0.0, progress)), projected_steps_to_threshold

    @staticmethod
    def _fit_line(values: list[float]) -> tuple[float, float]:
        n = len(values)
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n
        denominator = sum((idx - x_mean) ** 2 for idx in range(n))
        if denominator == 0:
            return 0.0, y_mean
        slope = sum((idx - x_mean) * (value - y_mean) for idx, value in enumerate(values)) / denominator
        intercept = y_mean - slope * x_mean
        return slope, intercept
