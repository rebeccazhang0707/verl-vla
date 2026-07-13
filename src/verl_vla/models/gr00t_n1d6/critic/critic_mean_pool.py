# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Mean-pool critic backend (re-exported from the shared GR00T critic builders)."""

from .critic_cross_attn import MeanPoolCriticBackend

__all__ = ["MeanPoolCriticBackend"]
