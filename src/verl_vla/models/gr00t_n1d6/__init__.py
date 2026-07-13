# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Thin verl adapters for the optional external GR00T N1.6 package.

SFT uses ``trainable_model.Gr00tN1d6TrainableModel`` via ``build_vla_model``.
Arena SAC uses ``modeling_gr00t_sac.Gr00tN1d6ForSAC`` (also via the builder when
``override_config.sac_enable`` / ``policy_type=arena``).
"""

GR00T_N1D6_COMMIT = "e29d8fc50b0e4745120ae3fb72447986fe638aa6"

__all__ = ["GR00T_N1D6_COMMIT"]
