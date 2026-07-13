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

"""Embodiment adapters for the Isaac Lab Arena env.

The gym wrapper (:class:`~verl_vla.envs.arena.arena_env.IsaacLabArenaEnv`)
is embodiment-agnostic: every robot/control-mode-specific concern (CLI args, env
cfg patching, policy->sim action conversion, proprio-state extraction, camera
extraction, and the stable-hold joint indices) lives in an
:class:`ArenaEmbodiment` adapter, selected by ``arena_state_mode`` and held on the
wrapper as ``self.embodiment``.

Adapters are grouped by their obs/action space. Robots are NOT their own classes: a
robot is described entirely by its simulator yaml (single source of truth). There are
just two classes, one per control space:

* :class:`JointSpaceEmbodiment` — policy speaks robot joint positions/targets. One
  config-driven class backs every joint-space ``arena_state_mode``:
    * ``g1_wbc_joint`` (default): Unitree G1 whole-body-control, 50-DOF joint action
      passed through unchanged (the *identity* case, no ``arena_joint_space_spec``);
      state = ``robot_joint_pos``; camera ``robot_head_cam_rgb`` (see ``arena.yaml``).
    * ``gr1_joint``: Fourier GR1 humanoid (``arena_joint_space_spec: gr1``): 54-DOF sim
      state gathered to 26 GR00T joints; 26 GR00T action scattered to a 36-DOF sim action
      via :class:`ArenaJointMapping`; cameras from ``arena_gr1.yaml``.
    * ``joint_space``: generic alias for a brand-new joint-space robot configured purely
      from yaml.
* :class:`TaskSpaceEmbodiment` — Franka LIBERO Abs-IK (``eef_pose``). Policy uses
  rotvec layout ``pos(3)+rotvec(3)+gripper(1)``; sim uses ``pos(3)+quat_xyzw(4)+gripper(1)``
  (Isaac Lab 3.0 ``xyzw``). The adapter converts at the env boundary.

Import safety: ``isaaclab*`` / ``gr00t`` / ``lightwheel_sdk`` are heavy optional
deps that are usually unavailable outside the Isaac Sim / GR00T docker images.
Everything that touches them is imported lazily inside methods so that
``import verl_vla.envs.arena.embodiment`` (and the pure-python
gather/scatter / tensor transforms) work without any of them installed.

Joint-space index tables are **derived from the Arena embodiment YAMLs**
(:meth:`ArenaJointMapping.from_yaml_dir`) rather than hand-copied, so they stay in sync
with Arena. A mapped joint-space robot supplies its joint-space config entirely from the
simulator yaml (``arena_joint_space_spec``, ``arena_joint_space_dir``, and the three
``arena_joint_space_*_yaml`` file names). ``arena_joint_space_dir`` must be set when a
spec is configured (e.g. via ``${oc.env:...}`` in the simulator yaml).
"""

from __future__ import annotations

import abc
import argparse
import logging
from collections import OrderedDict
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids importing models.gr00t at runtime
    from verl_vla.models.gr00t_n1d6.utils import EmbodimentSpec

logger = logging.getLogger(__name__)

__all__ = [
    "ArenaJointMapping",
    "ArenaEmbodiment",
    "JointSpaceEmbodiment",
    "TaskSpaceEmbodiment",
    "make_arena_embodiment",
    "DEFAULT_ARENA_STATE_MODE",
]

#: Default ``arena_state_mode`` when a config omits it (keeps old configs working).
DEFAULT_ARENA_STATE_MODE = "g1_wbc_joint"


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dataclass config, an OmegaConf node, or a plain dict."""
    if is_dataclass(cfg):
        return getattr(cfg, key, default)
    getter = getattr(cfg, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(cfg, key, default)


# ===========================================================================
# GR1 joint-space gather/scatter (pure python; no isaac / gr00t deps)
# ===========================================================================


@dataclass(frozen=True)
class ArenaJointMapping:
    """GR00T policy-order joints <-> Isaac Lab Arena sim joint-space, bound to a spec.

    One cohesive object per (embodiment, simulator): it pairs the model-side
    embodiment spec (which joint groups / dims the GR00T checkpoint speaks) with
    the simulator-specific gather/scatter index tables, and exposes the
    conversions as methods so callers never touch raw index lists.

    Fields (all derived from the Arena joint-space YAMLs):
        spec:                 model-side embodiment spec (duck-typed; only
                              ``.action_dim`` is read, so unit tests may pass a
                              lightweight stand-in).
        state_full_to_policy: gather indices, full sim state -> policy order
                              (len == ``spec.action_dim``).
        policy_to_action:     scatter indices, policy order -> full sim action
                              (len == ``spec.action_dim``).
        sim_action_dim:       width of the sim action vector.
        state_full_dim:       width of the full sim state vector.
    """

    spec: EmbodimentSpec
    state_full_to_policy: list[int]
    policy_to_action: list[int]
    sim_action_dim: int
    state_full_dim: int

    def __post_init__(self):
        assert len(self.state_full_to_policy) == self.spec.action_dim, (
            f"state map len {len(self.state_full_to_policy)} != spec.action_dim {self.spec.action_dim}"
        )
        assert len(self.policy_to_action) == self.spec.action_dim, (
            f"action map len {len(self.policy_to_action)} != spec.action_dim {self.spec.action_dim}"
        )

    @property
    def policy_dim(self) -> int:
        """Real (unpadded) policy action/state width (== ``spec.action_dim``)."""
        return self.spec.action_dim

    def gather_state(self, full_state):
        """Full sim state -> policy order. ``(B, state_full_dim) -> (B, policy_dim)``.

        Accepts a numpy array or torch tensor (name-based column gather).
        """
        return full_state[:, self.state_full_to_policy]

    def scatter_action(self, policy_action):
        """Policy action -> full sim action. ``(B, policy_dim) -> (B, sim_action_dim)``.

        Joints not controlled by the policy stay at zero; dtype/device preserved.
        Expects a torch tensor (uses ``new_zeros`` + advanced-index assignment).
        """
        sim_action = policy_action.new_zeros(policy_action.shape[0], self.sim_action_dim)
        sim_action[:, self.policy_to_action] = policy_action
        return sim_action

    def gather_action(self, sim_action):
        """Inverse of :meth:`scatter_action`. ``(B, sim_action_dim) -> (B, policy_dim)``."""
        return sim_action[:, self.policy_to_action]

    # --- construction from the Arena joint-space YAMLs (single source of truth) ---

    #: Default joint-space YAML file names (the GR1 embodiment layout). Overridable per
    #: embodiment / via cfg; kept here so the builder has a sensible default.
    DEFAULT_POLICY_YAML = "gr00t_26dof_joint_space.yaml"
    DEFAULT_ACTION_YAML = "36dof_joint_space.yaml"
    DEFAULT_STATE_YAML = "54dof_joint_space.yaml"

    @staticmethod
    def _load_yaml(path: str | Path) -> dict:
        import yaml

        with open(path) as f:
            return yaml.safe_load(f)

    @classmethod
    def build_index_maps_from_yaml(
        cls,
        joint_dir: str | Path,
        *,
        policy_yaml: str = DEFAULT_POLICY_YAML,
        action_yaml: str = DEFAULT_ACTION_YAML,
        state_yaml: str = DEFAULT_STATE_YAML,
        expected_group_dims: "Optional[OrderedDict[str, int]]" = None,
    ) -> tuple[list[int], list[int], int, int]:
        """Derive ``(state_full_to_policy, policy_to_action, sim_action_dim, state_full_dim)``.

        Replicates the name-based lookup Arena uses in ``joints_conversion``: flatten the
        policy groups (in YAML order) into the ordered policy joint names, then look each
        name up in the state/action ``name -> index`` dicts. ``expected_group_dims`` (from
        the model-side spec) is an optional consistency cross-check only.
        """
        joint_dir = Path(joint_dir)
        policy_groups = cls._load_yaml(joint_dir / policy_yaml)["joints"]  # group -> [name]
        action_cfg_yaml = cls._load_yaml(joint_dir / action_yaml)
        state_cfg_yaml = cls._load_yaml(joint_dir / state_yaml)
        action_cfg = action_cfg_yaml["joints"]  # name -> idx
        state_cfg = state_cfg_yaml["joints"]  # name -> idx

        # Flatten policy groups (in YAML order) into the ordered policy joint names.
        flat_names = [name for names in policy_groups.values() for name in names]

        if expected_group_dims is not None:
            yaml_group_dims = OrderedDict((g, len(names)) for g, names in policy_groups.items())
            if list(yaml_group_dims.items()) != list(expected_group_dims.items()):
                logger.warning(
                    "Arena policy YAML groups %s differ from spec group dims %s",
                    dict(yaml_group_dims),
                    dict(expected_group_dims),
                )

        state_indices = [state_cfg[name] for name in flat_names]
        action_map = [action_cfg[name] for name in flat_names]
        sim_action_dim = int(action_cfg_yaml.get("total_joints", len(action_cfg)))
        state_full_dim = int(state_cfg_yaml.get("total_joints", len(state_cfg)))
        return state_indices, action_map, sim_action_dim, state_full_dim

    @classmethod
    def from_yaml_dir(
        cls,
        spec: EmbodimentSpec,
        joint_dir: str | Path,
        *,
        policy_yaml: str = DEFAULT_POLICY_YAML,
        action_yaml: str = DEFAULT_ACTION_YAML,
        state_yaml: str = DEFAULT_STATE_YAML,
    ) -> "ArenaJointMapping":
        """Build a mapping for ``spec`` from the Arena joint-space YAMLs under ``joint_dir``.

        The YAMLs are the single source of truth (no hardcoded index fallback). Callers
        resolve ``joint_dir`` lazily so importing this module never needs the Arena YAMLs
        or the ``gr00t`` package.
        """
        state_idx, action_map, sim_action_dim, state_full_dim = cls.build_index_maps_from_yaml(
            joint_dir,
            policy_yaml=policy_yaml,
            action_yaml=action_yaml,
            state_yaml=state_yaml,
            expected_group_dims=getattr(spec, "state_group_dims", None),
        )
        logger.info("Loaded joint-space maps for %s from %s", getattr(spec, "name", spec), joint_dir)
        return cls(
            spec=spec,
            state_full_to_policy=state_idx,
            policy_to_action=action_map,
            sim_action_dim=sim_action_dim,
            state_full_dim=state_full_dim,
        )


# ===========================================================================
# Camera helpers (pure numpy/torch)
# ===========================================================================


def _to_uint8_rgb(image) -> np.ndarray:
    """Normalise an RGB(A) frame to a contiguous uint8 ``(..., 3)`` array."""
    frame = np.asarray(image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else image)
    if frame.shape[-1] > 3:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        if frame.max(initial=0) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


# ===========================================================================
# Embodiment adapters
# ===========================================================================


class ArenaEmbodiment(abc.ABC):
    """Adapter that owns the embodiment + sim-specific glue for one Arena layout.

    Genuinely robot-specific behaviour is expressed as abstract methods every
    embodiment must implement:
      * ``add_cli_args``        — extra args the example env's ``get_env`` reads.
      * ``policy_to_sim_action``— policy action vector -> sim action tensor.
      * ``extract_state``       — proprio state vector for the policy.

    Everything else is SHARED and lives on this base class; embodiments customise it
    via declarative class attributes rather than overriding methods:
      * ``patch_env_cfg``       — default RL-reward gating.
      * ``extract_images``      — read ``camera_names`` from ``raw_obs["camera_obs"]``,
                                  emit ``{observation.images.<name>: (B,H,W,C) uint8}``.

    Stable-hold class attributes are consumed by the wrapper's stable-action
    adapter (only meaningful for embodiments that hold a pose during teleop, i.e.
    the G1 WBC smoke); they default to ``None`` (feature disabled) so other
    embodiments are unaffected:
      * ``stable_hold_joint_slice`` — number of leading joint columns to hold.
      * ``base_height_index``       — action column carrying the base-height cmd.
      * ``base_height_command``     — held base-height command value.
    """

    #: matches ``arena_state_mode``
    state_mode: str = "base"

    #: stable-hold config (see class docstring); None => stable-hold disabled.
    stable_hold_joint_slice: Optional[int] = None
    base_height_index: Optional[int] = None
    base_height_command: float = 0.0

    #: Whether the wrapper steps the raw policy action (True) or routes through the
    #: stable-hold / teleop adapter (False). Base default is True; G1 WBC identity
    #: joint-space overrides to False in :class:`JointSpaceEmbodiment.__init__`
    #: when no ``arena_joint_space_spec`` is set (stable-hold / teleop). Policy-driven
    #: mapped embodiments (GR1) and task-space (Franka LIBERO) keep True.
    use_policy_action: bool = True

    #: Declarative camera model consumed by the SHARED :meth:`extract_images`.
    #: Differences between embodiments are expressed as data here, not as overridden
    #: methods. There is NO head/wrist distinction: every camera lives in the single
    #: ``camera_names`` list and is treated uniformly. Frames are always read from
    #: ``raw_obs["camera_obs"]`` (observation manager).
    #:   * ``DEFAULT_CAMERA_NAMES`` — camera(s) to use when the cfg omits them.
    DEFAULT_CAMERA_NAMES: tuple[str, ...] = ()

    def __init__(self, cfg: Any, num_envs: int = 1):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.enable_cameras = bool(_cfg_get(cfg, "enable_cameras", True))
        # A single flat camera list for every embodiment (no head/wrist split).
        self.camera_names = self._resolve_camera_config(cfg)
        # Every knob below has a class-attribute default that a yaml can override (cfg
        # value ``None`` => keep the class default). Promoted to instance attributes; the
        # wrapper reads them from here. This is what lets G1 vs GR1 differ by config only.
        self.use_policy_action = bool(self._cfg_or_class(cfg, "use_policy_action"))
        self.stable_hold_joint_slice = self._cfg_or_class(cfg, "stable_hold_joint_slice")
        self.base_height_index = self._cfg_or_class(cfg, "base_height_index")
        self.base_height_command = float(self._cfg_or_class(cfg, "base_height_command"))

    def _resolve_camera_config(self, cfg: Any) -> list[str]:
        """Resolve the single camera-name list from cfg ``camera_names`` + class default.

        ``None`` means "absent" (fall back to :attr:`DEFAULT_CAMERA_NAMES`); an explicit
        empty list is honoured (no cameras).
        """
        names = _cfg_get(cfg, "camera_names", None)
        if names is None:
            names = self.DEFAULT_CAMERA_NAMES
        # Normalise to list[str]; a bare string is wrapped (guards the footgun where it
        # would otherwise be iterated character by character), None -> no cameras.
        if names is None:
            return []
        if isinstance(names, str):
            return [names]
        return [str(name) for name in names]

    def _cfg_or_class(self, cfg: Any, name: str) -> Any:
        """Read ``name`` from cfg; a ``None`` / absent value falls back to the class default."""
        val = _cfg_get(cfg, name, None)
        return getattr(type(self), name) if val is None else val

    @abc.abstractmethod
    def add_cli_args(self, args: argparse.Namespace, cfg: Any) -> None:
        """Add embodiment/task-specific attributes to the ArenaEnvBuilder args (in place)."""

    def patch_env_cfg(self, env_cfg, cfg: Any) -> None:
        """Patch the built (pre-instantiation) env cfg for RL.

        The Arena task defines no reward, so (gated on ``rl_success_reward``, default
        True) turn the composite ``success`` termination into a sparse ``RewTerm`` --
        WITHOUT touching the termination terms, so IsaacLab keeps owning per-step
        episode auto-reset (see ``docs/mdp_auto_reset.md``). The episode horizon is left
        to the Arena task's native ``episode_length_s`` (sim ``time_out``).
        """
        from verl_vla.envs.arena.utils import apply_arena_rl_reward

        if bool(_cfg_get(cfg, "rl_success_reward", True)):
            apply_arena_rl_reward(env_cfg, subtask_reward=bool(_cfg_get(cfg, "subtask_reward", False)))

    @abc.abstractmethod
    def policy_to_sim_action(self, actions, device) -> torch.Tensor:
        """Convert a policy action ``(B, policy_dim)`` to the sim action tensor."""

    @abc.abstractmethod
    def extract_state(self, raw_obs, scene) -> np.ndarray:
        """Return the proprio state ``(B, state_dim)`` float32 for the policy."""

    def extract_images(self, raw_obs) -> dict[str, np.ndarray]:
        """Return ``{observation.images.<name>: (B, H, W, C) uint8}`` for each configured camera."""
        camera_obs = raw_obs.get("camera_obs", {}) if isinstance(raw_obs, dict) else {}
        if self.enable_cameras and (not camera_obs or any(name not in camera_obs for name in self.camera_names)):
            available = list(camera_obs)
            missing = [name for name in self.camera_names if name not in camera_obs]
            if not camera_obs:
                raise KeyError("Camera observations are missing although enable_cameras=True")
            raise KeyError(f"Camera(s) {missing} not found in camera_obs; available={available}")
        return {
            f"observation.images.{name}": _to_uint8_rgb(camera_obs[name]) for name in self.camera_names
        }

    @property
    def policy_action_dim(self) -> Optional[int]:
        """Policy action width to declare to the recorder, or ``None`` for the sim dim.

        The recorder logs the POLICY action that ``env.step`` receives (what the
        model actually emits), not the scattered sim action. Identity embodiments
        (policy width == sim width) return ``None`` so the wrapper falls back to the
        sim ``action_dim`` (unchanged behaviour). Embodiments that remap the policy
        action to a wider sim action (e.g. GR1 26 -> 36) return their real policy
        width so the recorded ``action`` feature dim matches the logged frames.
        """
        return None


class JointSpaceEmbodiment(ArenaEmbodiment):
    """The yaml-driven joint-space obs/action embodiment.

    A joint-space embodiment's policy speaks robot joint positions/targets. This one
    concrete class serves *every* joint-space robot (G1 WBC, GR1, future robots); there
    are no per-robot subclasses -- a robot is entirely described by its simulator yaml.
    Which of the two flavours is active is decided purely by config:

      * *mapped* (e.g. GR1): ``arena_joint_space_spec`` is set, so an
        :class:`ArenaJointMapping` (built lazily from the Arena joint-space YAMLs) gathers
        the policy-order state from the full sim ``robot_joint_pos`` and scatters the
        policy action to the full sim action.
      * *identity* (e.g. G1 WBC): no spec -> :attr:`joint_map` is ``None`` and the same
        methods pass the joint action/state straight through (with a defensive
        empty-policy fallback for the state).

    The class attributes below are only neutral fallbacks; the actual per-robot values
    (cameras, ``object``, ``use_policy_action``, stable-hold, joint-space spec/dir/yamls)
    live *only* in the simulator yaml (``arena.yaml`` for G1, ``arena_gr1.yaml`` for GR1).
    The mapping is built lazily + cached (import-safe: the Arena YAMLs / ``gr00t`` package
    are only touched then, never at construction).
    """

    state_mode = "joint_space"

    # Neutral fallbacks; per-robot values live in the simulator yaml.
    #   cfg keys: object / kitchen_style / arena_joint_space_spec / arena_joint_space_dir /
    #   arena_joint_space_{policy,action,state}_yaml
    # Class default True covers mapped joint-space (GR1). Identity G1 WBC is forced to
    # False below when cfg omits use_policy_action (stable-hold / teleop smoke).
    use_policy_action = True
    default_object: Optional[str] = None

    def __init__(self, cfg: Any, num_envs: int = 1):
        self._joint_map: Optional[ArenaJointMapping] = None
        super().__init__(cfg, num_envs=num_envs)
        # Descriptive label for logging/recorder: mirror ``arena_state_mode`` (e.g.
        # ``g1_wbc_joint`` / ``gr1_joint``) since one class now backs several modes.
        self.state_mode = str(_cfg_get(cfg, "arena_state_mode", DEFAULT_ARENA_STATE_MODE))
        # State width fallback for the (rare) empty-policy-obs identity path; mirrors the
        # wrapper's ``state_dim or action_dim`` default.
        self.state_dim = int(_cfg_get(cfg, "state_dim", None) or _cfg_get(cfg, "action_dim", 50))
        # G1 identity joint-space historically needs stable-hold (use_policy_action=False).
        # Mapped embodiments (arena_joint_space_spec set, e.g. GR1) keep the class/cfg True.
        # Explicit cfg.use_policy_action still wins via _cfg_or_class above; only apply the
        # G1 default when the cfg left the knob unset (None).
        if _cfg_get(cfg, "use_policy_action", None) is None and not _cfg_get(
            cfg, "arena_joint_space_spec", None
        ):
            self.use_policy_action = False

    def add_cli_args(self, args: argparse.Namespace, cfg: Any) -> None:
        # Generic Arena builder knobs (object / kitchen_style / object_set). Arena envs
        # that do not use them ignore them harmlessly.
        args.object = _cfg_get(cfg, "object", None) or self.default_object
        args.kitchen_style = _cfg_get(cfg, "kitchen_style", 2)
        args.object_set = _cfg_get(cfg, "object_set", None)

    def build_joint_map(self) -> Optional[ArenaJointMapping]:
        """Build the policy <-> Arena-sim joint mapping from the joint-space YAMLs.

        Returns ``None`` for an identity joint-space (no ``arena_joint_space_spec``,
        e.g. G1 WBC). Otherwise reads ``arena_joint_space_dir`` and the three YAML file
        names from cfg. Called lazily once (see :attr:`joint_map`).
        """
        spec_name = _cfg_get(self.cfg, "arena_joint_space_spec", None)
        if not spec_name:
            return None
        joint_dir = _cfg_get(self.cfg, "arena_joint_space_dir", None)
        if not joint_dir or not Path(joint_dir).is_dir():
            raise RuntimeError(
                f"arena_joint_space_dir must be a directory containing the joint-space "
                f"YAMLs for spec {spec_name!r}; got {joint_dir!r}"
            )
        from verl_vla.models.gr00t_n1d6.utils import EMBODIMENTS

        try:
            spec = EMBODIMENTS[spec_name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown arena_joint_space_spec {spec_name!r}; known: {list(EMBODIMENTS)}"
            ) from exc

        return ArenaJointMapping.from_yaml_dir(
            spec,
            joint_dir,
            policy_yaml=_cfg_get(self.cfg, "arena_joint_space_policy_yaml", ArenaJointMapping.DEFAULT_POLICY_YAML),
            action_yaml=_cfg_get(self.cfg, "arena_joint_space_action_yaml", ArenaJointMapping.DEFAULT_ACTION_YAML),
            state_yaml=_cfg_get(self.cfg, "arena_joint_space_state_yaml", ArenaJointMapping.DEFAULT_STATE_YAML),
        )

    @property
    def joint_map(self) -> Optional[ArenaJointMapping]:
        """Embodiment <-> Arena-sim joint mapping (``None`` for identity), lazy on first use."""
        if self._joint_map is None:
            self._joint_map = self.build_joint_map()
        return self._joint_map

    @property
    def policy_action_dim(self) -> Optional[int]:
        """Recorded policy action width: the mapping's policy width, or ``None`` (identity
        joint-space) so the wrapper falls back to the sim ``action_dim``."""
        jm = self.joint_map
        return None if jm is None else jm.policy_dim

    def policy_to_sim_action(self, actions, device) -> torch.Tensor:
        tensor = torch.as_tensor(actions, dtype=torch.float32, device=device)
        jm = self.joint_map
        if jm is None:
            # Identity joint-space (G1 WBC): policy joints == sim joints, pass through.
            return tensor
        # Mapped (GR1): scatter the policy joints to the full sim action (rest -> 0).
        return jm.scatter_action(tensor)

    def extract_state(self, raw_obs, scene) -> np.ndarray:
        jm = self.joint_map
        if jm is not None:
            # Mapped (GR1): gather the policy-order state from the full sim joint state.
            robot_joint_pos = raw_obs["policy"]["robot_joint_pos"]  # (B, state_full_dim)
            if isinstance(robot_joint_pos, torch.Tensor):
                robot_joint_pos = robot_joint_pos.detach().cpu().numpy()
            else:
                robot_joint_pos = np.asarray(robot_joint_pos)
            return jm.gather_state(robot_joint_pos).astype(np.float32)  # (B, policy_dim)

        # Identity (G1 WBC): return robot_joint_pos as-is, with a defensive fallback for
        # the (essentially never hit) obs that lack it.
        policy_obs = raw_obs.get("policy", {}) if isinstance(raw_obs, dict) else {}
        if "robot_joint_pos" in policy_obs:
            state = policy_obs["robot_joint_pos"]
        else:
            parts = list(policy_obs.values())
            if not parts:
                return np.zeros((self.num_envs, self.state_dim), dtype=np.float32)
            tensors = [part if isinstance(part, torch.Tensor) else torch.as_tensor(part) for part in parts]
            state = torch.cat([tensor.reshape(tensor.shape[0], -1) for tensor in tensors], dim=-1)
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        return np.asarray(state, dtype=np.float32)



def _quat_xyzw_to_rotvec(quat_xyzw: np.ndarray) -> np.ndarray:
    """Quaternion ``[x, y, z, w]`` -> axis-angle rotation vector."""
    w = np.clip(quat_xyzw[..., 3:4], -1.0, 1.0)
    xyz = quat_xyzw[..., 0:3]
    angle = 2.0 * np.arccos(np.abs(w))
    den = np.sqrt(1.0 - w * w)
    small = den < 1e-8
    with np.errstate(invalid="ignore", divide="ignore"):
        rot = xyz / den * angle * np.sign(w)
    return np.where(small, np.zeros_like(xyz), rot)


def _rotvec_to_quat_xyzw(rotvec: torch.Tensor) -> torch.Tensor:
    """Axis-angle rotation vector -> quaternion ``[x, y, z, w]``."""
    angle = torch.norm(rotvec, dim=-1, keepdim=True).clamp(min=1e-8)
    axis = rotvec / angle
    half = angle * 0.5
    w = torch.cos(half)
    xyz = axis * torch.sin(half)
    return torch.cat([xyz, w], dim=-1)


class TaskSpaceEmbodiment(ArenaEmbodiment):
    """Task-space (end-effector / Cartesian) obs/action embodiment for Franka LIBERO.

    The policy is trained in rotvec (axis-angle) layout:
    ``pos(3) + rotvec(3) + gripper(1) = 7``. The sim Abs-IK action is
    ``pos(3) + quat_xyzw(4) + gripper(1) = 8``; quaternions use Isaac Lab 3.0
    ``(x, y, z, w)`` throughout. This adapter converts rotvec <-> quat_xyzw at
    the env boundary.
    """

    state_mode = "task_space"
    use_policy_action = True

    def __init__(self, cfg: Any, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)
        self.state_mode = str(_cfg_get(cfg, "arena_state_mode", None) or type(self).state_mode)
        self.action_dim = int(_cfg_get(cfg, "action_dim", 7))
        self.state_dim = int(_cfg_get(cfg, "state_dim", None) or self.action_dim)

    def add_cli_args(self, args: argparse.Namespace, cfg: Any) -> None:
        # LIBERO external-env knobs (ignored by other task-space envs).
        args.task_suite = str(_cfg_get(cfg, "libero_task_suite", "libero_10"))
        args.task_id = int(_cfg_get(cfg, "libero_task_id", 0))
        args.randomize_object_pose = bool(_cfg_get(cfg, "libero_randomize_object_pose", False))
        args.robot_init_noise_std = float(_cfg_get(cfg, "libero_robot_init_noise_std", 0.0))
        args.libero_in_lab_root = _cfg_get(cfg, "arena_libero_in_lab_root", None)
        args.libero_config_dir = _cfg_get(cfg, "arena_libero_config_dir", None)
        args.libero_assets_dir = _cfg_get(cfg, "arena_libero_assets_dir", None)
        args.libero_assembled_dataset_dir = _cfg_get(cfg, "arena_libero_assembled_dataset_dir", None)

    @property
    def policy_action_dim(self) -> int:
        return self.action_dim

    def policy_to_sim_action(self, actions, device) -> torch.Tensor:
        """Policy rotvec action -> sim Abs-IK action (pos + quat_xyzw + gripper)."""
        a = torch.as_tensor(actions, dtype=torch.float32, device=device)
        pos = a[..., :3]
        quat_xyzw = _rotvec_to_quat_xyzw(a[..., 3:6])
        grip = a[..., 6:7]
        return torch.cat([pos, quat_xyzw, grip], dim=-1)

    def extract_state(self, raw_obs, scene) -> np.ndarray:
        """Sim obs (pos + quat_xyzw + gripper) -> policy rotvec state."""
        del scene
        eef_pos, eef_quat_xyzw, gripper_state = self._split_libero_policy_obs(raw_obs)
        eef_rotvec = _quat_xyzw_to_rotvec(eef_quat_xyzw)
        return np.concatenate([eef_pos, eef_rotvec, gripper_state], axis=-1).astype(np.float32)

    def _split_libero_policy_obs(self, raw_obs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(eef_pos, eef_quat_xyzw, gripper_first_finger)`` from sim obs."""
        policy = raw_obs["policy"]
        if isinstance(policy, dict):
            eef_pose = policy["eef_pose"]
            gripper_pos = policy["gripper_pos"]
            if isinstance(eef_pose, torch.Tensor):
                eef_pose = eef_pose.detach().cpu().numpy()
            else:
                eef_pose = np.asarray(eef_pose)
            if isinstance(gripper_pos, torch.Tensor):
                gripper_pos = gripper_pos.detach().cpu().numpy()
            else:
                gripper_pos = np.asarray(gripper_pos)
            return eef_pose[..., :3], eef_pose[..., 3:7], gripper_pos[..., 0:1]
        if isinstance(policy, torch.Tensor):
            policy = policy.detach().cpu().numpy()
        else:
            policy = np.asarray(policy)
        eef_pose = policy[..., :7]
        return eef_pose[..., :3], eef_pose[..., 3:7], policy[..., 7:8]


# Registry mapping ``arena_state_mode`` -> the adapter *class* (this is routing, not
# config: the actual per-robot values all live in the simulator yaml). Every joint-space
# robot -- G1 WBC, GR1, and any future one -- is the same config-driven
# :class:`JointSpaceEmbodiment`; every task-space robot is :class:`TaskSpaceEmbodiment`.
_ARENA_EMBODIMENTS: dict[str, type[ArenaEmbodiment]] = {
    "joint_space": JointSpaceEmbodiment,
    "g1_wbc_joint": JointSpaceEmbodiment,
    "gr1_joint": JointSpaceEmbodiment,
    "eef_pose": TaskSpaceEmbodiment,
    "task_space": TaskSpaceEmbodiment,
}


def make_arena_embodiment(cfg: Any, num_envs: int = 1) -> ArenaEmbodiment:
    """Build the embodiment adapter selected by ``cfg.arena_state_mode``.

    Defaults to :data:`DEFAULT_ARENA_STATE_MODE` (``g1_wbc_joint``) so configs that
    predate the embodiment abstraction keep working unchanged.
    """
    state_mode = str(_cfg_get(cfg, "arena_state_mode", DEFAULT_ARENA_STATE_MODE))
    try:
        adapter_cls = _ARENA_EMBODIMENTS[state_mode]
    except KeyError as exc:
        raise ValueError(f"Unknown arena_state_mode {state_mode!r}; known: {sorted(_ARENA_EMBODIMENTS)}") from exc
    return adapter_cls(cfg, num_envs=num_envs)
