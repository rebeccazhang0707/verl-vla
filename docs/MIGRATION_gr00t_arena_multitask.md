# Multi-Task Migration Guide: GR00T + Isaac Lab Arena + SAC

> 配套文档：`docs/MIGRATION_gr00t_arena.md`（单任务迁移记录）、`docs/README_gr00t_arena_sac.md`（使用说明）。
> 本文档回答：**当前单任务实现中，哪些是多任务已就绪的、哪些是被硬编码成单任务的，做多任务需要改哪些文件、怎么改。**

---

## 0. TL;DR（结论先行）

多任务的**基础设施层（trainer / replay pool / 数据通路 / 指标）已经就绪**——`task_ids` 已从 dataset 流到
per-task replay pool（`SACReplayPool.task_pools`）和 per-task 指标（`compute_per_task_trajectory_metrics`）。

被硬编码成单任务的只有 **三层**：

| 层 | 文件 | 单任务硬编码点 |
|---|---|---|
| **环境** | `src/verl_vla/envs/arena_env/arena_env.py` | `self.task_description` 单字符串；`_wrap_obs` 用 `[self.task_description]*num_envs`；`reset_envs_to_state_ids` 忽略 `task_ids`；`get_all_state_ids` 返回 `range(num_envs)` |
| **数据集** | `scripts/prepare_arena_dataset.py` | `task_ids` 恒为 0、单 `prompt` |
| **模型** | `modeling_gr00t_sac.py` | `embodiment_id` 单常量；`sac_forward_actor/critic` `del task_ids`（**仅当跨 embodiment 多任务时才需要改**）|

**关键判断**：GR00T 通过 **language tokens** 做任务条件化，critic 用的是包含语言的 pooled backbone 特征
→ **同一 embodiment 下的多任务，模型层不需要 `task_ids`**（语言已经区分任务，replay pool 仍按 `task_ids`
做均衡采样）。因此多任务的主要工作量集中在 **环境层 + 数据集层**。

---

## 1. 先决问题：你要的是哪一种"多任务"？

Arena 多任务和 LIBERO 多任务有一个**模拟器层面的本质差异**，必须先确定方案：

| 方案 | 含义 | 模拟器约束 | 难度 |
|---|---|---|---|
| **A. 同场景同 embodiment，多指令/多布局** | 同一个 Arena example env（如 `put_item_in_fridge_and_close_door`），不同物体 / 摆放 / 语言指令 | 一个 vectorized env 内 `num_envs` 个并行实例可以**共享同一 scene cfg**，靠 placement_seed / object 变体 + 不同 language 区分 | **低** |
| **B. 多个 Arena example env（异构场景）** | 不同任务是不同的 Arena 场景（开冰箱 vs 摆盘 vs 关抽屉） | IsaacLab 一个 vectorized env 的 `num_envs` 实例**必须是同一个 scene cfg**，无法在 slot 间混合不同场景 | **高**（见 §4）|
| **C. 跨 embodiment 多任务** | 不同任务用不同机器人（GR1 / G1 / Franka） | 关节维度、`embodiment_id`、joint map 全不同 | **最高**（见 §5）|

> LIBERO 的多任务属于 B 的特例：它通过 `reconfigure_env_fns` 在 slot 级别切换 bddl，IsaacLab Arena
> 当前封装没有暴露等价的"逐 slot 重配场景"能力，所以 B 在 Arena 下要走不同的路线。

下面按方案分别给出改动清单。**绝大多数实际需求是方案 A**，先把 A 做扎实。

---

## 2. 已就绪、无需改动的部分（核对清单）

迁移时这些已经是多任务安全的，**不要重复造轮子**：

- **`SACReplayPool`**（`utils/replay_pool.py`）：`task_pools: dict[str, _DualPoolState]` 已按
  `_normalize_task_id(task_ids[idx])` 分桶，positive/negative 双池、跨任务均衡采样、save/load 全部 per-task。
- **Trainer**（`trainer/sac/sac_ray_trainer.py`）：
  - `_reset_envs` 已从 `gen_batch.non_tensor_batch["task_ids"]` 读取并通过
    `reset_prompts` 传给 `env_wg.reset_envs_to_state_ids`；
  - `meta_info["task_ids"]` → `info.task_ids` 写入 replay；
  - `compute_per_task_trajectory_metrics` 已按 `np.unique(task_ids)` 输出 per-task 成功率/轨迹长度。
- **模型 SAC 接口**：`sac_forward_actor/critic/get_critic_value` 都已带 `task_ids=None` 形参（签名就绪），
  critic 用 pooled backbone（含语言）→ 隐式任务条件化。
- **`build_inputs`**：`_wrap_obs` 调用 `adapter.build_inputs(full_image, state_26, task_descriptions)`，
  其中 `task_descriptions` **已经是一个 list**——只要喂入逐 env 不同的字符串即可，无需改 adapter。

---

## 3. 方案 A 改动清单（同场景多指令/多布局，推荐起点）

### 3.1 环境层 `arena_env.py`（核心改动）

把"实例级单任务描述"升级为"**逐 env-slot 任务描述 + 逐 slot 布局**"，对齐 LIBERO 的
`self.task_descriptions`（list）模式。

1. **状态从标量改为逐 slot 向量**
   - `self.task_description: str` → `self.task_descriptions: list[str]`（长度 `num_envs`）。
   - 新增 `self.task_ids: np.ndarray (num_envs,)`、可选 `self.state_ids`。

2. **`reset_envs_to_state_ids(state_ids_list, task_ids_list)` 真正消费 `task_ids`**（当前直接忽略）：
   ```python
   def reset_envs_to_state_ids(self, state_ids_list, task_ids_list):
       self.task_ids = np.asarray(task_ids_list)
       # task_id -> (prompt, object, placement_seed, ...) 的映射表
       self.task_descriptions = [self._task_table[t]["prompt"] for t in task_ids_list]
       self._per_env_placement = [self._task_table[t].get("placement", None) for t in task_ids_list]
       self._init_env()                 # 见下方关于"是否每次重建"的说明
       raw_obs, infos = self.env.reset()
       ...
   ```
   - 需要一个 **task 表**（`task_id -> {prompt, object/variant, placement_seed}`），由 cfg 注入
     （见 §3.3），而不是 `__init__` 里的单个 `arena_object` / `task_description`。

3. **`_wrap_obs` 用逐 slot 描述**：
   ```python
   # 之前：task_descriptions = [self.task_description] * self.num_envs
   task_descriptions = self.task_descriptions          # 已是 list[num_envs]
   inputs, _ = self.adapter.build_inputs(full_image, state_26, task_descriptions)
   ```
   `build_inputs` 已支持 list，模型 backbone 即可按 slot 做语言条件化。

4. **`get_all_state_ids` 不再是 dummy**：要让 trainer 能枚举 (task, trial) 组合并均衡覆盖所有任务，
   对齐 LIBERO 的 `cumsum_trial_id_bins` 思路：
   ```python
   def get_all_state_ids(self):
       # 例如每个 task 给 K 个布局变体；state_id 编码 (task_id, variant_id)
       return list(range(self.num_tasks * self.variants_per_task))
   ```
   并在 reset 时把 `state_id` 解码回 `(task_id, variant_id)`（若数据集已直接携带 `task_ids`，
   也可让 state_id 仅作占位、task_ids 直接生效——取决于 §3.2 的数据集设计）。

5. **同场景下避免每次 `_init_env()` 重建**（性能）：当前 `reset_envs_to_state_ids` 每次都
   `self._init_env()` 完整重建 Isaac 场景，非常昂贵。方案 A 同场景时应尽量只做
   **scene-level 软 reset + 改 placement/object 变体**，仅在 task 真正切换且需要不同 scene cfg 时才重建。
   可参考 LIBERO 的 `_reconfigure` 仅对变化的 slot 重配。

### 3.2 数据集 `prepare_arena_dataset.py`

把 `build_rows` 从"恒 0 单 prompt"升级为"逐 task 行 + 真实 task_ids"，对齐 `prepare_libero_dataset.py`：

```python
# 入参：tasks = [{"task_id": 0, "prompt": "...", "object": "...", "variants": K}, ...]
for task in tasks:
    for v in range(task["variants"]):
        rows.append({
            "data_source": ...,
            "prompt": task["prompt"],
            "state_ids": encode_state_id(task["task_id"], v),  # 或单调递增占位
            "task_ids": task["task_id"],                       # 不再恒 0
            "ability": "robot",
            "extra_info": {"task_description": task["prompt"], "arena_env_name": ..., ...},
        })
```

- **行数约束**仍需满足 `TRAIN_BATCH_SIZE * ROLLOUT_N == NUM_ENV_GPUS * NUM_STAGE * NUM_ENV`；
  多任务时建议每个 batch 内**均衡覆盖各 task**（让 replay pool 各桶都有数据）。
- `task_ids` 列必须是 trainer 期望的 `int64`（`_next_rollout_batch` 会 `np.asarray(..., dtype=np.int64)`）。

### 3.3 配置 / run 脚本

当前 run 脚本通过 `+env.train.arena_env_name / arena_object / kitchen_style / task_description`
注入**单任务**。多任务需要换成一个**任务表**结构（Hydra list/dict），例如：

```bash
+env.train.arena_tasks='[
  {task_id:0, prompt:"Place the bottle ...", object:ranch_dressing_hope_robolab},
  {task_id:1, prompt:"Put the can ...",      object:soda_can_variant},
  ...
]'
```

并在 `arena_env.__init__` 里解析成 `self._task_table`。`arena_env_name` 在方案 A 下仍是单值（同场景）。

### 3.4 模型层

**方案 A 无需改模型**。`embodiment_id` 同 embodiment 保持常量；任务区分完全靠 language。
`task_ids` 形参继续被忽略是**正确**的（不要强行接进 critic）。

### 3.5 测试

- `tests/envs/arena_env/arena_env_test.py`：新增"逐 slot 不同 `task_descriptions` 被正确传入
  `build_inputs`"、"`reset_envs_to_state_ids` 写入正确的 per-slot task_ids/prompt"、
  "`get_all_state_ids` 覆盖所有 (task, variant)"。
- 新增 trainer 级用例：构造 2 个 task_id 的 rollout batch，断言 `SACReplayPool.task_pools` 出现 2 个桶、
  `compute_per_task_trajectory_metrics` 输出 2 套指标。
- `prepare_arena_dataset.py`：断言生成的 parquet 含多个 distinct `task_ids`，且行数满足 batch 约束。

---

## 4. 方案 B 改动清单（异构 Arena 场景）

核心障碍：**一个 vectorized Isaac env 的 `num_envs` 实例必须同 scene cfg**。两条可行路线：

### 路线 B-1：每任务一个 env worker（推荐）
- 让 `env_worker.init_worker` 的 arena 分支按 **task 列表**为每个 task 建一个
  `EnvManager(env_cls=IsaacLabArenaEnv, ...)`（每个绑定不同 `arena_env_name`）。
- pipeline stage / GPU 分配需相应扩展；trainer 的 `task_ids` 路由到对应 worker。
- 改动面：`env_worker.py`（多 worker 构建 + 路由）、`reset_envs_to_state_ids`（按 task 分发）、
  容量规划（每任务独占若干 GPU）。
- 优点：场景隔离干净；缺点：GPU 成本随任务数线性增长。

### 路线 B-2：单 worker 内串行切换场景
- `reset_envs_to_state_ids` 收到新 task 时 `self.env.close()` + `omni.usd new_stage` + 用新
  `arena_env_name` 重新 `_init_env()`（当前代码已有 close/new_stage 雏形）。
- 一个 rollout 周期内**所有 slot 同任务**（即 `task_ids` 在一个 reset 内必须同值），任务在 reset 间切换。
- 改动面：`reset_envs_to_state_ids` 增加"整 worker 同 task"断言；dataset 需保证同一 reset 批次内
  `task_ids` 一致（类似 isaac 的 `assert len(set(stage_state_ids))==1`，但 arena 当前**故意没加**——
  方案 B-2 下要反过来加上 per-task 一致性约束）。
- 优点：不增加 GPU；缺点：场景重建开销大、任务间吞吐不均。

> 两条路线都需要 §3.1–3.3 的环境/数据集改动作为前提；B 只是在其之上解决"场景如何切换"。

---

## 5. 方案 C 改动清单（跨 embodiment）

最重的改动，仅在确有跨机器人需求时做：

1. **`embodiment.py`**：`GR1_ARENA` 是单 embodiment 的 `ArenaJointMapping`。需要一个
   `dict[embodiment_tag -> ArenaJointMapping]`，并为每个机器人提供 26/36/54 这类 joint-space YAML。
2. **`utils.py`**：`EMBODIMENTS` 当前只有 `GR1`，需注册 G1/Franka 等 spec（`state_group_dims`、
   `embodiment_id`）。
3. **模型 `modeling_gr00t_sac.py`**：`self.embodiment_id` 单常量需改为**逐样本** `embodiment_id`
   （`_state_features_impl` 里 `torch.full((B,), self.embodiment_id)` 改成从 obs/batch 读 per-sample id）。
   critic 的 `action_dim` / state 维度也随 embodiment 变化——这会破坏当前"单一 `critic_input_dim`"的假设，
   需要 padding 到 `max_action_dim`/`max_state_dim`（GR00T 本身按 128 padding，所以 critic 用 padded 维度
   即可统一，但 BC loss 的 `self.action_dim` 截断需 per-embodiment）。
4. **env / joint map**：`gather_state`(54→26) / `scatter_action`(26→36) 维度 per-embodiment。
5. **数据集**：增加 `embodiment_tag` 列，trainer/replay 需要按 (task_id, embodiment) 联合分桶。

> C 实际上接近"把 GR00T 多 embodiment 能力完整接进 RL"，建议作为独立大阶段，不要和 A/B 混做。

---

## 6. 顺带发现的疏漏 / 建议（与多任务相关或迁移遗留）

下列为审查中发现、建议在做多任务前一并修正的点（详见对话）：

1. **rollout 文档漂移**：`naive_rollout_gr00t.py` 顶部 docstring（约 26–28、40–43 行）仍写
   "env 每步 decode（decode_actions_flat → scatter_action）"，但 §8.J P0 修复后已改为
   **每 chunk decode 一次**。多任务前应同步更正，避免误导。
2. **`reset(env_idx)` 的"部分 reset"名不副实**：`self.env.reset()` 会重置所有 slot，
   而 metrics 只重置 `env_idx`。单任务无碍，多任务+逐 slot 任务切换时要小心 metric/任务对齐。
3. **`get_all_state_ids` 为 dummy `range(num_envs)`**：多任务必须改（§3.1.4），否则无法覆盖全部任务。
4. **`action_horizon` guard 上界依赖 `override_config.action_horizon=50`**，否则回退到
   `GR00TDim.ACTION_HORIZON=16` 会误拒 (16,50] 的 chunk。已在文档强调，多任务 run 脚本同样必须设置。
5. **`_record_metrics` 的 `reward = return/elapsed_steps`** 用全局 `any(elapsed>0)` 守卫，
   个别 slot `elapsed==0` 时该 slot 会出现除零→inf。多任务下各 slot 进度更易不齐，建议改成逐 slot 守卫。
6. **arena 故意不加 `assert len(set(stage_state_ids))==1`**：方案 B-2 下需要反向加"整 worker 同 task"约束。

---

## 7. 推荐落地顺序

1. **先做方案 A**（同场景多指令）：改 `arena_env.py`（逐 slot 描述 + 真实 task_ids 消费 + get_all_state_ids）
   → 改 `prepare_arena_dataset.py`（多 task_ids）→ run 脚本任务表 → 补测试。**不动模型、不动 trainer、不动 replay。**
2. 验证 per-task replay 分桶 + per-task 指标在 docker 端跑通（复用 §11.C 的 Gate 2，断言出现 ≥2 个 task 桶）。
3. 视需求再上方案 B（异构场景，优先 B-1 多 worker）。
4. 跨 embodiment（方案 C）作为独立阶段。
