# LIBERO

## EGL rendering uses the wrong GPU

### Symptoms

On a multi-GPU machine, a LIBERO process may create its EGL rendering context
on a different physical GPU from the one assigned through
`CUDA_VISIBLE_DEVICES`. For example, an environment worker with
`CUDA_VISIBLE_DEVICES=2` may appear on GPU 1 in `nvidia-smi`. This can make
simulator workers overlap with model workers even when Ray places them in
separate GPU resource pools.

### Cause

robosuite 1.4.0 treats the CUDA device identifier in
`CUDA_VISIBLE_DEVICES` as an index into the list returned by
`eglQueryDevicesEXT()`. CUDA and EGL maintain separate device orderings, so
the same integer is not guaranteed to identify the same physical GPU.

### Apply the patch

Activate the Python environment used to launch verl-vla, then run:

```bash
python scripts/patch_robosuite_egl.py
```

The script patches robosuite in the active Python environment. It supports the
verified robosuite 1.4.0 source, validates both target files before modifying
them, and is safe to run more than once. It refuses to modify an unsupported
version or an unrecognized source tree.

Confirm that the patch is present:

```bash
python scripts/patch_robosuite_egl.py --check
```

The patched renderer queries `EGL_CUDA_DEVICE_NV` and selects the EGL device
that corresponds to the process-local CUDA ordinal. This preserves
`CUDA_VISIBLE_DEVICES` filtering and reordering without relying on the host's
EGL enumeration order.

If `MUJOCO_EGL_DEVICE_ID` was set as a previous workaround, remove it before
launching the simulator so that CUDA visibility controls device placement:

```bash
unset MUJOCO_EGL_DEVICE_ID
```

Reapply the patch after recreating the Python environment or reinstalling
robosuite.

### Verify device placement

In one terminal, monitor the GPUs:

```bash
watch -n 0.5 nvidia-smi
```

In another terminal, create a temporary EGL context on the desired GPU:

```bash
CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl python - <<'PY'
from robosuite.renderers.context.egl_context import EGLGLContext

context = EGLGLContext(16, 16, device_id=-1)
input("EGL context is active. Check nvidia-smi, then press Enter to exit.")
context.free()
PY
```

The additional graphics memory should appear on physical GPU 2. Change
`CUDA_VISIBLE_DEVICES` to verify another GPU. This check only validates EGL
device placement; it does not start a LIBERO task or a training run.
