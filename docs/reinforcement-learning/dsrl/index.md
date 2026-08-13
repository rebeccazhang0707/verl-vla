# DSRL

DSRL trains a compact latent-noise steering actor while keeping the native VLA
policy frozen. The actor samples flow-matching initial noise conditioned on the
policy features and robot state, and SAC updates the steering module from
online replay.

```{toctree}
:maxdepth: 1

pi05/index
```
