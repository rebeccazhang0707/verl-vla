# GR00T N1.6

These recipes apply DSRL to GR00T N1.6 by training a Transformer noise actor
over the flow-matching initial noise, plus its SAC critic, while leaving the
GR00T policy weights unchanged. They run on Isaac Lab Arena rather than the
MuJoCo LIBERO harness used by the PI0.5 recipes.

```{toctree}
:maxdepth: 1

arena-libero-spatial
```
