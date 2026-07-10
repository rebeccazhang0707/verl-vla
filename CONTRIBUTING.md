# Contributing to verl-vla

## Commit Messages

Commit titles use the same module-first style as `verl`:

```text
[module] type: description
[module, module] type: description
[BREAKING][module] type: description
```

Examples:

```text
[teleop] feat: add gamepad calibration
[env, recorder] fix: preserve terminal signals across action chunks
[BREAKING][cfg] refactor: rename rollout configuration fields
```

Allowed types are:

- `feat`: add or extend functionality
- `fix`: correct incorrect behavior
- `refactor`: restructure code without changing its behavior
- `chore`: perform maintenance work
- `test`: add or update tests

Allowed modules are:

```text
trainer rollout worker env model data teleop recorder entrypoints cfg
docker ci doc perf misc
```

Follow these title rules:

- Use lowercase module and type names.
- Separate multiple modules with a comma followed by one space.
- Write the description in English, using an imperative verb with a lowercase first letter.
- Do not end the description with a period.
- Keep the title at or below 100 characters.
- Prefix an incompatible API, CLI, or configuration change with `[BREAKING]`.

Autosquash titles such as `fixup!`, `squash!`, and `amend!` are accepted when
the title they reference follows this convention. Git-generated merge and
revert titles are also accepted.

Install the repository hooks after cloning:

```bash
pip install pre-commit
pre-commit install
```
