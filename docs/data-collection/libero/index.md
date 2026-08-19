# LIBERO

LIBERO data collection supports browser-based and hardware input devices for
teleoperation, demonstration recording, and human intervention during policy
rollout.

Complete the [LIBERO installation](installation.md) before selecting a device
example. The installation is shared by all LIBERO data-collection workflows.

## Supported devices

| Device | Teleoperation | Demonstration Recording | DAgger |
| --- | --- | --- | --- |
| [Keyboard](keyboard.md) | Supported | Supported | Supported |
| [Gamepad](gamepad.md) | Supported | Supported | Supported |
| [XR Controller](xr-controller.md) | Supported | Supported | Supported |

Each device page contains its launch commands, dashboard controls, episode
lifecycle, and output locations.

For the best control experience, we recommend the XR Controller: its
motion-based input makes LIBERO manipulation smooth and intuitive. A gamepad
is the next best option when an XR device is unavailable. Keyboard control is
the most accessible option, but its discrete inputs generally provide a less
fluid manipulation experience.

```{toctree}
:maxdepth: 1
:titlesonly:

installation
keyboard
gamepad
xr-controller
```
