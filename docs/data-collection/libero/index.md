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
| [LeRobot Leader Arm](lerobot-leader-arm.md) | Experimental | Experimental | Experimental |

Each device page contains its launch commands, dashboard controls, episode
lifecycle, and output locations.

For the best control experience, we recommend the XR Controller: its
motion-based input makes LIBERO manipulation smooth and intuitive. A gamepad
is the next best option when an XR device is unavailable. Keyboard control is
the most accessible option, but its discrete inputs generally provide a less
fluid manipulation experience.

LeRobot Leader Arm input is included as an experimental integration across all
three workflows. Because the leader arm and the LIBERO robot are heterogeneous
embodiments, their motion must be mapped across different control spaces, and
the resulting control may not feel as smooth as with the other input methods.

```{toctree}
:maxdepth: 1
:titlesonly:

installation
keyboard
gamepad
xr-controller
lerobot-leader-arm
```
