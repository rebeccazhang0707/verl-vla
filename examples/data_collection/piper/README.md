# Piper data collection

Install the unified ROS, verl-vla, and LeRobot environment:

```bash
examples/data_collection/piper/setup.sh
conda activate verl-vla-piper
```

Start keyboard teleoperation or record a LeRobot dataset:

```bash
examples/data_collection/piper/run.sh
examples/data_collection/piper/run.sh record num_episodes=10
```

Arm model/CAN/reset configuration and arbitrary V4L2 cameras live in
`src/verl_vla/workflows/config/env/simulator/piper.yaml`. See
`docs/data-collection/piper/` for the complete hardware and operation guide.
