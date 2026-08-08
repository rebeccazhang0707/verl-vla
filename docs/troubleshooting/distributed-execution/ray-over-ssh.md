# Ray over SSH

Use this setup when the cloud machine exposes only SSH and a local robot
machine must join its Ray cluster. It does not require a TAP device, a VPC
route, EasyTier, or `CAP_NET_ADMIN`. Ray traffic remains inside one encrypted
SSH connection, and no Ray port is exposed on a public interface.

This guide uses a local Piper robot as the concrete example. The SSH and Ray
connection is not Piper-specific: `piper` is only the custom resource label
used by the example and can be replaced with the label of another local robot
or environment worker.

```text
Cloud: Ray head and GPU workers (resource: cloud)
127.77.0.1
        |
        | one SSH connection with many -L and -R channels
        |
127.0.0.2
Local: robot environment worker (resource: piper)
```

## How it works

Ray normally opens several services and dynamically assigns worker ports. An
SSH tunnel cannot forward unknown ports, so the commands below assign every
cross-node Ray listener a fixed port:

| Node | Ray ports |
| --- | --- |
| Cloud | GCS `26379`, services `28076-28082`, workers `28100-28131` |
| Local | services `29076-29081`, workers `29100-29115` |

SSH then multiplexes all of these forwards over one connection:

- `-L` forwards the cloud listeners to the local machine, allowing the local
  worker to connect to the Ray head.
- `-R` forwards the local listeners to the cloud machine, allowing the head,
  object manager, and cloud workers to connect back to the local worker.

The two nodes advertise dedicated loopback addresses. On the local machine,
`127.77.0.1` reaches the cloud through the `-L` forwards. On the cloud machine,
`127.0.0.2` reaches the local worker through the `-R` forwards. Because both
addresses are in `127.0.0.0/8`, no virtual network interface is required.

Only listening ports require forwarding. TCP client source ports are temporary
and travel through the established connections, while node-local Ray ports do
not cross the tunnel. The fixed worker ranges bound the maximum concurrent Ray
workers; enlarge the ranges in both the Ray arguments and SSH forwards if more
worker ports are required.

Ray resource labels control placement. The cloud node advertises `cloud`, and
the local robot node advertises `piper`. A workflow can therefore keep model
workers on cloud GPUs while forcing the physical environment worker onto the
local machine.

## Requirements

- SSH key authentication from the local machine to the cloud machine works.
- The cloud SSH server enables `GatewayPorts clientspecified`, so reverse
  forwards can bind `127.0.0.2` without listening on the public interface.
- Python and Ray have exactly the same versions on both machines. No other
  dependency or environment layout needs to match.
- The fixed loopback addresses and ports listed above are unused.

Check the only required software-version contract on both machines:

```bash
python -c 'import platform, ray; print(platform.python_version(), ray.__version__)'
```

## Connect the nodes

The following commands assume `ray` resolves to the intended executable in
each terminal. Replace the SSH destination where indicated.

First, start the head in a cloud terminal:

```bash
CLOUD_WORKER_PORTS=$(seq -s, 28100 28131)
ray start --head \
  --node-ip-address=127.77.0.1 --port=26379 \
  --object-manager-port=28076 --node-manager-port=28077 \
  --runtime-env-agent-port=28078 --dashboard-agent-listen-port=28079 \
  --dashboard-agent-grpc-port=28080 --metrics-export-port=28081 \
  --ray-client-server-port=28082 --worker-port-list="$CLOUD_WORKER_PORTS" \
  --include-dashboard=false --num-cpus=8 --num-gpus=8 \
  --resources='{"cloud":1}'
```

Then, in a local terminal, create the bidirectional SSH forwards and join the
robot worker:

```bash
CLOUD_SSH_HOST=root@cloud.example.com
CLOUD_SSH_PORT=22
SSH_CONTROL=/tmp/vvla-ray-ssh.sock

SSH_FORWARDS=()
for port in 26379 {28076..28082} {28100..28131}; do
  SSH_FORWARDS+=( -L "127.77.0.1:$port:127.77.0.1:$port" )
done
for port in {29076..29081} {29100..29115}; do
  SSH_FORWARDS+=( -R "127.0.0.2:$port:127.0.0.2:$port" )
done

ssh -p "$CLOUD_SSH_PORT" -M -S "$SSH_CONTROL" -fN \
  -o ExitOnForwardFailure=yes "${SSH_FORWARDS[@]}" "$CLOUD_SSH_HOST"

LOCAL_WORKER_PORTS=$(seq -s, 29100 29115)
ray start --address=127.77.0.1:26379 \
  --node-ip-address=127.0.0.2 \
  --object-manager-port=29076 --node-manager-port=29077 \
  --runtime-env-agent-port=29078 --dashboard-agent-listen-port=29079 \
  --dashboard-agent-grpc-port=29080 --metrics-export-port=29081 \
  --worker-port-list="$LOCAL_WORKER_PORTS" --num-cpus=3 \
  --resources='{"piper":1}'
```

Verify the connection from the local terminal:

```bash
ray status --address=127.77.0.1:26379
```

The output should contain two active nodes and the custom resources `cloud`
and `piper`.

To disconnect, stop Ray locally, close the SSH connection, and run `ray stop
--force` once in the cloud terminal:

```bash
ray stop --force
ssh -p "$CLOUD_SSH_PORT" -S "$SSH_CONTROL" -O exit "$CLOUD_SSH_HOST"
```

## Submit work from the cloud

SSH to the cloud machine and point the workflow at the tunneled Ray head:

```bash
ssh -p 22 root@cloud.example.com
cd /path/to/verl-vla
export RAY_ADDRESS=127.77.0.1:26379
```

Add these placement overrides to a verl-vla command:

```bash
+ray_kwargs.ray_init._node_ip_address=127.77.0.1 \
cluster.resource.controller_label=cloud \
cluster.resource.model.resource_label=cloud \
cluster.resource.env.resource_label=piper
```

`_node_ip_address=127.77.0.1` makes the cloud driver advertise the tunneled
address. The resource labels place model work on the cloud and robot I/O on the
local node. Env-only workflows such as replay do not need the model override.

## Common failures

- `remote port forwarding failed`: enable `GatewayPorts clientspecified` in
  the active sshd configuration, or stop the process holding a fixed port.
- A node is visible but a worker remains pending: make sure the node has enough
  CPU/GPU resources for the requested placement bundle.
- The cloud driver hangs during `ray.init`: pass
  `+ray_kwargs.ray_init._node_ip_address=127.77.0.1`.
- A worker fails to start or deserialize a task: compare Python and Ray versions
  on both machines and make them identical.
- The local node disappears: check the SSH control connection and ensure an
  HTTP proxy is not intercepting SSH.
