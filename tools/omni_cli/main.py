#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
import os
import yaml
import argparse
import subprocess
import json
import shlex
import tempfile
from pathlib import Path

from typing import Dict, Any, List, Tuple, Optional
from omni_cli.config_transform import transform_deployment_config
from omni_cli.config_transform import detect_file_encoding
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import tempfile
import shlex
import omni_cli.proxy
from omni_cli.mk_inventory_yml import add_node, rm_node
from omni_cli.omni_cfg import parse_node_name
from omni_cli.omni_cfg import parse_remaining_args
from omni_cli.omni_cfg import cfg_set_process
from omni_cli.omni_cfg import cfg_delete_process
from omni_cli.omni_inspect import print_node_config

_DEFAULT_DEPLOY_PATH = None

def execute_command(command):
    """Execute the ansible command"""
    process = subprocess.Popen(
        command,
        shell=True
    )

    return_code = process.wait()
    if return_code != 0:
        print(f"Deployment failed with return code {return_code}")

    return return_code

def _walk_hosts(node: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Flatten an Ansible-style inventory dict into [(host, hostvars)]."""
    out: List[Tuple[str, Dict[str, Any]]] = []
    if not isinstance(node, dict):
        return out
    for h, vars_ in (node.get("hosts") or {}).items():
        out.append((h, vars_ or {}))
    for _g, child in (node.get("children") or {}).items():
        out.extend(_walk_hosts(child))
    return out

def _double_quotes(s: str) -> str:
    """Wrap value in double quotes for a safe shell arg."""
    s = str(s)
    s = s.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`")
    return f'"{s}"'

def _build_export_block(env: Dict[str, Any]) -> str:
    """Build a sequence of export lines."""
    items_plain = [(k, v) for k, v in env.items() if "$" not in str(v)]
    items_refs  = [(k, v) for k, v in env.items() if "$" in str(v)]

    lines = []
    for k, v in items_plain + items_refs:
        if v is None:
            v = ""
        lines.append(f'export {k}={_double_quotes(v)}')
    return "\n".join(lines)

def _build_json_args(cfg: Dict[str, Any]) -> str:
    """Turn a dict into a JSON-like string"""
    parts: List[str] = []
    for k, v in cfg.items():
        key = json.dumps(str(k))  # always quoted JSON key
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("$"):              # let shell expand
                parts.append(f"{key}:{s}")
            else:                              # normal JSON string
                parts.append(f"{key}:{json.dumps(s)}")
        elif isinstance(v, (int, float)):
            parts.append(f"{key}:{v}")
        else:
            parts.append(f"{key}:{json.dumps(v)}")
    return "{" + ", ".join(parts) + "}"

def _build_string_args(extra_args: dict) -> str:
    """Convert extra-args dict to CLI string"""
    parts = []
    for k, v in extra_args.items():
        if v == "":
            parts.append(f"--{k}")
        else:
            parts.append(f"--{k} {v}")
    return " ".join(parts)

def _build_args_line(args: Dict[str, Any]) -> str:
    """
    Build a flat CLI argument string like:
      --tp "16" --kv-transfer-config "{\"kv_connector\":\"...\",\"kv_rank\":$KV_RANK}"
    Rules:
      - If a value is None -> boolean-like flag (emit `--flag` only)
      - Else -> `--flag "value"` where value is safely double-quoted
      - Assumes caller passes kv-transfer-config as a JSON string already
    """
    parts = []
    for k, v in (args or {}).items():
        flag = f"--{k}"
        if v is None:
            continue
        elif v == "":
            parts.append(flag)
        elif k == "kv-transfer-config" and isinstance(v, dict):
            inline = _build_json_args(v)
            parts.append(f"{flag} {_double_quotes(inline)}")
        elif k == "extra-args" and isinstance(v, dict):
            inline = _build_string_args(v)
            parts.append(f"{flag} {_double_quotes(inline)}")
        elif k == "additional-config" and isinstance(v, dict):
            inline = _build_json_args(v)
            parts.append(f"{flag} {_double_quotes(inline)}")
        else:
            parts.append(f"{flag} {_double_quotes(v)}")
    return " ".join(parts)

def _verify_and_fix_env_vars(
    inventory: Dict[str, Any],
    inventory_path: str = None,
) -> List[str]:
    """
    Detect port conflicts per machine (same IP). If conflicts found, bump by `offset` repeatedly until unique.
    """

    port_vars = ["API_PORT", "MASTER_PORT", "VLLM_LLMDATADIST_ZMQ_PORT"]
    offset: int = 16
    all_hosts = _walk_hosts(inventory.get("all", inventory))  # {host: {vars}}

    # group by IP
    ip_to_hosts = {}
    for host, hv in all_hosts:
        env = hv.get("env", {}) or {}
        ip = hv.get("ansible_host", {})
        ip_to_hosts.setdefault(ip, []).append((host, env))

    # per IP, ensure uniqueness for each monitored port var
    for ip, items in ip_to_hosts.items():
        for pv in port_vars:
            used_ports: Dict[int, str] = {}  # port -> host that currently holds it

            conflicts = []
            # identify conflicts
            for host, env in items:
                port = int(env.get(pv)) if pv in env and str(env.get(pv)).isdigit() else None
                if port is None:
                    print(f"[warn] host={host} has no {pv}; skipped.")
                    continue
                if port in used_ports:
                    conflicts.append((host, env, port))
                else:
                    used_ports[port] = host

            # resolve conflicts by bumping with `offset` until free
            for host, env, original_port in conflicts:
                new_port = original_port
                while True:
                    new_port += offset
                    if new_port not in used_ports:
                        env[pv] = str(new_port)
                        used_ports[new_port] = host
                        print(f"[fix] ip={ip} {pv} conflict: host={host} {original_port} -> {new_port}")
                        break
    need_overwrite_inv = len(conflicts) > 0

    # calculate pod num, server ip list and server offset, kv rank
    kv_rank_dict, kv_rank = {}, 0
    prefill_pod_num = 0
    server_offset_dict, server_offset = {}, 0
    server_ip_list_temp = []
    host_ip_dict = {}
    num_server_dict, num_dp_dict_p, num_dp_dict_d = {}, {}, {}
    total_dp_d = 0
    for host, hv in inventory['all']['children']['P']['hosts'].items():
        prefill_pod_num += 1
        server_offset_dict[host] = server_offset  # for P always 0
        host_ip_dict[host] = hv.get('host_ip', None)
        kv_rank_dict[host] = kv_rank
        kv_rank += 1
        device_count = hv.get('ascend_rt_visible_devices','').count(',') + 1
        tp = int(hv.get('args', {}).get('tp', 16))
        num_server_dict[host] = device_count // tp
        num_dp_dict_p[host] = device_count // tp
    for host, hv in inventory['all']['children']['D']['hosts'].items():
        ip = hv.get('ansible_host', '')
        if ip:
            server_ip_list_temp.append(f"{ip}")
        server_offset_dict[host] = server_offset
        device_count = hv.get('ascend_rt_visible_devices','').count(',') + 1
        server_offset += device_count
        host_ip_dict[host] = hv.get('host_ip', None)
        kv_rank_dict[host] = kv_rank
        tp = int(hv.get('args', {}).get('tp', 1))
        num_server_dict[host] = device_count // tp
        total_dp_d += device_count // tp
        num_dp_dict_d[host] = total_dp_d

    # update num_dp_dict
    num_dp_dict_d = {k: total_dp_d for k in num_dp_dict_d.keys()}
    num_dp_dict = {**num_dp_dict_p, **num_dp_dict_d}
    server_ip_list = ','.join(server_ip_list_temp)

    ## update inventory
    need_overwrite_inv = False
    for host, hv in all_hosts:
        if "PREFILL_POD_NUM" in hv.get("env", {}):
            if hv.get("env", {}).get("PREFILL_POD_NUM") != prefill_pod_num:
                need_overwrite_inv = True
                hv.get("env", {})["PREFILL_POD_NUM"] = prefill_pod_num
                print(f"[info] host={host} PREFILL_POD_NUM set to {prefill_pod_num}")
        if "SERVER_IP_LIST" in hv.get("env", {}):
            if hv.get("env", {}).get("SERVER_IP_LIST") != server_ip_list:
                need_overwrite_inv = True
                hv.get("env", {})["SERVER_IP_LIST"] = server_ip_list
                print(f"[info] host={host} SERVER_IP_LIST set to {server_ip_list}")
        if "SERVER_OFFSET" in hv.get("env", {}):
            server_offset = server_offset_dict.get(host, None)
            if server_offset is not None and hv.get("env", {}).get("SERVER_OFFSET") != server_offset:
                hv.get("env", {})["SERVER_OFFSET"] = server_offset
                need_overwrite_inv = True
                print(f"[info] host={host} SERVER_OFFSET set to {server_offset}")
        if "KV_RANK" in hv.get("env", {}):
            kv_rank = kv_rank_dict.get(host, None)
            if kv_rank is not None and hv.get("env", {}).get("KV_RANK") != kv_rank:
                hv.get("env", {})["KV_RANK"] = kv_rank
                need_overwrite_inv = True
                print(f"[info] host={host} KV_RANK set to {kv_rank}")
        if "HOST_IP" in hv.get("env", {}):
            host_ip = host_ip_dict.get(host, None)
            if host_ip is not None and hv.get("env", {}).get("HOST_IP") != host_ip:
                hv.get("env", {})["HOST_IP"] = host_ip
                need_overwrite_inv = True
                print(f"[info] host={host} HOST_IP set to {host_ip}")
        if "num-servers" in hv.get("args", {}):
            num_server = num_server_dict.get(host, None)
            if num_server is not None and hv.get("args", {}).get("num-servers") != num_server:
                hv.get("args", {})["num-servers"] = num_server
                need_overwrite_inv = True
                print(f"[info] host={host} num-servers set to {num_server}")
        if "num-dp" in hv.get("args", {}):
            num_dp = num_dp_dict.get(host, None)
            if num_dp is not None and hv.get("args", {}).get("num-dp") != num_dp:
                hv.get("args", {})["num-dp"] = num_dp
                need_overwrite_inv = True
                print(f"[info] host={host} num-dp set to {num_dp}")

    if need_overwrite_inv:
        with open(inventory_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(inventory, f, default_flow_style=False, sort_keys=False)
        print(f"[info] inventory written back to {inventory_path}")



def omni_ranktable(inventory):
    cur_dir = os.path.dirname(__file__)
    cmd = "ansible-playbook -i " + str(inventory) + " " + str(cur_dir) + "/configs/generate_ranktable.yml"
    os.system(cmd)

def omni_cli_start(
    inventory_path: str = "./serving_profiles.yml",
    host_pattern: Optional[str] = None,   # e.g., "127.0.0.1"
    role_filter: Optional[str] = None,    # e.g., "P" or "D"
    python_bin: str = "python",
    entry_py: str = "start_api_servers.py"
) -> None:
    """
    Read inventory YAML, generate a per-host bash script, and run it via:
      ansible <host> -i <inventory> -m script -a <script_path>
    """
    omni_ranktable(inventory_path)

    inv_file = Path(inventory_path).expanduser().resolve()
    with open(inventory_path, "r", encoding="utf-8") as f:
        inv = yaml.safe_load(f)
    _verify_and_fix_env_vars(inv, inv_file)

    omni_cli.proxy.omni_run_proxy(inventory_path)

    all_hosts = _walk_hosts(inv.get("all", inv))
    if not role_filter:
        role_filter = ["prefill", "decode"]
    selected: List[Tuple[str, Dict[str, Any]]] = []
    for host, hv in all_hosts:
        if host_pattern and host != host_pattern:
            continue
        if role_filter and hv.get("env").get("ROLE") not in role_filter:
            continue
        selected.append((host, hv))

    if not selected:
        raise RuntimeError("No matching hosts found with given filters.")

    for host, hv in selected:
        env: Dict[str, Any] = hv.get("env", {}) or {}
        args: Dict[str, Any] = hv.get("args", {}) or {}
        container_name: str = hv.get("container_name")

        code_path = str(env.get("CODE_PATH") or "").strip()
        log_path = str(env.get("LOG_PATH") or "").strip()
        log_path = f"{log_path}/{host.replace('.', '_')}"
        env["LOG_PATH"] = log_path

        export_block = _build_export_block(env)
        args_line = _build_args_line(args)

        with tempfile.NamedTemporaryFile(
            "w", delete=False,
            dir="./",
            prefix=f"omni_start_{host.replace('.', '_')}_",
            suffix=".sh"
        ) as tf:
            script_path = Path(tf.name)
            tf.write("#!/usr/bin/env bash\n")
            tf.write("set -euo pipefail\n\n")

            tf.write(f"docker exec -i {shlex.quote(container_name)} bash -s <<'EOF'\n")
            tf.write("source ~/.bashrc\n\n")

            tf.write(f"if [ ! -d {shlex.quote(log_path)} ]; then\n")
            tf.write(f"  mkdir -p {shlex.quote(log_path)}\n")
            tf.write("fi\n")

            tf.write("# Export environment variables\n")
            tf.write(export_block + "\n\n")
            tf.write(f'echo "{export_block}\n" > {log_path}/omni_cli.log\n\n')

            tf.write("# Exec the command\n")
            tf.write(f"cd {_double_quotes(code_path)}/tools/scripts\n\n")
            tf.write(f'echo "cd {_double_quotes(code_path)}/tools/scripts" >> {log_path}/omni_cli.log\n\n')
            tf.write(f"{python_bin} {entry_py} {args_line} >> {log_path}/omni_cli.log 2>&1 &\n")
            tf.write(f'echo "{python_bin} {entry_py} {args_line} >> {log_path}/omni_cli.log 2>&1 &" >> {log_path}/omni_cli.log\n')
            tf.write("EOF\n")

        os.chmod(script_path, 0o755)

        cmd = (
            f"ansible {shlex.quote(host)} "
            f"-i {shlex.quote(str(inv_file))} "
            f"-m script "
            f"-a {shlex.quote(str(script_path))}"
        )

        try:
            execute_command(cmd)
        finally:
            try:
                script_path.unlink(missing_ok=True)
            except Exception:
                pass

def omni_cli_stop(
    inventory_path: str = "./serving_profiles.yml",
    host_pattern: Optional[str] = None,   # e.g., "127.0.0.1"
    role_filter: Optional[str] = None,    # e.g., "P" or "D"
) -> None:
    """kill python and vllm processes in the containers
    """
    inv_file = Path(inventory_path).expanduser().resolve()
    with open(inv_file, "r", encoding="utf-8") as f:
        inv = yaml.safe_load(f)

    all_hosts = _walk_hosts(inv.get("all", inv))

    if not role_filter:
        role_filter = ["prefill", "decode"]
    selected: List[Tuple[str, Dict[str, Any]]] = []
    for host, hv in all_hosts:
        if host_pattern and host != host_pattern:
            continue
        if role_filter and hv.get("env").get("ROLE") not in role_filter:
            continue
        selected.append((host, hv))

    if not selected:
        raise RuntimeError("No matching hosts found with given filters.")

    for host, hv in selected:
        env: Dict[str, Any] = hv.get("env", {}) or {}
        container_name: str = hv.get("container_name")

        with tempfile.NamedTemporaryFile(
            "w", delete=False,
            dir="./",
            prefix=f"omni_stop_{host.replace('.', '_')}_",
            suffix=".sh"
        ) as tf:
            script_path = Path(tf.name)

            tf.write("#!/usr/bin/env bash\n")
            tf.write(f"docker exec -i {shlex.quote(container_name)} bash -s <<'EOF'\n")
            tf.write("pkill -9 python || true\n")
            tf.write("pkill -9 vllm   || true\n")
            tf.write("EOF\n")
        os.chmod(script_path, 0o755)

        cmd = (
            f"ansible {shlex.quote(host)} "
            f"-i {shlex.quote(str(inv_file))} "
            f"-m script "
            f"-a {shlex.quote(str(script_path))}"
        )

        try:
            execute_command(cmd)
        finally:
            try:
                script_path.unlink(missing_ok=True)
                pass
            except Exception:
                pass

def get_host_groups(inv_data: dict) -> Dict[str, List[str]]:
    """Build host-to-group mappings"""
    host_groups = {}

    # Traverse the entire inventory structure
    def traverse(node, current_groups=None):
        if current_groups is None:
            current_groups = []

        # Dealing with the host of the current node
        hosts = node.get("hosts", {})
        for host, host_vars in hosts.items():
            # Merge current groups and host custom groups
            merged_groups = current_groups.copy()
            if "groups" in host_vars:
                merged_groups.extend(host_vars["groups"])
            host_groups[host] = list(set(merged_groups))  # 去重

        # Handle child nodes
        children = node.get("children", {})
        for child_name, child_data in children.items():
            # Child nodes inherit the parent node's group and add their own group name
            child_groups = current_groups + [child_name]
            traverse(child_data, child_groups)

    # Traversing from the root node
    traverse(inv_data.get("all", inv_data))
    return host_groups

def sync_dev(
    inventory_path: str = "omni_cli/configs/servering_profiles.yml",
    dry_run: bool = False,
    code_path: str = None
) -> None:
    """Sync code to all relevant hosts and containers"""
    if not code_path:
        print("Error: code_path is required")
        return

    # Read inventory file
    with open(inventory_path, 'r') as f:
        inventory_data = yaml.safe_load(f)

    # Get host-group mapping
    host_groups = get_host_groups(inventory_data)

    # Get all hosts and their variables using _walk_hosts
    all_hosts = dict(_walk_hosts(inventory_data.get("all", inventory_data)))

    # Identify all hosts that need processing
    p_hosts = set()
    d_hosts = set()
    c_hosts = set()

    for host, groups in host_groups.items():
        if 'P' in groups:
            p_hosts.add(host)
        if 'D' in groups:
            d_hosts.add(host)
        if 'C' in groups:
            c_hosts.add(host)

    # Find C nodes that need processing (not in P or D)
    c_hosts_to_process = c_hosts - (p_hosts | d_hosts)

    # All hosts that need processing
    all_target_hosts = p_hosts | d_hosts | c_hosts_to_process

    # Create temporary script file
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sh") as tf:
        script_path = tf.name
        
        # Write script header
        tf.write("#!/bin/bash\n\n")
        tf.write(f"CODE_SRC=\"{code_path}\"\n\n")

        # Create directories and sync code to all target hosts
        for host in all_target_hosts:
            host_vars = all_hosts.get(host, {})
            # Get actual connection address for the host
            host_addr = host_vars.get("ansible_host", host)
            
            # Get SSH key path (if exists)
            ssh_private_key = host_vars.get("ansible_ssh_private_key_file", "")
            ssh_user = host_vars.get("ansible_user", "")
            
            # Build SSH command prefixes
            ssh_prefix = "ssh"
            rsync_prefix = "rsync -avz --delete"
            
            if ssh_private_key:
                ssh_prefix = f"ssh -i {ssh_private_key}"
                rsync_prefix = f"rsync -avz --delete -e 'ssh -i {ssh_private_key}'"
            
            if ssh_user:
                ssh_prefix = f"{ssh_prefix} -l {ssh_user}"
                rsync_prefix = f"{rsync_prefix} -e 'ssh -l {ssh_user}'"
                if ssh_private_key:
                    rsync_prefix = f"rsync -avz --delete -e 'ssh -i {ssh_private_key} -l {ssh_user}'"

            tf.write(f"echo \"Creating directory on {host}\"\n")
            tf.write(f"{ssh_prefix} {host_addr} \"mkdir -p {code_path}\"\n\n")
            
            tf.write(f"echo \"Syncing code to {host}\"\n")
            tf.write(f"{rsync_prefix} {code_path}/omniinfer/ {host_addr}:{code_path}/omniinfer/\n\n")
        
            # Handle docker cp for all hosts that need it
            container_name = host_vars.get("container_name", "")
            if container_name:
                tf.write(f"echo \"Copying code to container on {host}\"\n")
                tf.write(f"{ssh_prefix} {host_addr} \"docker cp {code_path}/omniinfer {container_name}:/workspace/\"\n\n")
            else:
                print(f"Missing container_name for host {host}")

    # Set script execution permissions
    os.chmod(script_path, 0o755)

    # Execute script or display dry run information
    if dry_run:
        print("Dry run: would execute the following script:")
        with open(script_path, 'r') as f:
            print(f.read())
    else:
        print("Executing sync script...")
        try:
            result = subprocess.run([script_path], capture_output=True, text=True, check=True)
            print("Script output:")
            print(result.stdout)
            if result.stderr:
                print("Script errors:")
                print(result.stderr)
            print("\nSync process completed.")
        except subprocess.CalledProcessError as e:
            print(f"Sync process failed with return code {e.returncode}")
            print(f"Error output: {e.stderr}")
        finally:
            # Clean up temporary file
            try:
                os.unlink(script_path)
            except:
                pass

def install_dev(
    inventory_path: str = "omni_cli/configs/servering_profiles.yml",
    dry_run: bool = False,
) -> None:
    """Install code inside container"""

    inv_file = Path(inventory_path).expanduser().resolve()

    # Load inventory
    with open(inv_file, "r", encoding="utf-8") as f:
        inv = yaml.safe_load(f)

    all_hosts = _walk_hosts(inv.get("all", inv))
    host_groups = get_host_groups(inv)

    if not all_hosts:
        print("Warning: No hosts found in inventory.")
        return

    # Update command templates
    docker_update_code_cmd = """
    /bin/bash -c '. ~/.bashrc 
    export http_proxy=http://10.155.96.5:8081
    export https_proxy=http://10.155.96.5:8081
    sed -i s#https://pypi.tuna.tsinghua.edu.cn/simple#https://mirrors.tools.huawei.com/pypi/simple#g /root/.config/pip/pip.conf && sed -i s#pypi.tuna.tsinghua.cn#mirrors.tools.huawei.com#g /root/.config/pip/pip.conf
    pip install setuptools_scm
    cd /workspace/omniinfer/infer_engines 
    git config --global --add safe.directory /workspace/omniinfer/infer_engines/vllm 
    bash bash_install_code.sh 
    pip uninstall vllm -y 
    pip uninstall omniinfer -y 
    cd vllm 
    SETUPTOOLS_SCM_PRETEND_VERSION=0.9.0 VLLM_TARGET_DEVICE=empty pip install -e . --no-deps --no-build-isolation 
    cd ../../ 
    pip install -e . --no-deps --no-build-isolation 
    > ${LOG_PATH}/{{ inventory_hostname }}/pip.log'
"""
    for host, hv in all_hosts:
        groups = host_groups.get(host, [])
        print(f"\nProcessing host: {host} (groups: {groups})")

        # Determine the role (P, D, or C)
        role = "C"  # Default
        if 'P' in groups:
            role = "P"
        elif 'D' in groups:
            role = "D"
        elif 'C' in groups:
            role = "C"

        # Only groups P and D are processed
        if role not in ['P', 'D']:
            print(f"Skipping host {host} (not in P or D group)")
            continue

        # Get environment variables and other host variables
        env = hv.get("env", {})
        log_path = env.get("LOG_PATH")
        container_name = hv.get("container_name")

        # Check the necessary variables
        if not log_path:
            print(f"Warning: LOG_PATH not defined for host {host}, skipping")
            continue
        if not container_name:
            print(f"Warning: container_name not defined for host {host}, skipping")
            continue

        # Create a temporary script file
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sh") as tf:
            script_path = Path(tf.name)

            # Write scripted content
            tf.write("#!/bin/bash\n")
            tf.write("set -euo pipefail\n\n")

            # Export environment variables
            tf.write("# Export environment variables\n")
            for key, value in env.items():
                tf.write(f"export {key}={shlex.quote(str(value))}\n")

            # Add role-specific variables
            tf.write("\n# Role-specific variables\n")
            tf.write(f"export CONTAINER_NAME={shlex.quote(container_name)}\n")

            # Update code inside container - fix command formatting
            tf.write("\n# Update code inside container\n")
            # Replace the variable in the command
            update_cmd = docker_update_code_cmd.replace(
                "${LOG_PATH}/{{ inventory_hostname }}/pip.log",
                f"{log_path}/{host}/pip.log"
            )
            tf.write(f"echo \"Updating code inside container {container_name}\"\n")
            tf.write(f"docker exec {shlex.quote(container_name)} /bin/bash -c {shlex.quote(update_cmd)}\n")
            os.chmod(script_path, 0o755)

            # Print script content in dry run mode
            if dry_run:
                print(f"=== DRY RUN: Script for host {host} ===")
                with open(script_path, "r") as script_file:
                    print(script_file.read())
                print("===")

        if not dry_run:
            try:
                # Build ansible commands
                cmd = (
                    f"ansible {shlex.quote(host)} "
                    f"-i {shlex.quote(str(inv_file))} "
                    f"-m script "
                    f"-a {shlex.quote(str(script_path))}"
                )
                execute_command(cmd)
            finally:
                # Clean up temporary files
                try:
                    script_path.unlink(missing_ok=True)
                except Exception as e:
                    print(f"Warning: Failed to delete temp file: {e}")
        else:
            # In dry run mode, only the commands we will run are displayed
            print(f"DRY RUN: Would execute for host {host}:")
            print(f"  ansible {host} -i {inv_file} -m script -a {script_path}")

    print("\nInstall process completed.")

def get_default_deploy_path():
    """Get or create the default deployment path (file is created only on first call)"""
    global _DEFAULT_DEPLOY_PATH
    
    # Return immediately if already set
    if _DEFAULT_DEPLOY_PATH is not None:
        return _DEFAULT_DEPLOY_PATH
    
    # Create default file and save path
    current_dir = Path.cwd()
    deploy_path = current_dir / "servering_profiles.yml"
    
    if not deploy_path.exists():
        # Create default inventory structure
        default_inventory = {
            "all": {
                "children": {
                    "P": {"hosts": {}},
                    "D": {"hosts": {}},
                    "C": {"hosts": {}}
                }
            }
        }
        
        # Write to file
        with open(deploy_path, "w") as f:
            yaml.dump(default_inventory, f)
        
        print(f"Created default inventory file at: {deploy_path}")
    
    # Save as absolute path
    _DEFAULT_DEPLOY_PATH = deploy_path.resolve()
    return _DEFAULT_DEPLOY_PATH

def _walk_hosts(node: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Flatten an Ansible-style inventory dict into [(host, hostvars)]."""
    out: List[Tuple[str, Dict[str, Any]]] = []
    if not isinstance(node, dict):
        return out
    for h, vars_ in (node.get("hosts") or {}).items():
        out.append((h, vars_ or {}))
    for _g, child in (node.get("children") or {}).items():
        out.extend(_walk_hosts(child))
    return out

def run_docker_containers(
    inventory_path: str = "omni_cli/configs/servering_profiles.yml",
    dry_run: bool = False,
) -> None:
    """
    Run Docker containers for all hosts in inventory using variables from inventory.
    Generates and executes temporary scripts for each host.
    """
    inv_file = Path(inventory_path).expanduser().resolve()

    # Load inventory
    with open(inv_file, "r", encoding="utf-8") as f:
        inv = yaml.safe_load(f)

    all_hosts = _walk_hosts(inv.get("all", inv))
    host_groups = get_host_groups(inv)

    if not all_hosts:
        print("Warning: No hosts found in inventory.")
        return

    # Base Docker command template
    base_docker_run_cmd = """docker run -it --shm-size=500g \\
        --net=host \\
        --privileged=true \\
        -u root \\
        -w /data \\
        --device=/dev/davinci_manager \\
        --device=/dev/hisi_hdc \\
        --device=/dev/devmm_svm \\
        --entrypoint=bash \\
        -v /data:/data \\
        -v /tmp:/tmp \\
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \\
        -v /usr/local/dcmi:/usr/local/dcmi \\
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \\
        -v /etc/ascend_install.info:/etc/ascend_install.info \\
        -v /usr/local/sbin:/usr/local/sbin \\
        -v /etc/hccn.conf:/etc/hccn.conf \\
        -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \\
        -v $LOG_PATH:$LOG_PATH \\
        -v /tmp/ranktable_save_path:/tmp/ranktable_save_path \\
        -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime"""

    for host, hv in all_hosts:
        print(f"\nProcessing host: {host}")

        # Determine role
        groups = host_groups.get(host, [])
        role = "C"
        if 'P' in groups:
            role = "P"
        elif 'D' in groups:
            role = "D"
        elif 'C' in groups:
            role = "C"

        # Get environment variables and host variables
        env = hv.get("env", {})
        log_path = env.get("LOG_PATH")
        model_path = env.get("MODEL_PATH")
        docker_image_id = hv.get("DOCKER_IMAGE_ID")

        # Check required variables
        if not log_path:
            print(f"Warning: LOG_PATH not defined for host {host}, skipping")
            continue
        if not docker_image_id:
            print(f"Warning: DOCKER_IMAGE_ID not defined for host {host}, skipping")
            continue

        # For P and D roles, MODEL_PATH is required
        if role in ['P', 'D'] and not model_path:
            print(f"Warning: MODEL_PATH not defined for host {host} (role {role}), skipping")
            continue

        # Create temporary script file
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sh") as tf:
            script_path = Path(tf.name)

            # Write script content
            tf.write("#!/bin/bash\n")
            tf.write("set -euo pipefail\n\n")

            # Export environment variables
            tf.write("# Export environment variables\n")
            for key, value in env.items():
                tf.write(f"export {key}={shlex.quote(str(value))}\n")

            # Add role-specific variables
            tf.write("\n# Role-specific variables\n")
            tf.write(f"export DOCKER_NAME={shlex.quote(hv.get('container_name', ''))}\n")
            tf.write(f"export DOCKER_IMAGE_ID={shlex.quote(str(docker_image_id))}\n")

            # Clean up existing container
            tf.write("\n# Cleanup existing container\n")
            tf.write("if docker inspect --format='{{.Name}}' \"$DOCKER_NAME\" &>/dev/null; then\n")
            tf.write("    docker stop \"$DOCKER_NAME\"\n")
            tf.write("    docker rm -f \"$DOCKER_NAME\"\n")
            tf.write("fi\n")

            # Build Docker run command
            tf.write("\n# Build Docker run command\n")
            tf.write(f"docker_cmd={shlex.quote(base_docker_run_cmd)}\n")

            # Add MODEL_PATH mount for P and D roles
            if role in ['P', 'D'] and model_path:
                tf.write("docker_cmd+=\" -v $MODEL_PATH:$MODEL_PATH\"\n")
            
            tf.write("docker_cmd+=\" -d --name $DOCKER_NAME $DOCKER_IMAGE_ID\"\n")

            # Execute command
            tf.write("\n# Execute Docker run command\n")
            tf.write("echo \"Starting container with command: $docker_cmd\"\n")
            tf.write("eval \"$docker_cmd\"\n")

            # Print container status
            tf.write("\n# Verify container status\n")
            tf.write("docker ps -a --filter \"name=$DOCKER_NAME\"\n")

            # Set execution permissions
            os.chmod(script_path, 0o755)

            # Print script content in dry-run mode
            if dry_run:
                print(f"=== DRY RUN: Script for host {host} ===")
                with open(script_path, "r") as script_file:
                    print(script_file.read())
                print("===")

        if not dry_run:
            try:
                # Build ansible command
                cmd = (
                    f"ansible {shlex.quote(host)} "
                    f"-i {shlex.quote(str(inv_file))} "
                    f"-m script "
                    f"-a {shlex.quote(str(script_path))}"
                )

                print(f"Executing script on host {host}:")
                execute_command(cmd)
            finally:
                # Clean up temporary file
                try:
                    script_path.unlink(missing_ok=True)
                except Exception as e:
                    print(f"Warning: Failed to delete temp file: {e}")
        else:
            # In dry-run mode, just show what command would be executed
            print(f"DRY RUN: Would execute for host {host}:")
            print(f"  ansible {host} -i {inv_file} -m script -a {script_path}")

    print("\nAll hosts processed.")


def prepare_omni_service_in_developer_mode(config_path):
    """In developer mode, preparing to run the omni service."""
    transform_deployment_config(config_path)
    command = f"ansible-playbook -i omni_infer_inventory.yml omni_infer_server.yml --skip-tags 'sync_code,pip_install,run_server,fetch_log'"
    execute_command(command)

def run_omni_service_in_developer_mode():
    """In developer mode, running the omni service."""
    command = f"ansible-playbook -i omni_infer_inventory.yml omni_infer_server.yml --tags run_server"
    execute_command(command)

def synchronize_code():
    """In developer mode, copy the code from the execution machine to the target machine container."""
    command = f"ansible-playbook -i omni_infer_inventory.yml omni_infer_server.yml --tags sync_code"
    execute_command(command)

def install_packages():
    """In developer mode, copy the code and install the packages."""
    command = f"ansible-playbook -i omni_infer_inventory.yml omni_infer_server.yml --tags 'sync_code,pip_install'"
    execute_command(command)

def inspect_configuration(config_path):
    """Inspect detailed configuration information"""
    encoding = detect_file_encoding(config_path)
    with open(config_path, 'r', encoding=encoding) as file:
        data = json.load(file)

    print(json.dumps(
        data,
        indent=4,
        sort_keys=True,
        ensure_ascii=False
    ))

def upgrade_packages():
    """Install the latest wheel package"""
    command = f"ansible-playbook -i omni_infer_inventory.yml omni_infer_server.yml --tags pip_install"
    execute_command(command)

def fetch_logs():
    """Fetch logs"""
    command = f"ansible-playbook -i omni_infer_inventory.yml omni_infer_server.yml --tags fetch_log"
    execute_command(command)

def main():
    # Create main argument parser with description
    parser = argparse.ArgumentParser(description="Omni Inference Service Management")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # START command configuration
    start_parser = subparsers.add_parser("start", help="Start the omni services")
    start_parser.add_argument(
        "config_path",
        nargs='?',
        default=None,
        help='Start in normal mode with config file'
    )
    start_group = start_parser.add_mutually_exclusive_group()
    start_group.add_argument(
        "--normal",
        nargs=1,
        metavar='config_path',
        help="Start in normal mode (default) with config file"
    )
    start_group.add_argument(
        "--prepare_dev",
        nargs=1,
        metavar='config_path',
        help="Start in developer mode with config file: Environmental preparation"
    )
    start_group.add_argument("--run_dev", action="store_true", help="Start in developer mode: Start the service")

    # STOP command configuration
    subparsers.add_parser("stop", help="Stop the omni service")

    # CFG command configuration
    cfg_parser = subparsers.add_parser("cfg", help="Modify configuration")
    cfg_group = cfg_parser.add_mutually_exclusive_group(required=True)
    cfg_group.add_argument("--set", action='store_true', help="Set configuration key-value pairs (e.g., --key value)")
    cfg_parser.add_argument('name', nargs=1, help='Node name (e.g., prefill_0)')
    cfg_parser.add_argument('remaining_args', nargs=argparse.REMAINDER, help='Additional optional parameters')
    cfg_group.add_argument("--delete", action='store_true', help="Delete configuration keys (e.g., --key)")

    # INSPECT command configuration
    inspect_parser = subparsers.add_parser("inspect", help="Inspect Configuration")
    inspect_parser.add_argument('name', nargs=1, help='Node name (e.g., prefill_0)')

    # UPGRADE command configuration
    subparsers.add_parser("upgrade", help="Upgrade packages")

    # FETCH_LOG command configuration
    subparsers.add_parser("fetch_log", help="Fetch logs")

    # ADD_NODE command configuration
    addnode_parser = subparsers.add_parser("add_node", help="Add a node to servering_profiles.yml")
    default_deploy_path = get_default_deploy_path()

    addnode_parser.add_argument(
        "--deploy_path",
        default=str(default_deploy_path),
        help=f"Path to servering_profiles.yml (default: {default_deploy_path})"
    )
    addnode_parser.add_argument("--role", required=True, choices=['P', 'D', 'C'], help="Node role")
    addnode_parser.add_argument("--name", required=True, help="Node name")
    addnode_parser.add_argument("--host_ip", required=True, help="host_ip")
    addnode_parser.add_argument("--user", default="root", help="ansible_user")
    addnode_parser.add_argument("--ssh_common_args", default="-o StrictHostKeyChecking=no -o IdentitiesOnly=yes", help="ssh_common_args")
    addnode_parser.add_argument("--ssh_private_key_file", required=True, help="ssh_private_key_file")
    addnode_parser.add_argument("--master_ip", help="master_ip:The default value is set to host_ip")
    addnode_parser.add_argument("--docker_image_id", required=True, help="docker_image_id")
    addnode_parser.set_defaults(func=add_node)

    # RM_NODE command configuration
    rmnode_parser = subparsers.add_parser("rm_node", help="Remove a node from servering_profiles.yml")
    rmnode_parser.add_argument("--role", required=True, choices=['P', 'D', 'C'], help="Node role")
    rmnode_parser.add_argument(
        "--deploy_path",
        default=str(default_deploy_path),
        help=f"Path to servering_profiles.yml (default: {default_deploy_path})"
    )
    rmnode_parser.add_argument("--name", required=True, help="Node name to remove")
    rmnode_parser.set_defaults(func=rm_node)

    # RUN_DOCKER command configuration
    docker_run_parser = subparsers.add_parser("docker_run", help="Run Docker containers based on inventory")
    docker_run_parser.add_argument(
        "--inventory", "-i",
        default=str(default_deploy_path),
        help="Path to inventory file (default: omni_cli/configs/servering_profiles.yml)"
    )
    docker_run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode - show what would be done without executing"
    )
    docker_run_parser.set_defaults(func=lambda args: run_docker_containers(
        inventory_path=args.inventory,
        dry_run=args.dry_run
    ))
    # SYNC_DEV command configuration
    sync_parser = subparsers.add_parser("sync_dev", help="Developer mode: Synchronize the code")
    sync_parser.add_argument(
        "--deploy_path",
        default=str(default_deploy_path),
        help=f"Path to servering_profiles.yml (default: {default_deploy_path})"
    )
    sync_parser.add_argument(
        "--dry_run", 
        action="store_true", 
        help="Show what would be done without making any changes"
    )
    sync_parser.add_argument("--code_path", required=True, help="code_path")

    sync_parser.set_defaults(func=lambda args:sync_dev(
        inventory_path=args.deploy_path,
        dry_run=args.dry_run,
        code_path=args.code_path
    ))

    # INSTALL_DEV command configuration
    install_parser = subparsers.add_parser("install_dev", help="Developer mode: Install packages")
    install_parser.add_argument(
        "--deploy_path",
        default=str(default_deploy_path),
        help=f"Path to servering_profiles.yml (default: {default_deploy_path})"
    )
    install_parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be done without making any changes"
    )
    install_parser.set_defaults(func=lambda args:install_dev(
        inventory_path=args.deploy_path,
        dry_run=args.dry_run
    ))
    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
        return

    if args.command == "start" and not any([args.normal, args.prepare_dev, args.run_dev]):
        args.normal = True

    # Command processing logic
    if args.command == "start":
        print("Start omni service.")
        if args.config_path is not None:
            print("Normal mode.")
            omni_cli_start(inventory_path=default_deploy_path)
        elif args.normal:
            print("Normal mode.")
            omni_cli_start(inventory_path=default_deploy_path)
        elif args.prepare_dev:
            print("Developer mode: Environmental preparation.")
            prepare_omni_service_in_developer_mode(args.prepare_dev[0])
        elif args.run_dev:
            print("Developer mode: Start the service.")
            run_omni_service_in_developer_mode()
    elif args.command == "stop":
        print("Stop omni service.")
        omni_cli_stop(inventory_path=default_deploy_path)
    elif args.command == "sync_dev":
        print("Synchronize the code.")
        synchronize_code()
    elif args.command == "install_dev":
        print("Install packages.")
        install_packages()
    elif args.command == "cfg":
        node_type, node_name = parse_node_name(args.name[0])
        sections = parse_remaining_args(node_type, node_name, args.set, args.remaining_args, default_deploy_path)
        if args.set:
            print("Set configuration.")
            cfg_set_process(node_type, node_name, args, sections, default_deploy_path)
        elif args.delete:
            print("Delete configuration.")
            cfg_delete_process(node_type, node_name, args, sections, default_deploy_path)
    elif args.command == "inspect":
        print("Inspect configuration.")
        print_node_config(default_deploy_path, args.name[0])
    elif args.command == "upgrade":
        print("Upgrade packages")
        upgrade_packages()
    elif args.command == "fetch_log":
        print("Fetch logs")
        fetch_logs()

if __name__ == "__main__":
    main()