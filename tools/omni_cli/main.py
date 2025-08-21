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
from omni_cli.omni_cfg import *

def execute_command(command):
    """Execute the ansible command"""
    process = subprocess.Popen(
        command,
        shell=True
    )

    return_code = process.wait()
    if return_code != 0:
        print(f"Deployment failed with return code {return_code}")
    else:
        print("Deployment succeeded")
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
        else:
            parts.append(f"{flag} {_double_quotes(v)}")
    return " ".join(parts)

def omni_ranktable(inventory):
    cur_dir = os.path.dirname(__file__)
    cmd = "ansible-playbook -i " + inventory + " " + cur_dir + "/ansible/ranktable.yml"
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
    omni_cli.proxy.omni_run_proxy(inventory_path)
    inv_file = Path(inventory_path).expanduser().resolve()
    with open(inv_file, "r", encoding="utf-8") as f:
        inv = yaml.safe_load(f)

    all_hosts = _walk_hosts(inv.get("all", inv))

    selected: List[Tuple[str, Dict[str, Any]]] = []
    for host, hv in all_hosts:
        if host_pattern and host != host_pattern:
            continue
        if role_filter and hv.get("role") != role_filter:
            continue
        selected.append((host, hv))

    if not selected:
        raise RuntimeError("No matching hosts found with given filters.")

    for host, hv in selected:
        env: Dict[str, Any] = hv.get("env", {}) or {}
        args: Dict[str, Any] = hv.get("args", {}) or {}
        docker_name: str = hv.get("docker_name")

        export_block = _build_export_block(env)
        args_line = _build_args_line(args)

        with tempfile.NamedTemporaryFile(
            "w", delete=False,
            dir="./",
            prefix=f"omni_start_{host.replace('.', '_')}_",
            suffix=".sh"
        ) as tf:
            script_path = Path(tf.name)
            code_path = str(env.get("CODE_PATH") or "").strip()
            log_path = str(env.get("LOG_PATH") or "").strip()
            tf.write("#!/usr/bin/env bash\n")
            tf.write("set -euo pipefail\n\n")

            tf.write(f"docker exec -i {shlex.quote(docker_name)} bash -s <<'EOF'\n")
            tf.write("source ~/.bashrc\n\n")

            tf.write(f"if [ ! -d {shlex.quote(log_path)} ]; then\n")
            tf.write(f"  mkdir -p {shlex.quote(log_path)}\n")
            tf.write("fi\n")

            tf.write("# Export environment variables\n")
            tf.write(export_block + "\n\n")
            tf.write(f'echo "{export_block}\n" > {log_path}/omni_cli.log\n\n')

            tf.write("# Exec the command\n")
            tf.write(f"cd {_double_quotes(code_path)}/tools/scripts\n\n")
            tf.write(f"{python_bin} {entry_py} {args_line} >> {log_path}/omni_cli.log 2>&1 &\n")
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

    selected: List[Tuple[str, Dict[str, Any]]] = []
    for host, hv in all_hosts:
        if host_pattern and host != host_pattern:
            continue
        if role_filter and hv.get("role") != role_filter:
            continue
        selected.append((host, hv))

    if not selected:
        raise RuntimeError("No matching hosts found with given filters.")

    for host, hv in selected:
        env: Dict[str, Any] = hv.get("env", {}) or {}
        docker_name: str = hv.get("docker_name")

        with tempfile.NamedTemporaryFile(
            "w", delete=False,
            dir="./",
            prefix=f"omni_stop_{host.replace('.', '_')}_",
            suffix=".sh"
        ) as tf:
            script_path = Path(tf.name)

            tf.write("#!/usr/bin/env bash\n")
            tf.write(f"docker exec -i {shlex.quote(docker_name)} bash -s <<'EOF'\n")
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

def create_default_inventory_if_needed():
    """在当前目录创建默认的 servering_profiles.yml 文件（如果不存在）"""
    current_dir = Path.cwd()
    deploy_path = current_dir / "servering_profiles.yml"
    
    if not deploy_path.exists():
        # 创建默认的 inventory 结构
        default_inventory = {
            "all": {
                "children": {
                    "P": {"hosts": {}},
                    "D": {"hosts": {}},
                    "C": {"hosts": {}}
                }
            }
        }
        
        # 写入文件
        with open(deploy_path, "w") as f:
            yaml.dump(default_inventory, f)
        
        print(f"Created default inventory file at: {deploy_path}")
    
    return deploy_path.resolve()

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
        -v $MODEL_PATH:$MODEL_PATH \\
        -v /tmp/ranktable_save_path:/tmp/ranktable_save_path \\
        -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime"""

    for host, hv in all_hosts:
        print(f"\nProcessing host: {host}")
        
        # Determine the role (P, D, or C) based on group membership
        role = "P"  # Default to P if not specified
        if 'P' in hv.get('groups', []):
            role = "P"
        elif 'D' in hv.get('groups', []):
            role = "D"
        elif 'C' in hv.get('groups', []):
            role = "C"

        # Get environment variables and other host variables
        env = hv.get("env", {})

        # Obtain Key Variables - Directly from the Host Variables
        log_path = env.get("LOG_PATH")
        model_path = env.get("MODEL_PATH")
        docker_image_id = hv.get("DOCKER_IMAGE_ID")  # 与 env 同级

        # Check for required variables
        if not log_path:
            raise ValueError(f"Required environment variable 'LOG_PATH' not defined for host {host}")
        if not model_path:
            raise ValueError(f"Required environment variable 'MODEL_PATH' not defined for host {host}")
        if not docker_image_id:
            raise ValueError(f"Required variable 'DOCKER_IMAGE_ID' not defined for host {host}")
        
        # Create a temporary script file
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sh") as tf:
            script_path = Path(tf.name)
            
            # Write the script content
            tf.write("#!/bin/bash\n")
            tf.write("set -euo pipefail\n\n")
            
            # Export environment variables
            tf.write("# Export environment variables\n")
            for key, value in env.items():
                tf.write(f"export {key}={shlex.quote(str(value))}\n")

            # Add role-specific variables
            tf.write("\n# Role-specific variables\n")
            tf.write(f"export DOCKER_NAME={shlex.quote(hv.get('docker_name', ''))}\n")
            tf.write(f"export DOCKER_IMAGE_ID={shlex.quote(str(docker_image_id))}\n")  # 导出 DOCKER_IMAGE_ID

            # Cleanup existing container
            tf.write("\n# Cleanup existing container\n")
            tf.write("if docker inspect --format='{{.Name}}' \"$DOCKER_NAME\" &>/dev/null; then\n")
            tf.write("    docker stop \"$DOCKER_NAME\"\n")
            tf.write("    docker rm -f \"$DOCKER_NAME\"\n")
            tf.write("fi\n")

            # Build the Docker run command
            tf.write("\n# Build Docker run command\n")
            tf.write(f"docker_cmd={shlex.quote(base_docker_run_cmd)}\n")
            tf.write("docker_cmd+=\" -d --name $DOCKER_NAME $DOCKER_IMAGE_ID\"\n")

            # Execute the command
            tf.write("\n# Execute Docker run command\n")
            tf.write("echo \"Starting container with command: $docker_cmd\"\n")
            tf.write("eval \"$docker_cmd\"\n")
            
            # Print container status
            tf.write("\n# Verify container status\n")
            tf.write("docker ps -a --filter \"name=$DOCKER_NAME\"\n")
            
            # Set execute permissions
            os.chmod(script_path, 0o755)
            
            # Print script content in dry run mode
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
            # In dry run mode, just show the command we would run
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

    # SYNC_DEV command configuration
    subparsers.add_parser("sync_dev", help="Developer mode: Synchronize the code")

    # INSTALL_DEV command configuration
    subparsers.add_parser("install_dev", help="Developer mode: Install packages")

    # CFG command configuration
    cfg_parser = subparsers.add_parser("cfg", help="Modify configuration")
    cfg_group = cfg_parser.add_mutually_exclusive_group(required=True)
    cfg_group.add_argument("--set", metavar="", help="Set configuration key-value pairs (e.g., --key value)")
    cfg_parser.add_argument('name', nargs=1, help='Node name (e.g., prefill_0)')
    cfg_parser.add_argument('remaining_args', nargs=argparse.REMAINDER, help='Additional optional parameters')
    cfg_group.add_argument("--delete", metavar="", help="Delete configuration keys (e.g., --key)")

    # INSPECT command configuration
    inspect_parser = subparsers.add_parser("inspect", help="Inspect Configuration")
    inspect_parser.add_argument('config_path', type=str, help='Path to the configuration file')

    # UPGRADE command configuration
    subparsers.add_parser("upgrade", help="Upgrade packages")

    # FETCH_LOG command configuration
    subparsers.add_parser("fetch_log", help="Fetch logs")

    # ADD_NODE command configuration
    addnode_parser = subparsers.add_parser("add_node", help="Add a node to servering_profiles.yml")
    default_deploy_path = create_default_inventory_if_needed()
    
    addnode_parser.add_argument(
        "--deploy_path", 
        default=str(default_deploy_path),
        help=f"Path to servering_profiles.yml (default: {default_deploy_path})"
    )
    addnode_parser.add_argument("--role", required=True, choices=['P', 'D', 'C'], help="Node role")
    addnode_parser.add_argument("--name", required=True, help="Node name")
    addnode_parser.add_argument("--ansible_host", required=True, help="ansible_host")
    addnode_parser.add_argument("--ansible_user", required=True, help="ansible_user")
    addnode_parser.add_argument("--ansible_ssh_common_args", required=True, help="ansible_ssh_common_args")
    addnode_parser.add_argument("--ansible_ssh_private_key_file", required=True, help="ansible_ssh_private_key_file")
    addnode_parser.add_argument("--host_ip", help="host_ip")
    addnode_parser.add_argument("--docker_image_id", required=True, help="docker_image_id")
    addnode_parser.add_argument("--env_overwrite", nargs='*', help="Overwrite env variables, format KEY=VALUE")
    addnode_parser.add_argument("--args_overwrite", nargs='*', help="Overwrite args variables, format KEY=VALUE")
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
            omni_cli_start()
        elif args.normal:
            print("Normal mode.")
            omni_cli_start()
        elif args.prepare_dev:
            print("Developer mode: Environmental preparation.")
            prepare_omni_service_in_developer_mode(args.prepare_dev[0])
        elif args.run_dev:
            print("Developer mode: Start the service.")
            run_omni_service_in_developer_mode()
    elif args.command == "stop":
        print("Stop omni service.")
        omni_cli_stop()
    elif args.command == "sync_dev":
        print("Synchronize the code.")
        synchronize_code()
    elif args.command == "install_dev":
        print("Install packages.")
        install_packages()
    elif args.command == "cfg":
        node_name, node_id = parse_node_name(args.name[0])
        sections = parse_remaining_args(args.set, args.remaining_args)
        if args.set:
            print("Set configuration.")
            cfg_set_process(node_name, node_id, args, sections)
        elif args.delete:
            print("Delete configuration.")
            cfg_delete_process(node_name, node_id, args, sections)
    elif args.command == "inspect":
        print("Inspect configuration.")
        inspect_configuration(args.config_path)
    elif args.command == "upgrade":
        print("Upgrade packages")
        upgrade_packages()
    elif args.command == "fetch_log":
        print("Fetch logs")
        fetch_logs()


if __name__ == "__main__":
    main()