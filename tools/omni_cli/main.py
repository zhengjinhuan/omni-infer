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
import yaml
import requests
import json
import os
from omni_cli.config_transform import transform_deployment_config
from omni_cli.config_transform import detect_file_encoding
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import tempfile
import shlex
import omni_cli.proxy

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

def set_configuration(config_path):
    """Set configuration."""
    transform_deployment_config(config_path)

def del_configuration(config_path):
    """Delete configuration"""
    transform_deployment_config(config_path)

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
    cfg_group = cfg_parser.add_mutually_exclusive_group()
    cfg_group.add_argument("--set", nargs=1, metavar='config_path', help="Set configuration")
    cfg_group.add_argument("--delete", nargs=1, metavar='config_path', help="Delete configuration")

    # INSPECT command configuration
    inspect_parser = subparsers.add_parser("inspect", help="Inspect Configuration")
    inspect_parser.add_argument('config_path', type=str, help='Path to the configuration file')

    # UPGRADE command configuration
    subparsers.add_parser("upgrade", help="Upgrade packages")

    # FETCH_LOG command configuration
    subparsers.add_parser("fetch_log", help="Fetch logs")

    args = parser.parse_args()
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
        if args.set:
            print("Set configuration.")
            set_configuration(args.set[0])
        elif args.delete:
            print("Delete configuration.")
            del_configuration(args.delete[0])
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
