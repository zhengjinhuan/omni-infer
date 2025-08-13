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
from omni.cli.config_transform import transform_deployment_config
from omni.cli.config_transform import detect_file_encoding

def execute_command(command):
    """Execute the ansible command"""
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=None,
        stderr=None
    )

    return_code = process.wait()
    if return_code != 0:
        print(f"Deployment failed with return code {return_code}")
    else:
        print("Deployment succeeded")

def start_omni_service_in_normal_mode(config_path):
    """Run the omni service in normal mode."""
    transform_deployment_config(config_path)
    command = f"ansible-playbook -i omni_infer_inventory.yml omni_infer_server.yml --skip-tags fetch_log"
    execute_command(command)

def prepare_omni_service_in_developer_mode(config_path):
    """In developer mode, preparing to run the omni service."""
    transform_deployment_config(config_path)
    command = f"ansible-playbook -i omni_infer_inventory.yml omni_infer_server.yml --skip-tags 'run_server,fetch_log'"
    execute_command(command)

def run_omni_service_in_developer_mode():
    """In developer mode, running the omni service."""
    command = f"ansible-playbook -i omni_infer_inventory.yml omni_infer_server.yml --tags run_server"
    execute_command(command)

def stop_omni_service():
    """Stop the omni service."""
    command = f"ansible-playbook -i omni_infer_inventory.yml omni_infer_server.yml --tags stop_server"
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

def main():
    # Create main argument parser with description
    parser = argparse.ArgumentParser(description="Omni Inference Service Management")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # START command configuration
    start_parser = subparsers.add_parser("start", help="Start the omni services")
    start_group = start_parser.add_mutually_exclusive_group()
    start_group.add_argument("--normal", action="store_true", help="Start in normal mode (default)")
    start_group.add_argument(
        "--prepare_dev",
        action="store_true",
        help="Start in developer mode: Environmental preparation"
    )
    start_group.add_argument("--run_dev", action="store_true", help="Start in developer mode: Start the service")

    # STOP command configuration
    subparsers.add_parser("stop", help="Stop the omni service")

    # SYNC_DEV command configuration
    subparsers.add_parser("install_dev", help="Developer mode: Install packages")

    # CFG command configuration
    cfg_parser = subparsers.add_parser("cfg", help="Modify configuration")
    cfg_group = cfg_parser.add_mutually_exclusive_group()
    cfg_group.add_argument("--set", action="store_true", help="Set configuration")
    cfg_group.add_argument("--delete", action="store_true", help="Delete configuration")

    # INSPECT command configuration
    subparsers.add_parser("inspect", help="Inspect Configuration")

    # UPGRADE command configuration
    subparsers.add_parser("upgrade", help="Upgrade packages")

    args = parser.parse_args()
    if args.command == "start" and not any([args.normal, args.prepare_dev, args.run_dev]):
        args.normal = True

    # Command processing logic
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = f"{current_dir}/omni_infer_deployment.json"
    if args.command == "start":
        print("Start omni service.")
        if args.normal:
            print("Normal Mode.")
            start_omni_service_in_normal_mode(config_path)
        elif args.prepare_dev:
            print("Developer mode: Environmental preparation.")
            prepare_omni_service_in_developer_mode(config_path)
        elif args.run_dev:
            print("Developer mode: Start the service.")
            run_omni_service_in_developer_mode()
    elif args.command == "stop":
        print("Stop omni service.")
        stop_omni_service()
    elif args.command == "sync_dev":
        print("Synchronize the code.")
        synchronize_code()
    elif args.command == "install_dev":
        print("Install packages.")
        install_packages()
    elif args.command == "cfg":
        if args.set:
            print("Set configuration.")
            set_configuration(config_path)
        elif args.delete:
            print("Delete configuration.")
            del_configuration(config_path)
    elif args.command == "inspect":
        print("Inspect configuration.")
        inspect_configuration(config_path=)
    elif args.command == "upgrade":
        print("Upgrade packages")
        upgrade_packages()


if __name__ == "__main__":
    main()
