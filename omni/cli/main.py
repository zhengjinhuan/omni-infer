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

def load_config(config_path):
    """Load and parse YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_ansible_playbook_with_config(config_path):
    """Run Ansible Playbook using the configuration file as inventory"""
    transform_deployment_config(config_path)
    command = f"ansible-playbook -i omni_infer_inventory.yml omni_infer_server.yml"
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(f"Deployment failed: stdout: stdout:{stdout.decode()} stderr:{stderr.decode()}")
    else:
        print(f"Deployment succeeded: {stdout.decode()}")


def check_service_health(config):
    """Perform service health check"""
    try:
        proxy_host = config['deployment']['proxy']['host']
        proxy_port = config['deployment']['proxy']['listen_port']
        model_path = config['services']['model_path']

        url = f"http://{proxy_host}:{proxy_port}/v1/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer YOUR_API_KEY"
        }

        payload = {
            "model": model_path,
            "prompt": "Alice is ",
            "max_tokens": 50,
            "temperature": 0
        }
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            print("Health check passed")
            print(f"Response content: {response.json()}")
            return True
        else:
            print(f"Service abnormal (Status code: {response.status_code})")
            print(f"Error message: {response.text}")
            return False

    except KeyError as e:
        print(f"Configuration error: Missing required configuration item {e}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return False
    except Exception as e:
        print(f"Unknown error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Omni Inference Service Management")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="Deploy inference services")
    serve_parser.add_argument("config", help="Path to configuration file")

    status_parser = subparsers.add_parser("status", help="Check service health")
    status_parser.add_argument("--config", default="omni_infer_deployment.yml", help="Path to configuration file")

    args = parser.parse_args()

    if args.command == "serve":
        print(f"Starting service deployment using configuration file: {args.config}")
        run_ansible_playbook_with_config(args.config)
    elif args.command == "status":
        print(f"Starting service health check using configuration file: {args.config}")
        config = load_config(args.config)
        check_service_health(config)


if __name__ == "__main__":
    main()
