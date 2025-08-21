import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import tempfile
import os
import shlex

def calculate_api_port(host_vars, global_vars, group_key):
    node_rank = host_vars.get('node_rank', 0)
    kv_rank = host_vars.get('kv_rank', 0)
    if group_key == 'C':
        return global_vars['proxy_port'] + node_rank
    if group_key == 'P':
        port_offset = global_vars['port_offset'][group_key]
        return global_vars['base_api_port'] + port_offset + kv_rank
    if group_key == 'D':
        port_offset = global_vars['port_offset'][group_key]
        return global_vars['base_api_port'] + port_offset + node_rank


def register_values(inventory):
    global_vars = inventory['all']['vars']
    
    # calculate PREFILL_API_SERVER_LIST
    prefill_api_server_list = []
    
    for host, vars in inventory['all']['children']['P']['hosts'].items():
        ansible_host_val = vars.get('ansible_host', '')
        host_ip_val = vars.get('ansible_host', '')
        api_port_val = calculate_api_port(vars, global_vars, 'P')
        
        if ansible_host_val and host_ip_val and ansible_host_val == host_ip_val:
            entry = f"{ansible_host_val}:{api_port_val}"
            if entry not in prefill_api_server_list:
                prefill_api_server_list.append(entry)

    prefill_api_server_list_result = ','.join(prefill_api_server_list)

    # calculate DECODE_API_SERVER_LIST
    decode_api_server_list = []
    
    for host, vars in inventory['all']['children']['D']['hosts'].items():
        ip = vars.get('ansible_host', '')
        api_port_val = calculate_api_port(vars, global_vars, 'D')
        num = vars.get('ascend_rt_visible_devices', '').count(',') + 1 
        
        if ip:
            entry = f"{ip}:{api_port_val}@{num}"
            if entry not in decode_api_server_list:
                decode_api_server_list.append(entry)

    decode_api_server_list_result = ','.join(decode_api_server_list)

    return {
        'PREFILL_API_SERVER_LIST': prefill_api_server_list_result,
        'DECODE_API_SERVER_LIST': decode_api_server_list_result
    }

def process_results(results, inventory):

    prefill_result = results['PREFILL_API_SERVER_LIST']
    decode_api_servers = results['DECODE_API_SERVER_LIST']

    prefill_result = ' '.join(prefill_result.split())

    decode_result = ""

    decode_array = decode_api_servers.split(',')

    for var in decode_array:
        address = var.split('@')[0]
        num = int(var.split('@')[1])

        ip = address.split(':')[0]
        port = int(address.split(':')[1])

        for i in range(num):
            if not decode_result:
                decode_result = f"{ip}:{port}"
            else:
                decode_result += f",{ip}:{port}"
            port += 1

    global_vars = inventory['all']['vars']
    for host, vars in inventory['all']['children']['C']['hosts'].items():
        print(host)
        host_ip_val = vars.get('ansible_host', '')
        api_port_val = calculate_api_port(vars, global_vars, 'C')
        print(inventory['all']['children']['C']['hosts'])
        print(host_ip_val)
        print(api_port_val)

    with tempfile.NamedTemporaryFile(
        "w", delete=False,
        dir="./",
        prefix=f"omni_proxy_",
        suffix=".sh") as tf:
        script_path = Path(tf.name)
        tf.write("#!/usr/bin/env bash\n")
        docker_name = "wcd_omni_infer_proxy_c0" 
        tf.write(f"docker exec -i {shlex.quote(docker_name)} bash -s <<'EOF'\n")
        tf.write(f"cd /workspace/omniinfer/tools/scripts; bash global_proxy.sh \\\n\
          --listen-port {api_port_val} \\\n\
          --prefill-servers-list {prefill_result} \\\n\
          --decode-servers-list {decode_result} \\\n\
          --log-file /tmp/nginx/nginx_error.log \\\n\
          --log-level notice \\\n\
          --core-num 4 \\\n\
          --start-core-index 16 \\\n\
          --prefill-lb-sdk $4 \\\n\
          --decode-lb-sdk $5\n\n")
        tf.write("EOF\n")

    os.chmod(script_path, 0o755)

    cmd = (
        f"ansible {shlex.quote(host)} "
        f"-i {shlex.quote(str(inv_file))} "
        f"-m script "
        f"-a {shlex.quote(str(script_path))}"
    )

    try:
        os.system(cmd)
    finally:
        try:
            script_path.unlink(missing_ok=True)
        except Exception:
            pass

inv_file = Path("./omni_infer_inventory_used_for_2P1D.yml").expanduser().resolve()
inv = None
with open(inv_file, "r", encoding="utf-8") as f:
    inv = yaml.safe_load(f)

result = register_values(inv)
print(result)


process_results(result, inv)
