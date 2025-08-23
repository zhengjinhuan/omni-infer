import yaml
import os
import re
import shlex

def get_data_from_yaml(yml_file_path):
    try:
        with open(yml_file_path, 'r') as file:
            data = yaml.safe_load(file)
    except FileNotFoundError:
        print("Error: The %s file is not exist." % yml_file_path)
        return None

    return data

def parse_node_name(name):
    """Parse node name, return node type and index"""
    if not name:
        return None, None

    match = re.match(r'^([pdc](?:0|[1-9]\d*)?|all)$', name)
    if match:
        full_match = match.group(0)
        if full_match == 'all':
            return full_match, None
        else:
            node_type = full_match[0].upper()
            return node_type, full_match
    return None, None

def parse_remaining_args_for_set(arg, remaining_args, sections, i):
    if remaining_args[i+1] in ['--additional-config', '--extra-args', '--kv-transfer-config']:
        if i + 2 >= len(remaining_args):
            raise ValueError(f"Missing value for key: '{remaining_args[i+1]}'")
        extra_args_list = shlex.split(remaining_args[i+2])
        j = 0
        while j < len(extra_args_list):
            extra_arg = extra_args_list[j]
            if not extra_arg.startswith('--'):
                raise ValueError(f"Invalid key format: '{extra_arg}'. Keys must start with '--'")
            elif (j + 1 < len(extra_args_list) and extra_args_list[j+1].startswith('--')) or \
                j + 1 == len(extra_args_list):
                sections.setdefault(arg, {}).setdefault(arg[2:], {})[extra_arg[2:]] = ''
                j += 1
            elif j + 1 < len(extra_args_list) and not extra_args_list[j+1].startswith('--'):
                sections.setdefault(arg, {}).setdefault(arg[2:], {})[extra_arg[2:]] = extra_args_list[j+1]
                j += 2
    elif i + 2 >= len(remaining_args) or remaining_args[i+2].startswith('--'):
        raise ValueError(f"Missing value for key: '{remaining_args[i+1]}'")
    else:
        sections[arg][remaining_args[i+1][2:]] = remaining_args[i+2]

def parse_remaining_args_for_delete(arg, remaining_args, sections, i):
    if remaining_args[i+1] in ['--additional-config', '--extra-args', '--kv-transfer-config']:
        if i + 2 >= len(remaining_args):
            raise ValueError(f"Missing value for key: '{remaining_args[i+1]}'")
        additional_config_list = shlex.split(remaining_args[i+2])
        j = 0
        while j < len(additional_config_list):
            additional_config = additional_config_list[j]
            if not additional_config.startswith('--'):
                raise ValueError(f"Invalid key format: '{additional_config}'. Keys must start with '--'")
            else:
                sections[remaining_args[i+1][2:]].append(additional_config[2:])
                j += 1
        i += 1
    else:
        sections[arg].append(remaining_args[i+1][2:])

    return i

def update_container_name(node_type, node_name, container_name_prefix, yml_file_path):
    data = get_data_from_yaml(yml_file_path)
    if data:
        if node_type == 'all':
            for n_type in data['all']['children']:
                for n_name in data['all']['children'][n_type]['hosts']:
                    data['all']['children'][n_type]['hosts'][n_name]['container_name'] = \
                        f'{container_name_prefix}_{n_name}'
        elif node_type == 'P' or node_type == 'D' or node_type == 'C':
            for n_name in data['all']['children'][node_type]['hosts']:
                data['all']['children'][node_type]['hosts'][n_name]['container_name'] = \
                    f'{container_name_prefix}_{n_name}'
        else:
            data['all']['children'][node_type]['hosts'][node_name]['container_name'] = \
                    f'{container_name_prefix}_{node_name}'
        
        with open(yml_file_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)
    else:
        raise ValueError(f"There is no data in {yml_file_path}")
        return

def parse_remaining_args(node_type, node_name, is_set, remaining_args, yml_file_path):
    """Resolve the remaining parameters."""
    if is_set:
        sections = {'env': {}, 'args': {}, 'DOCKER_IMAGE_ID': '', 'ascend_rt_visible_devices': '', 'container_name': ''}
    else:
        sections = {'env': [], 'args': [], 'DOCKER_IMAGE_ID': '', 'ascend_rt_visible_devices': '', \
            'container_name': '', 'extra-args': [], 'additional-config': [], 'kv-transfer-config': []}
    seen_sections = set()

    i = 0
    while i < len(remaining_args):
        arg = remaining_args[i]
        if arg in list(sections.keys())[:2]:
            if arg in seen_sections:
                raise ValueError(f"Duplicate section keyword '{arg}'")
            seen_sections.add(arg)
            if i + 1 >= len(remaining_args) or not remaining_args[i+1].startswith('--'):
                raise ValueError(f"Missing value for key: '{arg}'")
            elif remaining_args[i+1][2:] not in list(sections.keys())[2:]:
                if is_set:
                    parse_remaining_args_for_set(arg, remaining_args, sections, i)
                    i += 3
                else:
                    i = parse_remaining_args_for_delete(arg, remaining_args, sections, i)
                    i += 2
            else:
                raise ValueError(f"Unexpected argument '{remaining_args[i+1]}' after (env/arg)")
        elif arg[2:] in list(sections.keys())[2:6]:
            if arg in seen_sections:
                raise ValueError(f"Duplicate section keyword '{arg}'")
            seen_sections.add(arg)
            if i + 1 >= len(remaining_args) or remaining_args[i+1].startswith('--'):
                raise ValueError(f"Missing value for key: '{arg}'")
            if is_set:
                sections[arg[2:]] = remaining_args[i+1]
                i += 2
            else:
                sections[arg[2:]] = True
                i += 1
        elif arg[2:] == 'container_name_prefix':
            print("请注意你正在通过容器名前缀设置容器名。")
            if i + 1 >= len(remaining_args) or remaining_args[i+1].startswith('--'):
                raise ValueError(f"Missing value for key: '{arg}'")
            if is_set:
                update_container_name(node_type, node_name, remaining_args[i+1], yml_file_path)
                i += 1
            else:
                raise ValueError(f"Unexpected key {arg}")
        else:
            raise ValueError(f"Unexpected key {arg}")

    return sections

def first_check_model_path(default_cfg_path, sections, data, node_type, node_name):
    if node_type == 'C':
        return True
    model_path = data['profiles']['vllm']['deepseek'][node_type]['env']['MODEL_PATH']
    if model_path == '' or model_path is None:
        if 'MODEL_PATH' in sections['env']:
            return True
        else:
            raise ValueError(f"Error: The model_path is not configured in {node_name}. Please set the configuration.")

    data['profiles']['vllm']['deepseek'][node_type]['env']['MODEL_PATH'] = ""
    with open(default_cfg_path , 'w') as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)
    return True

def second_check_model_path(sections, data, node_type, node_name):
    if node_type == 'C':
        return True
    model_path = data['all']['children'][node_type]['hosts'][node_name]['env']['MODEL_PATH']
    if model_path == '' or model_path is None:
        if 'env' in sections and 'MODEL_PATH' in sections['env']:
            return True
        else:
            raise ValueError(f"Error: The model_path is not configured in {node_name}. Please set the configuration.")

    return True

def update_cfg_yml(node_type, node_name, sections, yml_file_path):
    data = get_data_from_yaml(yml_file_path)
    filtered_sections = {k: v for k, v in sections.items() if v not in ['', None, {}]}
    if data:
        if node_type == 'all':
            for n_type in data['all']['children']:
                for n_name in data['all']['children'][n_type]['hosts']:
                    data['all']['children'][n_type]['hosts'][n_name].update(filtered_sections)
                    if second_check_model_path(filtered_sections, data, n_type, n_name) is not True:
                        return
            print("你已修改所有节点的配置")
        elif node_type == 'P' or node_type == 'D' or node_type == 'C' and node_name is None:
            for n_name in data['all']['children'][node_type]['hosts']:
                data['all']['children'][node_type]['hosts'][n_name].update(filtered_sections)
                if second_check_model_path(filtered_sections, data, node_type, n_name) is not True:
                    return
            print("你已修改 %s 组所有节点的配置" % node_type)
        else:
            data['all']['children'][node_type]['hosts'][node_name].update(filtered_sections)
            if second_check_model_path(filtered_sections, data, node_type, node_name) is not True:
                return
            print("你已修改 %s 节点的配置" % node_name)

        with open(yml_file_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)
    else:
        print(f"Error: There is no data in {yml_file_path}.")
        return
    
def delete_cfg_yml_for_node(data, node_type, node_name, env_list, arg_list, DOCKER_IMAGE_ID, \
    ascend_rt_visible_devices, EXECUTOR_CODE_PATH, container_name, extra_args_list, additional_config_list, \
    kv_transfer_config_list):
    vars_dict = data['all']['children'][node_type]['hosts'][node_name]
    for key in env_list:
        if key in vars_dict['env']:
            del vars_dict['env'][key]
        else:
            print("Warning: No matching configuration %s found." % key)

    for key in arg_list:
        if key in vars_dict['args']:
            del vars_dict['args'][key]
        else:
            print("Warning: No matching configuration %s found." % key)

    if DOCKER_IMAGE_ID and 'DOCKER_IMAGE_ID' in vars_dict:
        del vars_dict['DOCKER_IMAGE_ID']

    if ascend_rt_visible_devices and 'ascend_rt_visible_devices' in vars_dict:
        del vars_dict['ascend_rt_visible_devices']

    if EXECUTOR_CODE_PATH and 'EXECUTOR_CODE_PATH' in vars_dict:
        del vars_dict['EXECUTOR_CODE_PATH']

    if container_name and 'container_name' in vars_dict:
        del vars_dict['container_name']

    for key in extra_args_list:
        if key in vars_dict['args']['extra-args']:
            del vars_dict['args']['extra-args'][key]
        else:
            print("Warning: No matching configuration %s found." % key)

    if 'extra-args' in vars_dict['args'] and vars_dict['args']['extra-args'] == {}:
        vars_dict['args']['extra-args'] = ''

    for key in additional_config_list:
        if key in vars_dict['args']['additional-config']:
            del vars_dict['args']['additional-config'][key]
        else:
            print("Warning: No matching configuration %s found." % key)

    if 'additional-config' in vars_dict['args'] and vars_dict['args']['additional-config'] == {}:
        vars_dict['args']['additional-config'] = ''

    for key in kv_transfer_config_list:
        if key in vars_dict['args']['kv-transfer-config']:
            del vars_dict['args']['kv-transfer-config'][key]
        else:
            print("Warning: No matching configuration %s found." % key)

    if 'kv-transfer-config' in vars_dict['args'] and vars_dict['args']['kv-transfer-config'] == {}:
        vars_dict['args']['kv-transfer-config'] = ''

def delete_cfg_yml(node_type, node_name, sections, yml_file_path):
    env_list = sections['env']
    arg_list = sections['arg']
    DOCKER_IMAGE_ID = sections['DOCKER_IMAGE_ID']
    ascend_rt_visible_devices = sections['ascend_rt_visible_devices']
    EXECUTOR_CODE_PATH = sections['EXECUTOR_CODE_PATH']
    container_name = sections['container_name']
    extra_args_list = sections['extra-args']
    additional_config_list = sections['additional-config']
    kv_transfer_config_list = sections['kv-transfer-config']
    data = get_data_from_yaml(yml_file_path)
    if data:
        if node_type == 'all':
            for n_type in data['all']['children']:
                for n_name in data['all']['children'][n_type]['hosts']:
                    print("你已删除所有节点的配置")
                    delete_cfg_yml_for_node(data, n_type, n_name, env_list, arg_list, DOCKER_IMAGE_ID, \
                        ascend_rt_visible_devices, EXECUTOR_CODE_PATH, container_name, extra_args_list, \
                        additional_config_list, kv_transfer_config_list)
        elif node_type == 'P' or node_type == 'D' or node_type == 'C':
            for n_name in data['all']['children'][node_type]['hosts']:
                print("你已删除 %s 组所有节点的配置" % node_type)
                delete_cfg_yml_for_node(data, node_type, n_name, env_list, arg_list, DOCKER_IMAGE_ID, \
                    ascend_rt_visible_devices, EXECUTOR_CODE_PATH, container_name, extra_args_list, \
                    additional_config_list, kv_transfer_config_list)
        else:
            print("你已删除 %s 节点的配置" % n_name)
            delete_cfg_yml_for_node(data, node_type, node_name, env_list, arg_list, DOCKER_IMAGE_ID, \
                ascend_rt_visible_devices, EXECUTOR_CODE_PATH, container_name, extra_args_list, \
                additional_config_list, kv_transfer_config_list)

        with open(yml_file_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)
    else:
        print(f"Error: There is no data in {yml_file_path}.")
        return

def cfg_set_process(node_type, node_name, args, sections, deploy_path):
    if node_type is None and node_name is None:
        print(f"错误：无效的节点名称 '{args.name[0]}'。")
        print("节点名称必须符合以下格式之一：")
        print("  - prefill_<number> (例如: prefill_0, prefiill_1, prefill_11)")
        print("  - decode_<number> (例如: decode_0, decode_1, decode_11)")
        return
    
    default_cfg_path = f'{os.path.dirname(__file__)}/configs/default_profiles.yml'
    default_cfg = get_data_from_yaml(default_cfg_path)
    data =  get_data_from_yaml(deploy_path)
    if data:
        if node_type == 'all':
            for n_type in default_cfg['profiles']['vllm']['deepseek']:
                for n_name in data['all']['children'][n_type]['hosts']:
                    if first_check_model_path(default_cfg_path, sections, default_cfg, n_type, n_name) is True:
                        sections_bak = default_cfg['profiles']['vllm']['deepseek'][n_type]
                        data['all']['children'][n_type]['hosts'][n_name].update(sections_bak)
        elif node_type == 'P' or node_type == 'D' or node_type == 'C' and node_name is None:
            for n_name in data['all']['children'][node_type]['hosts']:
                if first_check_model_path(default_cfg_path, sections, default_cfg, node_type, n_name) is True:
                    sections_bak = default_cfg['profiles']['vllm']['deepseek'][node_type]
                    data['all']['children'][node_type]['hosts'][n_name].update(sections_bak)
        else:
            if first_check_model_path(default_cfg_path, sections, default_cfg, node_type, node_name) is True:
                sections_bak = default_cfg['profiles']['vllm']['deepseek'][node_type]
                data['all']['children'][node_type]['hosts'][node_name].update(sections_bak)
        with open(deploy_path , 'w') as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)
    else:
        raise ValueError(f"There is no data in {deploy_path}")
        return

    update_cfg_yml(node_type, node_name, sections, deploy_path)

def cfg_delete_process(node_type, node_name, args, sections, deploy_path):
    if node_type is None and node_name is None:
        print(f"错误：无效的节点名称 '{args.name[0]}'。")
        print("节点名称必须符合以下格式之一：")
        print("  - prefill_<number> (例如: prefill_0, prefiill_1, prefill_11)")
        print("  - decode_<number> (例如: decode_0, decode_1, decode_11)")
        return

    delete_cfg_yml(node_type, node_name, sections, deploy_path)