import yaml
import os
import re
import shlex

def parse_node_name(name):
    """Parse node name, return node type and index"""
    if not name:
        return None, None

    match = re.match(r'^(prefill|decode|proxy)_(\d+)$', name)
    if match:
        node_type = match.group(1)[0].upper()
        if match.group(1) == 'proxy':
            node_type = 'C'
        node_index = f'{node_type.lower()}{int(match.group(2))}'
        return node_type, node_index
    return None, None

def parse_remaining_args_for_set(arg, remaining_args, sections, current_section, i):
    if arg == '--additional-config' or arg == '--extra-args':
        extra_args_list = shlex.split(remaining_args[i+1])
        j = 0
        while j < len(extra_args_list):
            extra_arg = extra_args_list[j]
            if not extra_arg.startswith('--'):
                raise ValueError(f"Invalid key format: '{extra_arg}'. Keys must start with '--'")
            elif (j + 1 < len(extra_args_list) and extra_args_list[j+1].startswith('--')) or \
                j + 1 == len(extra_args_list):
                sections.setdefault(current_section, {}).setdefault(arg[2:], {})[extra_arg[2:]] = ''
                j += 1
            elif j + 1 < len(extra_args_list) and not extra_args_list[j+1].startswith('--'):
                sections.setdefault(current_section, {}).setdefault(arg[2:], {})[extra_arg[2:]] = extra_args_list[j+1]
                j += 2
    elif i + 1 >= len(remaining_args) or remaining_args[i+1].startswith('--'):
        raise ValueError(f"Missing value for key: '{arg}'")
    else:
        sections[current_section][arg[2:]] = remaining_args[i+1]

def parse_remaining_args_for_delete(arg, remaining_args, sections, current_section, i):
    if arg == '--additional-config' or arg == '--extra-args':
        additional_config_list = shlex.split(remaining_args[i+1])
        j = 0
        while j < len(additional_config_list):
            additional_config = additional_config_list[j]
            if not additional_config.startswith('--'):
                raise ValueError(f"Invalid key format: '{additional_config}'. Keys must start with '--'")
            else:
                sections[arg[2:]].append(additional_config[2:])
                j += 1
        i += 1
    else:
        sections[current_section].append(arg[2:])

    return i

def parse_remaining_args(is_set, remaining_args):
    """Resolve the remaining parameters."""
    if is_set:
        sections = {'env': {}, 'arg': {}}
    else:
        sections = {'env': [], 'arg': [], 'extra-args': [], 'additional-config': []}
    current_section = None
    seen_sections = set()

    i = 0
    while i < len(remaining_args):
        arg = remaining_args[i]
        if arg in ['env', 'arg']:
            if arg in seen_sections:
                raise ValueError(f"Duplicate section keyword '{arg}'")
            seen_sections.add(arg)
            current_section = arg
        else:
            if current_section is None:
                raise ValueError(f"Unexpected argument '{arg}' before any section keyword (env/arg)")
            if not arg.startswith('--'):
                raise ValueError(f"Invalid key format: '{arg}'. Keys must start with '--'")
            if is_set:
                parse_remaining_args_for_set(arg, remaining_args, sections, current_section, i)
                i += 1
            else:
                i = parse_remaining_args_for_delete(arg, remaining_args, sections, current_section, i)
        i += 1

    return sections

def get_data_from_yaml(yml_file_path):
    try:
        with open(yml_file_path, 'r') as file:
            data = yaml.safe_load(file)
    except FileNotFoundError:
        print("Error: The %s file is not exist." % yml_file_path)
        return None

    if 'all' not in data or 'children' not in data['all']:
        print("Error: The %s file structure does not meet expectations, missing the 'all.children' section." \
            % yml_file_path)
        return None

    return data

def update_cfg_yml(node_name, node_id, env_dict, arg_dict, yml_file_path):
    data = get_data_from_yaml(yml_file_path)
    if data:
        data['all']['children'][node_name]['hosts'][node_id]['env'].update(env_dict)
        data['all']['children'][node_name]['hosts'][node_id]['args'].update(arg_dict)
        with open(yml_file_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)
    else:
        return

def delete_cfg_yml(node_name, node_id, env_list, arg_list, extra_args_list, additional_config_list, yml_file_path):
    data = get_data_from_yaml(yml_file_path)
    if data:
        vars_dict = data['all']['children'][node_name]['hosts'][node_id]
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

        for key in extra_args_list:
            if key in vars_dict['args']['extra-args']:
                del vars_dict['args']['extra-args'][key]
            else:
                print("Warning: No matching configuration %s found." % key)

        if vars_dict['args']['extra-args'] == {}:
            vars_dict['args']['extra-args'] = ''

        for key in additional_config_list:
            if key in vars_dict['args']['additional-config']:
                del vars_dict['args']['additional-config'][key]
            else:
                print("Warning: No matching configuration %s found." % key)

        if vars_dict['args']['additional-config'] == {}:
            vars_dict['args']['additional-config'] = ''

        with open(yml_file_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)
    else:
        return

def cfg_set_process(node_name, node_id, args, sections, deploy_path):
    if node_name is None or node_id is None:
        print(f"错误：无效的节点名称 '{args.name[0]}'。")
        print("节点名称必须符合以下格式之一：")
        print("  - prefill_<number> (例如: prefill_0, prefiill_1, prefill_11)")
        print("  - decode_<number> (例如: decode_0, decode_1, decode_11)")
        return

    update_cfg_yml(node_name, node_id, sections['env'], sections['arg'], deploy_path)

def cfg_delete_process(node_name, node_id, args, sections, deploy_path):
    if node_name is None or node_id is None:
        print(f"错误：无效的节点名称 '{args.name[0]}'。")
        print("节点名称必须符合以下格式之一：")
        print("  - prefill_<number> (例如: prefill_0, prefiill_1, prefill_11)")
        print("  - decode_<number> (例如: decode_0, decode_1, decode_11)")
        return

    delete_cfg_yml(node_name, node_id, sections['env'], sections['arg'], \
        sections['extra-args'], sections['additional-config'], deploy_path)