import yaml
import os

def load_yaml(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

def save_yaml(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True)

def add_node(args):
    """Add a node to servering_profiles.yml under all.children.{role}.hosts."""
    # Locate default_profiles.yml and servering_profiles.yml
    base_dir = os.path.dirname(__file__)
    default_path = os.path.join(base_dir, 'configs', 'default_profiles.yml')
    deploy_path = args.deploy_path
    default_profiles = load_yaml(default_path)
    if not default_profiles or 'profiles' not in default_profiles or 'vllm' not in default_profiles['profiles']:
        print("Error: default_profiles.yml not found or invalid.")
        return

    # Load or create deployment file
    deployment = load_yaml(deploy_path)
    if 'all' not in deployment:
        deployment['all'] = {'children': {'P': {'hosts': {}}, 'D': {'hosts': {}}, 'C': {'hosts': {}}}}
    children = deployment['all']['children']
    if args.role not in children:
        children[args.role] = {'hosts': {}}
    hosts = children[args.role]['hosts']

    # C node addition check
    if args.role == 'C' and hosts:
        print("Error: Only one C node allowed.")
        return

    # Duplicate name check
    if args.name in hosts:
        print(f"Error: Node name '{args.name}' already exists in role '{args.role}'.")
        return

    # Count nodes for port offset
    role_count = len(hosts)
    if args.role == 'P':
        offset = 0
    elif args.role == 'D':
        offset = 100
    elif args.role == 'C':
        offset = 200
    else:
        print("Error: role must be P, D, or C.")
        return

    # Prepare node info
    if not args.host_ip:
        args.host_ip = args.ansible_host
    
    # 根据角色设置容器名称前缀
    if args.role == 'P':
        docker_name_prefix = "you_name_omni_infer_prefill"
    elif args.role == 'D':
        docker_name_prefix = "you_name_omni_infer_decode"
    elif args.role == 'C':
        docker_name_prefix = "you_name_omni_infer_proxy"
    
    # 创建节点信息字典
    node = {
        'ansible_user': args.ansible_user,
        'ansible_ssh_common_args': args.ansible_ssh_common_args,
        'ansible_ssh_private_key_file': args.ansible_ssh_private_key_file,
        'ansible_host': args.ansible_host,
        'docker_name': f"{docker_name_prefix}_{args.name}"  # 设置容器名称
    }
    
    # 添加 DOCKER_IMAGE_ID
    if hasattr(args, 'docker_image_id') and args.docker_image_id:
        node['DOCKER_IMAGE_ID'] = args.docker_image_id

    # For P/D, add host_ip
    if args.role in ['P', 'D']:
        node['host_ip'] = args.host_ip

    env = default_profiles['profiles']['vllm']['deepseek']['env'].copy()
    args_dict = default_profiles['profiles']['vllm']['deepseek']['args'].copy()

    # Overwrite env/args with user input if provided
    if hasattr(args, 'env_overwrite') and args.env_overwrite:
        for kv in args.env_overwrite:
            k, v = kv.split('=', 1)
            env[k] = int(v)
    if hasattr(args, 'args_overwrite') and args.args_overwrite:
        for kv in args.args_overwrite:
            k, v = kv.split('=', 1)
            args_dict[k] = int(v)
    
    # Update PORT fields in env
    for k in env:
        if k.endswith('PORT'):
            user_port = None
            if hasattr(args, 'env_overwrite') and args.env_overwrite:
                for kv in args.env_overwrite:
                    k2, v2 = kv.split('=', 1)
                    if k2 == k and v2.isdigit():
                        user_port = int(v2)
            if user_port is not None:
                env[k] = user_port
            else:
                env[k] = env[k] + 16 * role_count + offset
    
    node['env'] = env
    node['args'] = args_dict

    hosts[args.name] = node
    save_yaml(deploy_path, deployment)
    print(f"Node '{args.name}' added successfully to role '{args.role}' with container name '{node['docker_name']}'.")

def rm_node(args):
    """Remove node from servering_profiles.yml and reassign ports for the role."""
    base_dir = os.path.dirname(__file__)
    deploy_path = args.deploy_path
    deployment = load_yaml(deploy_path)
    if 'all' not in deployment or 'children' not in deployment['all'] or args.role not in deployment['all']['children']:
        print(f"Error: Role '{args.role}' not found.")
        return
    hosts = deployment['all']['children'][args.role]['hosts']
    if args.name not in hosts:
        print(f"Error: Node '{args.name}' not found in role '{args.role}'.")
        return

    # Remove node
    del hosts[args.name]

    # Reassign ports for remaining nodes in this role
    # Load default profiles for base ports
    default_path = os.path.join(base_dir, 'configs', 'default_profiles.yml')
    default_profiles = load_yaml(default_path)
    env_template = default_profiles['profiles']['vllm']['deepseek']['env']
    if args.role == 'P':
        offset = 0
    elif args.role == 'D':
        offset = 100
    elif args.role == 'C':
        offset = 200
    else:
        offset = 0
    # Only for P/D, update env PORT fields
    if args.role in ['P', 'D']:
        sorted_items = list(hosts.items())
        for idx, (n, node) in enumerate(sorted_items):
            env = node.get('env', {})
            for k in env:
                if k.endswith('PORT'):
                    env[k] = env_template[k] + 16 * idx + offset
            node['env'] = env
            hosts[n] = node

    save_yaml(deploy_path, deployment)
    print(f"Node '{args.name}' removed from role '{args.role}'. Ports reassigned.")