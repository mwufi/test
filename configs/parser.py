import yaml
from box import Box


def load_yaml(yaml_path):
    # Load yaml file
    try:
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    except Exception as e:
        print('Error reading the config file')


def load_config(yaml_path):
    config = load_yaml(yaml_path)
    # Convert to object
    return Box(config)
