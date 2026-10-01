from pathlib import Path

import yaml


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text())


def load_config(path="configs/default.yaml"):
    config = load_yaml(path)
    config["_path"] = str(path)
    return config


def load_distortions(path="configs/distortions.yaml"):
    return load_yaml(path)


def get(config, dotted, default=None):
    node = config
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node
