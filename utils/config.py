import os
import socket
import sys
from copy import deepcopy
from typing import Dict, Any, Iterable

import yaml
import torch


def load_yaml(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_yaml(data: Dict[str, Any], path: str) -> None:
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result


def cli_keys_from_argv(argv: Iterable[str]) -> set:
    keys = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        key = token[2:]
        if "=" in key:
            key = key.split("=", 1)[0]
        key = key.replace("-", "_")
        keys.add(key)
    return keys


def resolve_dataset_config(cfg: Dict[str, Any]) -> str | None:
    paths = cfg.get("dataset_config_paths", {})
    if not isinstance(paths, dict) or not paths:
        return None
    if os.name == "nt":
        return paths.get("win")
    hostname = socket.gethostname()
    resolved = paths.get(hostname)
    if resolved:
        return resolved
    return paths.get("wsl")

def apply_config_to_args(args, cfg: Dict[str, Any], cli_keys: set) -> None:
    for key, value in cfg.items():
        if key in {"dataset_config_paths"}:
            continue
        if key in cli_keys:
            continue
        if key == "device" and value in {None, "auto"}:
            value = "cuda" if torch.cuda.is_available() else "cpu"
        setattr(args, key, value)
