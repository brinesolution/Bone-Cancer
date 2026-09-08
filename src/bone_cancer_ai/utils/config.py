"""Load explicit YAML mappings without executing constructors or hiding duplicate keys."""

from pathlib import Path
from typing import Any

import yaml


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject ambiguous mappings, including nested duplicate keys."""


def _mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode) -> dict:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        try:
            if key in result:
                raise ValueError(f"Duplicate YAML key: {key!r}")
            result[key] = loader.construct_object(value_node, deep=True)
        except TypeError as exc:
            raise ValueError("YAML mapping keys must be hashable") from exc
    return result


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a nonempty mapping; missing files and invalid configuration fail explicitly."""
    path = Path(path)
    try:
        with path.open(encoding="utf-8-sig") as stream:
            config = yaml.load(stream, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(config, dict) or not config:
        raise ValueError(f"Configuration must be a nonempty mapping: {path}")
    if not all(isinstance(key, str) for key in config):
        raise ValueError(f"Configuration root keys must be strings: {path}")
    return config
