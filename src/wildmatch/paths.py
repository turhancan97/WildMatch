"""Path profiles (``conf/paths/<profile>.yaml``) for code that does not run through Hydra.

Hydra runs choose a profile with ``paths=<name>``. Everything else (scripts, the registry,
the paper tooling) calls :func:`load_paths`, which picks the profile in this order:

1. the ``profile`` argument;
2. the ``WILDMATCH_PATHS`` environment variable;
3. ``paths: <name>`` in ``wildmatch.local.yaml`` in the working directory (gitignored; a
   checkout on the GMUM cluster holds ``paths: gmum`` there, so no variable is needed);
4. ``default``.

This module depends only on OmegaConf, so tools that run in other environments (for
example the SAM 3 segmentation script) can import it.
"""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf

LOCAL_SETTINGS = "wildmatch.local.yaml"
DEFAULT_PROFILE = "default"


def _conf_dir():
    return resources.files("wildmatch") / "conf"


def available_profiles() -> list[str]:
    return sorted(entry.name[:-5] for entry in (_conf_dir() / "paths").iterdir() if entry.name.endswith(".yaml"))


def active_profile(profile: Optional[str] = None, cwd: Optional[Path] = None) -> str:
    """Resolve the profile name (see the module docstring for the order)."""
    if profile:
        return profile
    env = os.environ.get("WILDMATCH_PATHS")
    if env:
        return env
    local = Path(cwd or Path.cwd()) / LOCAL_SETTINGS
    if local.is_file():
        settings = OmegaConf.load(local)
        name = settings.get("paths") if isinstance(settings, DictConfig) else None
        if name:
            return str(name)
    return DEFAULT_PROFILE


def load_paths(profile: Optional[str] = None, cwd: Optional[Path] = None) -> DictConfig:
    """The resolved path profile as a DictConfig (``data_root``, ``cache_root``, ...)."""
    name = active_profile(profile, cwd)
    source = _conf_dir() / "paths" / f"{name}.yaml"
    if not source.is_file():
        raise ValueError(f"unknown paths profile {name!r}; available: {', '.join(available_profiles())}")
    cfg = OmegaConf.create(source.read_text(encoding="utf-8"))
    OmegaConf.resolve(cfg)
    return cfg


def path(key: str, profile: Optional[str] = None) -> Optional[Path]:
    """One dotted entry of the active profile as a Path (``None`` when unset)."""
    value = OmegaConf.select(load_paths(profile), key)
    return None if value in (None, "") else Path(str(value))
