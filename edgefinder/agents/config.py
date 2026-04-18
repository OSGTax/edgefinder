"""Runtime kill-switch config for management agents.

The config lives at `.claude/agent-config.json` (gitignored) so operators
can flip an agent off without redeploying. The file is read on every
`is_agent_enabled()` call so changes take effect immediately. If the
file is missing, malformed, or unreadable, agents default to DISABLED —
fail-safe for a live trading system.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AgentConfigError(Exception):
    """Raised when the config file is present but invalid."""


@dataclass(frozen=True)
class AgentConfig:
    """Resolved config for a single agent."""

    enabled: bool
    settings: dict[str, Any]


_CONFIG_PATH_ENV = "EDGEFINDER_AGENT_CONFIG"
_DEFAULT_CONFIG_PATH = Path(".claude/agent-config.json")

_lock = threading.Lock()


def _config_path() -> Path:
    override = os.getenv(_CONFIG_PATH_ENV)
    return Path(override) if override else _DEFAULT_CONFIG_PATH


def _load_raw() -> dict[str, Any] | None:
    path = _config_path()
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise AgentConfigError(f"Invalid JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise AgentConfigError(f"Cannot read {path}: {exc}") from exc


def get_agent_config(agent_name: str) -> AgentConfig:
    """Resolve the effective config for one agent.

    Returns an AgentConfig with enabled=False if the global switch is off,
    the per-agent switch is off, or the config file is missing. Any
    malformed file raises AgentConfigError — callers should catch and
    treat as disabled.
    """
    with _lock:
        raw = _load_raw()
    if raw is None:
        return AgentConfig(enabled=False, settings={})

    if not raw.get("enabled", False):
        return AgentConfig(enabled=False, settings={})

    agents = raw.get("agents", {}) or {}
    agent_raw = agents.get(agent_name, {}) or {}
    return AgentConfig(
        enabled=bool(agent_raw.get("enabled", False)),
        settings=dict(agent_raw),
    )


def is_agent_enabled(agent_name: str) -> bool:
    """Shortcut: True iff the global and per-agent switches are both on.

    Swallows AgentConfigError and returns False — fail-safe default.
    """
    try:
        return get_agent_config(agent_name).enabled
    except AgentConfigError:
        logger.exception("agent-config.json invalid — treating '%s' as disabled", agent_name)
        return False


def reload_agent_config() -> None:
    """No-op kept for API symmetry with cached implementations.

    Each call to get_agent_config() already re-reads the file, so hot
    edits take effect without an explicit reload. This function exists
    so callers can be explicit about their intent.
    """
    return None
