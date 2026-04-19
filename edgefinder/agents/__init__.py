"""EdgeFinder agents — management agents (watchdog, strategist).

This package provides the shared infrastructure used by all management
agents: the runtime kill-switch config and the observation/action journal.
Individual agents (watchdog, strategist) are implemented as Claude Code
skills outside this package; they call into `config` and `journal` to
respect the kill switch and record their activity.
"""

from edgefinder.agents.config import (
    AgentConfigError,
    get_agent_config,
    is_agent_enabled,
    reload_agent_config,
)
from edgefinder.agents.journal import (
    record_action,
    record_observation,
    resolve_observation,
)
from edgefinder.agents.watchdog import (
    ObservationSpec,
    persist_checks,
    run_checks,
    run_once,
)

__all__ = [
    "AgentConfigError",
    "get_agent_config",
    "is_agent_enabled",
    "reload_agent_config",
    "record_action",
    "record_observation",
    "resolve_observation",
    "ObservationSpec",
    "persist_checks",
    "run_checks",
    "run_once",
]
