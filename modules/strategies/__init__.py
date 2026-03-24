"""
EdgeFinder Strategy Plugin System
==================================
Drop-in strategy plugins that inherit from BaseStrategy.
"""

from modules.strategies.base import BaseStrategy, StrategyRegistry, Signal

__all__ = ["BaseStrategy", "StrategyRegistry", "Signal"]
