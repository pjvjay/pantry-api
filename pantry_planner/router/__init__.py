"""Router package — exports the Router protocol and both implementations."""
from .base import Router
from .cascade import CascadeRouter
from .three_phase import ThreePhaseRouter

__all__ = ["Router", "CascadeRouter", "ThreePhaseRouter"]
