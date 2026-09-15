"""Map the policy's small action space to NES controller bitmasks."""

from __future__ import annotations

from enum import IntEnum
from operator import index
from typing import Any


class Action(IntEnum):
    """Actions exposed to the policy network."""

    NOOP = 0
    RIGHT = 1
    RIGHT_JUMP = 2


# Keep these aliases convenient for rollout and evaluation code.
NOOP = Action.NOOP
RIGHT = Action.RIGHT
RIGHT_JUMP = Action.RIGHT_JUMP

# nes-py represents the controller as an 8-bit mask. A is the jump button.
ACTION_MAP = {
    Action.NOOP: 0b00000000,
    Action.RIGHT: 0b10000000,
    Action.RIGHT_JUMP: 0b10000001,
}

ACTION_NAMES = {
    Action.NOOP: "NOOP",
    Action.RIGHT: "RIGHT",
    Action.RIGHT_JUMP: "RIGHT_JUMP",
}

NUM_ACTIONS = len(Action)


def _as_action(action: int | Action) -> Action:
    """Return a validated Action without accepting floats or strings."""

    if isinstance(action, bool):
        raise ValueError("action must be an integer from 0 to 2, not bool")

    try:
        action_id = index(action)
        return Action(action_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid action {action!r}; expected an integer from 0 to 2") from exc


def to_nes_action(action: int | Action) -> int:
    """Convert a policy action into the bitmask expected by nes-py."""

    return ACTION_MAP[_as_action(action)]


def action_name(action: int | Action) -> str:
    """Return a readable name for logging and debugging."""

    return ACTION_NAMES[_as_action(action)]


class ActionAdapter:
    """Send policy actions to an environment using NES controller values."""

    num_actions = NUM_ACTIONS

    def __init__(self, env: Any) -> None:
        self.env = env

    def step(self, action: int | Action) -> Any:
        """Translate one policy action and forward it to ``env.step``."""

        return self.env.step(to_nes_action(action))
