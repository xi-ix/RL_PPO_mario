"""Tests for the reduced Mario action space."""

from __future__ import annotations

import unittest

from src.action import (
    ACTION_MAP,
    NOOP,
    NUM_ACTIONS,
    RIGHT,
    RIGHT_JUMP,
    Action,
    ActionAdapter,
    action_name,
    to_nes_action,
)


class FakeEnv:
    """Record the raw action received from ActionAdapter."""

    def __init__(self) -> None:
        self.last_action: int | None = None
        self.step_result = ("observation", 1.0, False, {"x_pos": 42})

    def step(self, action: int):
        self.last_action = action
        return self.step_result


class ActionTest(unittest.TestCase):
    def test_policy_action_ids_are_contiguous(self) -> None:
        self.assertEqual([action.value for action in Action], [0, 1, 2])
        self.assertEqual(NUM_ACTIONS, 3)

    def test_named_constants_match_policy_actions(self) -> None:
        self.assertIs(NOOP, Action.NOOP)
        self.assertIs(RIGHT, Action.RIGHT)
        self.assertIs(RIGHT_JUMP, Action.RIGHT_JUMP)

    def test_actions_map_to_expected_nes_bitmasks(self) -> None:
        self.assertEqual(ACTION_MAP[NOOP], 0b00000000)
        self.assertEqual(ACTION_MAP[RIGHT], 0b10000000)
        self.assertEqual(ACTION_MAP[RIGHT_JUMP], 0b10000001)

    def test_to_nes_action_accepts_enum_and_integer(self) -> None:
        self.assertEqual(to_nes_action(Action.RIGHT), 128)
        self.assertEqual(to_nes_action(2), 129)

    def test_action_names_are_readable(self) -> None:
        self.assertEqual(action_name(0), "NOOP")
        self.assertEqual(action_name(1), "RIGHT")
        self.assertEqual(action_name(2), "RIGHT_JUMP")

    def test_invalid_actions_raise_value_error(self) -> None:
        for action in (-1, 3, 1.5, "1", True):
            with self.subTest(action=action):
                with self.assertRaises(ValueError):
                    to_nes_action(action)  # type: ignore[arg-type]

    def test_adapter_translates_and_forwards_step_result(self) -> None:
        env = FakeEnv()
        adapter = ActionAdapter(env)

        result = adapter.step(RIGHT_JUMP)

        self.assertEqual(env.last_action, 129)
        self.assertIs(result, env.step_result)
        self.assertEqual(adapter.num_actions, 3)


if __name__ == "__main__":
    unittest.main()
