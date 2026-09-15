"""Verify that the Mario emulator can reset, step, and optionally render."""

from __future__ import annotations

import argparse
from importlib.metadata import version

import gym_super_mario_bros


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open a game window while the random policy runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3")
    env.seed(0)

    observation = env.reset()
    total_reward = 0.0
    resets = 0
    max_x_pos = 0
    flag_get = False

    try:
        for _ in range(args.steps):
            observation, reward, done, info = env.step(env.action_space.sample())
            total_reward += float(reward)
            max_x_pos = max(max_x_pos, int(info.get("x_pos", 0)))
            flag_get = flag_get or bool(info.get("flag_get", False))

            if args.render:
                env.render()

            if done:
                observation = env.reset()
                resets += 1
    finally:
        env.close()

    print(f"gym-super-mario-bros={version('gym-super-mario-bros')}")
    print(f"nes-py={version('nes-py')}")
    print(f"gym={version('gym')}")
    print(f"numpy={version('numpy')}")
    print(f"observation={observation.shape} dtype={observation.dtype}")
    print(f"action_space={env.action_space}")
    print(
        f"steps={args.steps} resets={resets} reward={total_reward:.1f} "
        f"max_x_pos={max_x_pos} flag_get={flag_get}"
    )


if __name__ == "__main__":
    main()

