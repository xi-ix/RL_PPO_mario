# RL_PPO_mario

从基础组件开始实现 PPO，并训练智能体操作 NES 版 Super Mario Bros。

## 环境

本项目在 Apple Silicon Mac 上使用 Miniforge 和独立的 `mario-ppo`
环境。`gym-super-mario-bros` 只作为游戏控制接口，强化学习算法将由项目自行实现。

首次创建环境：

```bash
/Users/wangzhe/miniforge3/bin/conda env create -f environment.yml
```

当前 shell 未执行 `conda init`。使用下面的命令激活环境：

```bash
source /Users/wangzhe/miniforge3/bin/activate mario-ppo
```

无窗口运行 1,000 步环境测试：

```bash
python scripts/smoke_test.py
```

打开游戏窗口运行测试：

```bash
python scripts/smoke_test.py --steps 1_000 --render
```

旧版 Gym 会打印停止维护的提示，这是底层游戏包的已知依赖，并不表示测试失败。
项目固定使用 `gym==0.23.1`，因为该版本与游戏环境返回的四元组 `step()` 接口匹配。

项目的范围、阶段与验收标准见 [PROJECT_PLAN.md](PROJECT_PLAN.md)。

## 接口

最常用的就这几个：

```
import gym_super_mario_bros

env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
```

### `reset()`

重新开始一局，返回初始游戏画面：

```
state = env.reset()
```

`state` 是 RGB 图像，形状为：

```
(240, 256, 3)
```

### `step(action)`

执行一次操作：

```
next_state, reward, done, info = env.step(action)
```

- `next_state`：执行操作后的游戏画面
- `reward`：本次操作获得的奖励

```
reward = x_reward + time_penalty + death_penalty
reward = clip(reward, -15, 15)
横向位移 时间 死亡
没有通关奖励，金币奖励
```

- `done`：是否死亡或通关
- `info`：游戏的辅助信息

常用 `info` 字段：

```
info["x_pos"]      # 马里奥横向位置
info["y_pos"]      # 马里奥纵向位置
info["score"]      # 分数
info["time"]       # 剩余时间
info["life"]       # 生命数
info["status"]     # small、tall 或 fireball
info["flag_get"]   # 是否到达终点
```

### `render()`

显示游戏窗口：

```
env.render()
```

也可以直接获取当前画面：

```
image = env.render(mode="rgb_array")
```

### `close()`

结束时释放模拟器和窗口：

```
env.close()
```

### `action_space`

原始动作是 NES 手柄的 8 位按键组合：

```
print(env.action_space)          # Discrete(256)
action = env.action_space.sample()
```

实际训练通常会把 256 种组合缩减为几个常用动作，例如“不动、向右、向右跳”。

最小运行循环：

```
state = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    env.render()

    if done:
        state = env.reset()

env.close()
```

核心就是：`reset()` 开局，`step()` 操作，`render()` 显示，`close()` 结束。
