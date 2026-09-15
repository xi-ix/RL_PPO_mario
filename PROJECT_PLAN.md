# 基于 gym-super-mario-bros 的 PPO 学习项目任务书

## 1. 项目定位

本项目从最基础的组件开始，实现一个能够操作《Super Mario Bros.》第一关的强化学习智能体。

`gym-super-mario-bros` 和 `nes-py` 只负责运行游戏并提供控制接口：

```text
reset()                 重置游戏
step(action)            执行动作并返回画面、奖励和结束状态
render()                显示游戏窗口
observation             游戏画面
info                    角色位置、生命、关卡状态等信息
```

项目不使用 Stable-Baselines3、CleanRL、RLlib 等现成强化学习实现，也不使用 Gym 的向量环境、图像包装器、帧堆叠或训练工具。以下部分由项目自行编写：

- 动作映射
- 图像预处理
- 帧跳过与帧堆叠
- 轨迹采集和批次管理
- Actor-Critic 神经网络
- GAE 优势估计
- PPO 损失和参数更新
- 模型保存、加载、评估和录像

说明：`gym-super-mario-bros` 本身通过 Gym 风格 API 暴露 NES 模拟器，因此底层仍会出现 `reset()`、`step()` 和 `action_space` 等接口。本项目所说的“不使用 Gym 框架”，是指不依赖它完成强化学习逻辑，仅把它当作游戏控制库。

## 2. 学习目标

完成项目后，应能解释并独立实现以下完整数据流：

```text
游戏画面
  -> 图像预处理
  -> Actor-Critic 网络
  -> 动作概率分布
  -> 采样动作并操作游戏
  -> 收集奖励和下一状态
  -> 计算回报与优势
  -> PPO 小批量更新
  -> 保存并评估新策略
```

主要目标：

1. 理解强化学习的状态、动作、奖励、回合和策略。
2. 理解卷积神经网络如何从连续游戏画面判断运动状态。
3. 从公式和代码两个层面掌握 Actor-Critic、GAE 和 PPO。
4. 建立可观察、可测试、可复现的训练流程。
5. 训练模型在 `SuperMarioBros-1-1-v3` 中稳定向右前进并越过基础障碍。

首期不要求稳定通关。通关作为进阶目标，避免把环境调试、算法实现和长时间调参混在一起。

## 3. 技术边界

### 允许使用

- Python 3.10
- `gym-super-mario-bros` 和 `nes-py`：启动、控制和读取游戏
- PyTorch：张量、自动求导、神经网络和优化器
- NumPy：数值处理
- OpenCV 或 Pillow：图像缩放和灰度化
- TensorBoard：记录训练指标
- ImageIO 或 OpenCV：生成评估录像
- Pytest：单元测试

### 不使用

- Stable-Baselines3 的 PPO 或策略网络
- Gym/Gymnasium 的 observation、frame stack、reward、vector env 等包装器
- CleanRL、RLlib、Sample Factory 等训练框架
- 预训练模型和人工演示数据
- 未经确认来源的 ROM 或游戏资源

### 关于动作接口

第一版只使用三个离散动作：

```text
0: NOOP         不操作
1: RIGHT        向右移动
2: RIGHT_JUMP   向右移动并跳跃
```

动作到 NES 按键位的转换由项目自己的 `ActionAdapter` 完成。如果底层库版本只能通过 `JoypadSpace` 正确发送组合键，可以把它作为临时的游戏控制适配器，但不得在其中加入训练逻辑。

## 4. 首期功能范围

### 游戏环境

- 默认关卡：`SuperMarioBros-1-1-v3`
- 能用固定随机种子重置环境
- 能以人工键盘、固定动作或随机动作控制角色
- 能读取当前画面和 `info` 中的关键字段
- 能识别死亡、超时和通关
- 能在可视化与无窗口训练模式间切换

### 观测处理

- 将原始 RGB 图像转为灰度图
- 缩放为 `84 x 84`
- 数值归一化到 `[0, 1]`
- 每个动作持续 4 个模拟器帧
- 对跳过期间的最后两帧执行逐像素最大值，减少闪烁
- 堆叠最近 4 帧，最终状态形状为 `(4, 84, 84)`

### PPO 算法

- 共享 CNN 特征提取器
- Actor 输出三个动作的 logits
- Critic 输出状态价值标量
- 使用 categorical distribution 采样动作
- 使用 GAE-Lambda 计算优势
- 使用 clipped surrogate objective 更新策略
- 使用价值损失和熵奖励
- 支持梯度裁剪
- 对优势做标准化

### 训练和评估

- 定期保存 checkpoint
- 支持从 checkpoint 恢复训练
- 训练与评估使用不同的动作方式：训练采样，评估取最大概率动作
- 定期录制一局评估视频
- 输出奖励、最远横坐标、存活步数、策略损失、价值损失、熵和近似 KL
- 配置随机种子，尽量保证实验可复现

## 5. 建议目录结构

```text
PPO_RL/
├── README.md
├── PROJECT_PLAN.md
├── requirements.txt
├── src/
│   ├── game.py             # 模拟器创建、reset/step 兼容处理（似乎没兼容问题）
│   ├── actions.py          # 简化动作到 NES 按键的映射
│   ├── preprocessing.py    # 灰度、缩放、帧跳过、帧堆叠
│   ├── model.py            # CNN Actor-Critic
│   ├── rollout.py          # 轨迹缓冲区
│   ├── returns.py          # discounted return 和 GAE
│   ├── ppo.py              # PPO 损失与更新
│   ├── train.py            # 训练入口
│   └── evaluate.py         # 可视化评估与录像
├── tests/
│   ├── test_actions.py
│   ├── test_preprocessing.py
│   ├── test_rollout.py
│   ├── test_returns.py
│   └── test_model.py
├── configs/
│   └── default.yaml
├── checkpoints/
├── logs/
└── videos/
```

`checkpoints/`、`logs/` 和 `videos/` 中的运行产物不提交到版本控制。

## 6. 分阶段实施计划

### 阶段 0：环境与兼容性验证

任务：

- 创建 Python 3.10 虚拟环境。
- 安装并锁定依赖版本。
- 启动 `SuperMarioBros-1-1-v3`。
- 连续执行随机动作 1,000 步。
- 打印 observation 的形状、数据类型、奖励、结束标记和 `info` 字段。
- 验证窗口能够正常显示并关闭。

验收标准：

- 程序运行至少 1,000 步且没有接口或渲染异常。
- 角色死亡后能够自动开始新回合。
- 能明确记录当前依赖组合和 Mac 架构信息。

交付物：`requirements.txt`、`scripts/smoke_test.py`。

### 阶段 1：动作控制与人工试玩

任务：

- 实现三个简化动作及 NES 按键转换。
- 编写固定“持续向右”和随机动作策略。
- 编写可选的键盘试玩模式，用于验证按键语义。
- 统计每回合最远横坐标、奖励和结束原因。

验收标准：

- `RIGHT` 能持续向右移动。
- `RIGHT_JUMP` 能同时向右和跳跃。
- 动作编号、网络输出维度和实际游戏动作完全一致。

交付物：`src/actions.py`、`scripts/play_manual.py`、动作测试。

### 阶段 2：从原始画面构造状态

任务：

- 独立实现灰度化、缩放、归一化、帧跳过和帧堆叠。
- 保存原始帧和处理后帧进行肉眼对比。
- 对输入、输出形状和数值范围编写测试。

验收标准：

- 任意原始帧稳定转换为 `(1, 84, 84)`。
- 堆叠状态稳定为 `(4, 84, 84)`。
- reset 后的帧栈不包含上一回合残留图像。
- 预处理后仍能辨认角色、地面、敌人和水管。

交付物：`src/preprocessing.py`、相关测试和样例图片。

### 阶段 3：Actor-Critic 网络

任务：

- 编写 CNN 特征提取器。
- 编写策略头和价值头。
- 实现动作采样、动作 log probability、熵和状态价值计算。
- 用随机张量完成前向和反向传播测试。

验收标准：

- 输入 `(batch, 4, 84, 84)` 后输出 `(batch, 3)` logits 和 `(batch,)` value。
- 动作概率和为 1，所有输出有限。
- 所有预期参数都能获得有限梯度。

交付物：`src/model.py` 和模型测试。

### 阶段 4：轨迹缓冲区与 GAE

任务：

- 保存 state、action、reward、done、value 和 log probability。
- 区分真正终止与时间截断，正确处理 bootstrap value。
- 从公式独立实现 discounted return 和 GAE-Lambda。
- 使用手工构造的小数组验证计算结果。

核心公式：

```text
delta_t = r_t + gamma * (1 - done_t) * V(s_{t+1}) - V(s_t)
A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
return_t = A_t + V(s_t)
```

验收标准：

- 单回合、多回合和截断场景均通过精确数值测试。
- 缓冲区不会跨死亡边界传播优势。

交付物：`src/rollout.py`、`src/returns.py` 和数值测试。

### 阶段 5：PPO 更新器

任务：

- 重新计算新策略下的 log probability 和 value。
- 计算 probability ratio 和 clipped policy loss。
- 加入 value loss、entropy bonus 和梯度裁剪。
- 对同一批 rollout 执行多轮 shuffled minibatch 更新。
- 监控 clip fraction 和 approximate KL。

核心目标：

```text
ratio = exp(new_log_prob - old_log_prob)
policy_loss = -mean(min(ratio * advantage,
                        clip(ratio, 1-eps, 1+eps) * advantage))
total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
```

验收标准：

- 人造批次上的损失与手算结果一致。
- 一次更新后网络参数发生变化。
- 极端 advantage 或 ratio 不产生 NaN/Inf。

交付物：`src/ppo.py` 和 PPO 单元测试。

### 阶段 6：端到端短训练

任务：

- 串联游戏、预处理、模型、rollout 和 PPO 更新。
- 先执行约 100,000 到 500,000 环境步的短训练。
- 每轮输出关键训练指标。
- 定期保存 checkpoint 和一局评估录像。

验收标准：

- 训练可以连续运行、保存、退出和恢复。
- 所有损失保持有限值。
- 与随机策略相比，模型的平均最远横坐标有明确提升。
- 能加载 checkpoint 并在可视化窗口中运行。

交付物：`src/train.py`、`src/evaluate.py`、配置、checkpoint 和录像。

### 阶段 7：长训练与行为改进

只有阶段 6 稳定后才进行：

- 增加训练步数。
- 根据录像判断失败类型，而不是只看总奖励。
- 调整熵系数、学习率、rollout 长度和 PPO clip 范围。
- 必要时增加“不前进超时”规则。
- 比较原始游戏奖励与少量奖励塑形。

验收标准：

- 至少使用 20 个独立评估回合报告均值和标准差。
- 最远横坐标、越过障碍数量或通关率显著超过随机基线。
- 保留训练配置、随机种子和对应 checkpoint，结果可追溯。

## 7. 第一版建议超参数

这些参数只是可运行起点，不视为最终最优值：

```yaml
environment:
  level: SuperMarioBros-1-1-v3
  frame_skip: 4
  frame_stack: 4
  image_size: [84, 84]
  max_stall_steps: 500

ppo:
  gamma: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.2
  learning_rate: 0.00025
  rollout_steps: 2048
  minibatch_size: 256
  update_epochs: 4
  value_coef: 0.5
  entropy_coef: 0.01
  max_grad_norm: 0.5
```

单环境按顺序采集数据最利于理解，但样本生成会较慢。首期坚持单环境；算法正确后，再将多进程采样作为独立的性能优化任务。

## 8. 奖励策略

第一版使用游戏接口返回的原始奖励，先验证 PPO 是否能够学习，不立即设计复杂奖励。

建议同时记录但不一定加入奖励的指标：

- `x_pos`：最远横向位置
- `score`：游戏分数
- `time`：剩余时间
- `life`：剩余生命
- `flag_get`：是否到达终点

只有当录像证明原始奖励导致具体问题时，再逐项引入：

- 长时间没有刷新最远位置时提前结束回合。
- 死亡给予固定惩罚。
- 到达终点给予一次性奖励。

每次只改变一项，并保存对照实验，避免无法判断是哪项奖励改变了行为。

## 9. 测试与调试原则

调试顺序固定为：

1. 游戏动作是否正确。
2. 回合边界是否正确。
3. 图像和帧栈是否正确。
4. GAE 数值是否正确。
5. PPO ratio、clip 和梯度是否正确。
6. 最后才调整超参数和奖励。

必须具备以下保护：

- 对 observation、reward、loss 和 gradient 检查 NaN/Inf。
- 保存训练配置和随机种子。
- checkpoint 使用临时文件写完后再替换，避免中断产生损坏文件。
- 评估不参与训练数据收集。
- 定期保存录像，防止奖励升高但行为实际退化。

## 10. 风险与应对

### 依赖较旧

风险：`gym-super-mario-bros`、`nes-py` 与新版 Python、NumPy 或 Gym API 不兼容。

应对：使用 Python 3.10 独立环境，锁定全部版本；在写算法前完成阶段 0。兼容代码集中放在 `src/game.py`，不让旧 API 污染其余模块。

### macOS 渲染和编译问题

风险：Apple Silicon 上可能需要本地编译，窗口渲染方式也可能与 Linux 不同。

应对：先验证架构、Python 和编译工具；训练默认无窗口，仅评估时渲染。必要时再评估 Conda 或 Linux 环境，不在算法代码中写平台特例。

### 单环境采样较慢

风险：NES 模拟器主要消耗 CPU，长训练耗时明显。

应对：先以正确性和学习效果为目标；完成单环境版本后再做并行采样，不提前增加复杂度。

### 奖励投机

风险：模型可能持续撞墙、频繁无意义跳跃，或通过某种局部行为积累奖励。

应对：同时观察最远横坐标、终点率和录像；奖励调整必须基于可复现的失败行为。

### 游戏资源与许可

风险：模拟器、环境代码与游戏内容的许可并不相同。

应对：检查项目许可证及当地适用规则，只使用合法获得的游戏资源，不从不明站点下载或在仓库中提交 ROM。

## 11. 完成定义

首期项目在同时满足以下条件时完成：

- 项目可在一条文档化命令下启动训练。
- PPO、GAE、rollout 和图像处理均由本项目实现并有测试。
- 不依赖现成强化学习算法或训练框架。
- 能保存和恢复模型。
- 能实时显示训练后模型操作游戏，并能导出录像。
- 至少完成随机策略与训练策略的量化对比。
- 训练策略在第一关的平均最远横坐标明显超过随机基线。
- README 解释环境安装、训练、评估、配置和已知限制。

通关不是首期完成的硬性条件。首期的核心成果是正确、透明、能够学习的 PPO 实现。

## 12. 推荐执行顺序

严格按以下顺序推进，每一步通过验收后再进入下一步：

```text
环境冒烟测试
-> 动作控制
-> 图像预处理
-> Actor-Critic
-> Rollout 与 GAE
-> PPO 更新
-> 短训练
-> 可视化评估
-> 长训练与调参
```

第一个实际开发任务是阶段 0：在当前 Mac 上确定一组可运行的固定依赖，并让随机动作控制的马里奥连续运行 1,000 步。
