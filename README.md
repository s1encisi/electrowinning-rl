# Electrowinning RL

铜电积设定值优化的代理模型与约束强化学习实现。包含 ExtraTrees 代理、PPO-Lagrangian、NSGA-II、随机参照及统一可行性评价。

## 运行

使用 Python 3.11：

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m electrowinning_rl demo --config configs/smoke.json --condition 3 --output runs/smoke
python -m pytest -q
```

公开演示使用合成数据，无需私有数据集、GPU 或云端模型 API。优化在代理模型上进行，不代表工厂控制效果。源码位于 `src/electrowinning_rl/`，运行配置位于 `configs/`。

许可证：[MIT](LICENSE)。
