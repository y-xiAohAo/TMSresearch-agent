#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""s4lmodel：Sim4Life 建模脚本生成层。

架构约束（spec 2026-07-31_phase-b2-s4l-modeling.md）：
- 本包所有函数都是**纯脚本发射器**（参数 → s4l_v1 代码字符串），
  不 import s4l_v1、不依赖 S4L 环境，因此可无 S4L 单元测试。
- `setup.py` 为拓扑无关的共享组件（空气域/材料/仿真设置/SaveAs+实体报告）。
- 拓扑特化只允许出现在 `*_geometry.py` 模块中。
- 所有模板必须经 headless 实测验证后才可视为可用（冒烟测试保障）。
"""
