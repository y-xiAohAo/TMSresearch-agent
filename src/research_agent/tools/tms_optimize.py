#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tms_optimize 工具：TMS 流函数线圈优化的薄 facade（M1 单算法小参数模板）。

通过子进程调用 StreamFunctionTMS 项目自己的 venv（含 pymoo/cvxpy/h5py），
避免把重依赖引入 research-agent 环境。M1 只支持单椭圆目标 + NSGA2。
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from research_agent.config import SETTINGS
from research_agent.descriptor import ToolDescriptor, ToolSpec

_DRIVER_TEMPLATE = '''# -*- coding: utf-8 -*-
import json, sys
sys.path.insert(0, r"{tms_src}")
from tms_optimizer.config.schemas import ExperimentConfig
from tms_optimizer.orchestration.experiment import run_experiment

cfg = ExperimentConfig.model_validate(json.load(open(sys.argv[1], encoding="utf-8")))
result = run_experiment(cfg)
summary = {{
    "execution_time_sec": result.get("execution_time_sec"),
    "n_pareto_solutions": len(result.get("pareto_solutions", [])),
    "pareto_objectives": result.get("pareto_objectives"),
    "focus_areas_mean": result.get("focus_areas_mean"),
    "oa90_per_solution": result.get("oa90_per_solution"),
    "target2max_per_solution": result.get("target2max_per_solution"),
}}
print("TMS_RESULT_JSON=" + json.dumps(summary, default=str))
'''


def _tms_optimize(problem_spec: dict | None = None, budget_s: int = 120) -> dict:
    """运行一次小规模 NSGA2 优化，返回 {status, metrics, config_path, duration_s}。"""
    if not SETTINGS.tms_python or not Path(SETTINGS.tms_python).is_file():
        raise RuntimeError(f"TMS_PYTHON 未配置或不存在：{SETTINGS.tms_python}")
    problem_spec = problem_spec or {}
    tms_src = str(Path(SETTINGS.tms_project_dir) / "src")

    artifacts = Path(SETTINGS.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    run_dir = artifacts / f"tms_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "coil_radius": float(problem_spec.get("coil_radius", 0.090)),
        "head_radius": float(problem_spec.get("head_radius", 0.090)),
        "Nsteps": int(problem_spec.get("Nsteps", 3)),
        "ellipses": problem_spec.get(
            "ellipses",
            [{"center_theta": 0.5, "center_phi": 0.0, "a": 0.2, "b": 0.1,
              "alpha": 0.0, "sampling_interval": 0.05}],
        ),
        "nsga2": {
            "pop_size": int(problem_spec.get("pop_size", 10)),
            "n_gen": int(problem_spec.get("n_gen", 3)),
            "seed": int(problem_spec.get("seed", 42)),
        },
    }
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    driver_path = run_dir / "driver.py"
    driver_path.write_text(_DRIVER_TEMPLATE.format(tms_src=tms_src), encoding="utf-8")

    started = time.perf_counter()
    try:
        proc = subprocess.run(
            [SETTINGS.tms_python, str(driver_path), str(config_path)],
            cwd=SETTINGS.tms_project_dir,
            capture_output=True,
            text=True,
            timeout=max(10, int(budget_s)),
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "metrics": {},
            "config_path": str(config_path),
            "duration_s": budget_s,
        }
    duration = round(time.perf_counter() - started, 2)

    metrics: dict = {}
    for line in proc.stdout.splitlines():
        if line.startswith("TMS_RESULT_JSON="):
            try:
                metrics = json.loads(line[len("TMS_RESULT_JSON="):])
            except json.JSONDecodeError:
                pass
    status = "ok" if proc.returncode == 0 else "error"
    if proc.returncode != 0:
        metrics["stderr_tail"] = proc.stderr[-1500:] if proc.stderr else ""
    return {
        "status": status,
        "metrics": metrics,
        "config_path": str(config_path),
        "duration_s": duration,
    }


DESCRIPTOR = ToolDescriptor(
    spec=ToolSpec(
        name="tms_optimize",
        description=(
            "运行 TMS 流函数线圈优化（NSGA2，小参数模板）。输入椭圆目标区域等参数，"
            "返回优化摘要（耗时、解数、目标值）。M1 为单椭圆小规模模板。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "problem_spec": {
                    "type": "object",
                    "description": "可选覆盖：coil_radius/head_radius/Nsteps/ellipses/pop_size/n_gen/seed",
                },
                "budget_s": {"type": "integer", "default": 120, "description": "硬超时秒数"},
            },
            "required": [],
        },
        handler=_tms_optimize,
    ),
    category="compute",
    cost_hint="expensive",
    async_capable=True,
    requires=["tms_venv"],
    produces_artifacts=True,
)
