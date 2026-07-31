#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""s4l_model 工具：BackendTask → sim4life compiler → headless 执行 → 验证闭环。

编排链路（spec §3）：
  参数 → BackendTask → get_compiler("sim4life") → script_body
  → s4l_write_script（自动引导头）→ s4l_run_script（S4L 内核执行）
  → 验证（exit_code + .smash 产物 + REPORT| 实体报告断言）
  → 结构化结果 / 结构化错误（本期只记录，不自动修复）
"""

from __future__ import annotations

import time
from pathlib import Path

from research_agent.descriptor import ToolDescriptor, ToolSpec
from research_agent.literature.synthesis import BackendTask, get_compiler
from research_agent.tools import s4l_script


def _parse_report(stdout: str) -> dict:
    """解析脚本尾部打印的 REPORT| 实体报告。"""
    count = None
    names: list[str] = []
    materials: dict[str, str] = {}
    done = False
    for line in (stdout or "").splitlines():
        parts = line.strip().split("|")
        if len(parts) < 2 or parts[0] != "REPORT":
            continue
        if parts[1] == "ENTITY_COUNT" and len(parts) >= 3:
            try:
                count = int(parts[2])
            except ValueError:
                pass
        elif parts[1] == "ENTITY" and len(parts) >= 3:
            names.append(parts[2])
        elif parts[1] == "MATERIAL" and len(parts) >= 4:
            materials[parts[2]] = parts[3]
        elif parts[1] == "DONE":
            done = True
    return {"entity_count": count, "entity_names": names,
            "materials": materials, "report_done": done}


def _verify(run: dict, expected: dict) -> dict:
    """验证闭环：exit_code + .smash 产物 + 实体断言。"""
    checks: list[dict] = []

    ok_exit = run.get("exit_code") == 0
    checks.append({"name": "exit_code", "pass": ok_exit,
                   "detail": f"exit_code={run.get('exit_code')}"})

    smash = expected.get("smash_path", "")
    ok_smash = bool(smash) and Path(smash).is_file()
    checks.append({"name": "smash_exists", "pass": ok_smash, "detail": smash})

    report = _parse_report(run.get("stdout", ""))
    if report["entity_count"] is not None:
        need = expected.get("entity_count_min", 0)
        checks.append({
            "name": "entity_count",
            "pass": report["entity_count"] >= need,
            "detail": f"got {report['entity_count']}, need >= {need}",
        })
    else:
        checks.append({"name": "entity_count", "pass": False,
                       "detail": "no REPORT|ENTITY_COUNT in stdout"})

    missing = [n for n in expected.get("entity_names", [])
               if n not in report["entity_names"]]
    checks.append({"name": "entity_names",
                   "pass": not missing and report["report_done"],
                   "detail": f"missing={missing}" if missing else "all present"})

    # 材料指派核验（expected.materials 非空时）：每个实体材料值等于期望且非 FAIL
    exp_mats = expected.get("materials") or {}
    if exp_mats:
        bad = [
            f"{ent}: got {report['materials'].get(ent, '<missing>')}, want {want}"
            for ent, want in exp_mats.items()
            if report["materials"].get(ent) != want
        ]
        checks.append({"name": "materials",
                       "pass": not bad,
                       "detail": "; ".join(bad) if bad else "all assigned"})

    return {
        "all_pass": all(c["pass"] for c in checks),
        "checks": checks,
        "report": report,
    }


def _s4l_model(
    wing_diameter: float | None = None,
    radius: float | None = None,
    turns_per_wing: int = 1,
    wire_diameter_mm: float = 2.0,
    air_radius: float | None = None,
    timeout_s: int = 600,
) -> dict:
    """编译并 headless 执行一个 TMS 线圈建模脚本，返回验证报告。"""
    started = time.perf_counter()

    compiler = get_compiler("sim4life")
    if compiler is None:
        return {"status": "error", "stage": "compile",
                "error": "未注册 sim4life compiler"}

    if wing_diameter is None and radius is None:
        return {"status": "error", "stage": "input",
                "error": "wing_diameter 与 radius 至少给一个"}

    params = {"wire_diameter_mm": wire_diameter_mm}
    geom_params = {}
    if wing_diameter is not None:
        geom_params = {"radius": wing_diameter / 2.0, "wing_diameter": wing_diameter,
                       "turns_per_wing": turns_per_wing}
    else:
        geom_params = {"radius": radius}
    task = BackendTask(
        geometry_intent={"kind": "coil_sphere", "params": geom_params},
        constraints={"wire_diameter_mm": wire_diameter_mm},
        meta={"source": "s4l_model tool"},
    )

    overrides: dict = {}
    if air_radius:
        overrides["air_radius"] = air_radius
    compiled = compiler(task, overrides)
    body = compiled["script_body"]
    expected = compiled["expected"]

    # 写脚本 + 执行（复用 s4l_script 工具层）
    try:
        script_path = s4l_script._s4l_write_script(body, filename="s4l_model_task.py")["script_path"]
    except Exception as exc:
        return {"status": "error", "stage": "write_script", "error": str(exc)}

    try:
        run = s4l_script._s4l_run_script(script_path, timeout_s=timeout_s)
    except Exception as exc:
        return {"status": "error", "stage": "run", "error": str(exc),
                "script_path": script_path}

    verification = _verify(run, expected)
    duration = round(time.perf_counter() - started, 1)

    result = {
        "status": "ok" if verification["all_pass"] else "failed_verification",
        "script_path": script_path,
        "smash_path": expected.get("smash_path"),
        "verification": verification,
        "notes": compiled.get("notes", []),
        "duration_s": duration,
        "duration_kernel_s": run.get("duration_s"),
    }
    if run.get("exit_code") != 0:
        # 结构化错误（本期只记录）
        result["structured_error"] = {
            "stage": "s4l_kernel",
            "exit_code": run.get("exit_code"),
            "stderr_tail": (run.get("stderr") or "")[-500:],
        }
    return result


DESCRIPTOR = ToolDescriptor(
    spec=ToolSpec(
        name="s4l_model",
        description=(
            "用 Sim4Life headless 建一个 TMS 线圈模型（figure8 双翼或单环）："
            "编译经实测验证的模板脚本→S4L 内核执行→产出 .smash 并做实体断言验证。"
            "几何为同心圆环组近似（非真实螺旋）。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "wing_diameter": {"type": "number",
                                  "description": "figure8 翼径（米），给了就走双翼"},
                "radius": {"type": "number", "description": "单环半径（米），无翼径时用"},
                "turns_per_wing": {"type": "integer", "default": 1},
                "wire_diameter_mm": {"type": "number", "default": 2.0},
                "air_radius": {"type": "number", "description": "空气域球半径（米），默认自动"},
                "timeout_s": {"type": "integer", "default": 600,
                              "description": "S4L 内核执行超时（启动需 60-100s）"},
            },
            "required": [],
        },
        handler=_s4l_model,
    ),
    category="simulation",
    cost_hint="expensive",
    async_capable=True,
    requires=["sim4life_installed"],
    produces_artifacts=True,
)
