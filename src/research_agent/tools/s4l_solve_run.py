#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""s4l_solve_run 工具：编译 → 预检 → GUI 求解 → 场验证一键编排（L2，2026-08-29）。

分 stage 链路（每 stage 记 duration，失败即返回结构化错误，仿 _s4l_solve_benchmark）：
  compile   全配方编译两份产物：预检版（run_solve=False，切在 WriteInputFile 前）
            与求解版（run_solve=True）；
  preflight 预检版 headless 干跑（~25-60s，不占 GUI license），
            exit_code!=0 或缺 REPORT|PREFLIGHT|OK 即 fail-fast；
  wrap      求解版过 emit_gui_script（XCore.GetApp() 自适应头 + 日志文件重定向
            + os._exit(0)——自带头，不能再过 build_script 的 headless 引导头）；
  run       _s4l_run_gui（Sim4LifeLight.exe --minimized --run，超时杀进程树）；
  locate    results_dir = <smash>_Results，find_latest_output_h5；同时校验
            GUI 日志含 REPORT|SOLVE|HasResults|True；
  extract   extract_focal_profile（纯 h5py）；
  verify    两级：ref_h5 → compare_fields 定量锚（worst max_rel_err ≤1%）；
            shape 不一致（ValueError）降级特征判据并显式标注
            verify_mode=feature_fallback_shape_mismatch；无 ref → 特征判据。

磁盘保有（B3 71G 事故教训，spec R9）：报告 results_dir 全部 *_Output.h5 总体积，
只保留 mtime 最新 3 个，更老的删除并在 "cleaned" 字段记录（删除失败仅警告）。

license 单座纪律：QS_SOLVER 仅 GUI 进程持有，调用必须严格串行。
"""

from __future__ import annotations

import time
from pathlib import Path

from research_agent.config import SETTINGS
from research_agent.descriptor import ToolDescriptor, ToolSpec
from research_agent.literature.synthesis import BackendTask, get_compiler
from research_agent.s4lmodel import field_extract, gui_wrap, h5compare
from research_agent.tools import s4l_script, s4l_solve
from research_agent.tools.s4l_model import _parse_report

PREFLIGHT_TIMEOUT_S = 1500     # headless 预检超时（头模冷启动建模+体素化实测可 >600s，B2 冒烟口径）
REF_PASS_TOL = 0.01            # 定量锚：worst max_rel_err ≤1%（同基准复算口径）
OUTPUT_H5_KEEP = 3             # 磁盘保有：只保留最新 3 个 *_Output.h5


def _stage_error(stage: str, stages: dict, error: str, **extra) -> dict:
    return {"status": "error", "stage": stage, "stages": stages,
            "error": error, **extra}


def _retain_outputs(results_dir: str, keep: int = OUTPUT_H5_KEEP) -> dict:
    """磁盘保有：报告体积，删除最新 keep 个以外的 *_Output.h5。"""
    rd = Path(results_dir)
    outs = sorted(rd.glob("*_Output.h5"),
                  key=lambda p: p.stat().st_mtime, reverse=True) if rd.is_dir() else []
    total_mb = round(sum(p.stat().st_size for p in outs) / 1e6, 1)
    cleaned: list[str] = []
    warnings: list[str] = []
    for p in outs[keep:]:
        try:
            p.unlink()
            cleaned.append(str(p))
        except Exception as exc:  # 删除失败仅警告，不阻断报告
            warnings.append(f"清理失败 {p}: {exc}")
    return {"artifacts_size_mb": total_mb,
            "outputs_kept": [str(p) for p in outs[:keep]],
            "cleaned": cleaned, "warnings": warnings}


def _s4l_solve_run(
    wing_diameter: float | None = None,
    radius: float | None = None,
    turns_per_wing: int = 1,
    wire_diameter_mm: float = 2.0,
    with_head_model: bool = True,
    current_A: float = 1.0,
    freq_hz: float = 3000.0,
    ref_h5: str | None = None,
    timeout_s: int = 1800,
) -> dict:
    """编译→预检→GUI 求解→场提取→两级验证的分 stage 编排。"""
    stages: dict[str, dict] = {}

    def mark(name: str, t0: float, **info) -> None:
        stages[name] = {"ok": True, "duration_s": round(time.perf_counter() - t0, 2), **info}

    # ---- 1) compile：全配方两份产物（预检版 + 求解版）----
    t0 = time.perf_counter()
    compiler = get_compiler("sim4life")
    if compiler is None:
        return _stage_error("compile", stages, "未注册 sim4life compiler")
    if wing_diameter is None and radius is None:
        return _stage_error("compile", stages, "wing_diameter 与 radius 至少给一个")

    if wing_diameter is not None:
        geom_params = {"radius": wing_diameter / 2.0, "wing_diameter": wing_diameter,
                       "turns_per_wing": turns_per_wing}
    else:
        geom_params = {"radius": radius}
    task = BackendTask(
        geometry_intent={"kind": "coil_sphere", "params": geom_params},
        constraints={"wire_diameter_mm": wire_diameter_mm},
        meta={"source": "s4l_solve_run tool"},
    )
    overrides: dict = {
        "with_simulation": {"current_A": current_A, "freq_hz": freq_hz,
                            "run_solve": True},
    }
    if with_head_model:
        overrides["with_head_model"] = {"coil_gap_m": 0.002}
    try:
        solve_compiled = compiler(task, overrides)
        smash_path = solve_compiled["expected"]["smash_path"]
        # 预检版：同参数同 smash 路径，仅 run_solve=False（纯函数同参数→同脚本）
        preflight_overrides = dict(overrides, smash_path=smash_path)
        preflight_overrides["with_simulation"] = dict(
            overrides["with_simulation"], run_solve=False)
        preflight_compiled = compiler(task, preflight_overrides)
    except Exception as exc:
        return _stage_error("compile", stages, str(exc))
    mark("compile", t0, smash_path=smash_path)

    # ---- 2) preflight：headless 干跑，fail-fast 不触 GUI ----
    t0 = time.perf_counter()
    try:
        pre_script = s4l_script._s4l_write_script(
            preflight_compiled["script_body"], filename="s4l_preflight.py")["script_path"]
        pre_run = s4l_script._s4l_run_script(pre_script, timeout_s=PREFLIGHT_TIMEOUT_S)
    except Exception as exc:
        return _stage_error("preflight", stages, str(exc))
    if pre_run.get("exit_code") != 0 or \
            "REPORT|PREFLIGHT|OK" not in (pre_run.get("stdout") or ""):
        return _stage_error(
            "preflight", stages,
            "headless 预检未通过（脚本有错，未触 GUI）",
            preflight={"exit_code": pre_run.get("exit_code"),
                       "stderr_tail": (pre_run.get("stderr") or "")[-500:]},
            script_path=pre_script)
    mark("preflight", t0, duration_kernel_s=pre_run.get("duration_s"))

    # ---- 3) wrap：GUI --run 包装（自带 XCore 自适应头，直接写盘，不走
    #         _s4l_write_script——它强制加 headless 引导头）----
    t0 = time.perf_counter()
    artifacts = Path(SETTINGS.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    tag = int(time.time())
    gui_script = str(artifacts / f"s4l_solve_run_{tag}.py")
    log_path = str(artifacts / f"s4l_solve_run_{tag}.log")
    Path(gui_script).write_text(
        gui_wrap.emit_gui_script(solve_compiled["script_body"], log_path),
        encoding="utf-8")
    mark("wrap", t0, script_path=gui_script, log_path=log_path)

    # ---- 4) run：GUI --run 求解（license 单座，超时杀进程树）----
    t0 = time.perf_counter()
    try:
        run = s4l_solve._s4l_run_gui(gui_script, log_path, timeout_s=timeout_s)
    except Exception as exc:
        return _stage_error("run", stages, str(exc),
                            script_path=gui_script, log_path=log_path)
    if run.get("timed_out") or run.get("exit_code") != 0:
        return _stage_error(
            "run", stages,
            f"GUI 求解未完成（timed_out={run.get('timed_out')}, "
            f"exit_code={run.get('exit_code')}）",
            log_tail=(run.get("log") or "")[-1500:],
            script_path=gui_script, log_path=log_path)
    mark("run", t0, duration_gui_s=run.get("duration_s"))

    # ---- 5) locate：REPORT 校验 + 定位最新 Output.h5 ----
    t0 = time.perf_counter()
    report = _parse_report(run.get("log") or "")
    if "REPORT|SOLVE|HasResults|True" not in (run.get("log") or ""):
        return _stage_error(
            "locate", stages, "求解器未产出结果（日志无 REPORT|SOLVE|HasResults|True）",
            log_tail=(run.get("log") or "")[-1500:], report=report)
    results_dir = smash_path + "_Results"  # S4L 约定
    output_h5 = field_extract.find_latest_output_h5(results_dir)
    if not output_h5:
        return _stage_error("locate", stages,
                            f"结果目录无 *_Output.h5：{results_dir}")
    mark("locate", t0, output_h5=output_h5)

    # ---- 6) extract：焦点场指标（纯 h5py）----
    t0 = time.perf_counter()
    try:
        metrics = field_extract.extract_focal_profile(output_h5, current_A=current_A)
    except Exception as exc:
        return _stage_error("extract", stages, str(exc), output_h5=output_h5)
    mark("extract", t0)

    # ---- 7) verify：两级（定量锚优先，shape 不一致降级特征判据）----
    t0 = time.perf_counter()
    if ref_h5:
        try:
            stats = h5compare.compare_fields(output_h5, ref_h5)
            worst = stats["worst"]["max_rel_err"]
            verification = {
                "verify_mode": "reference",
                "verdict": "pass" if worst <= REF_PASS_TOL else "fail",
                "field_stats": stats,
                "tolerance": {"ref_pass_max_rel_err": REF_PASS_TOL},
                "ref_h5": ref_h5,
            }
        except ValueError as exc:
            # shape 不一致（全新体素化 vs 封存网格）→ 降级特征判据，显式标注
            feats = field_extract.verify_field_features(metrics)
            verification = {"verify_mode": "feature_fallback_shape_mismatch",
                            "fallback_reason": str(exc), "ref_h5": ref_h5, **feats}
    else:
        feats = field_extract.verify_field_features(metrics)
        verification = {"verify_mode": "feature", **feats}
    mark("verify", t0, verify_mode=verification["verify_mode"])

    # ---- 8) 报告 + 磁盘保有 ----
    retention = _retain_outputs(results_dir)
    result = {
        "status": "ok" if verification.get("verdict") == "pass" else "failed_verification",
        "stages": stages,
        "smash_path": smash_path,
        "output_h5": output_h5,
        "field_metrics": metrics,
        "verification": verification,
        "artifacts_size_mb": retention["artifacts_size_mb"],
        "outputs_kept": retention["outputs_kept"],
        "cleaned": retention["cleaned"],
        "notes": solve_compiled.get("notes", []),
        "script_path": gui_script,
        "log_path": log_path,
    }
    if retention["warnings"]:
        result["warnings"] = retention["warnings"]
    return result


DESCRIPTOR = ToolDescriptor(
    spec=ToolSpec(
        name="s4l_solve_run",
        description=(
            "Sim4Life 头模有损求解一键编排：论文参数 → 全配方编译 → headless 预检"
            "（fail-fast，不占 license）→ GUI --run 有损求解（QS_SOLVER，420s 级）"
            "→ Output.h5 焦点场提取 → 两级验证（有 ref_h5 走逐体素定量锚 ≤1%；"
            "shape 不一致自动降级特征判据并显式标注；无 ref 走特征判据）。"
            "分 stage 结构化错误（compile/preflight/wrap/run/locate/extract/verify），"
            "报告产物体积并自动清理旧输出（保留最新 3 个）。"
            "license 单座：调用必须严格串行，禁止与其他 S4L 任务并发。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "wing_diameter": {"type": "number",
                                  "description": "figure8 翼径（米），给了就走双翼"},
                "radius": {"type": "number", "description": "单环半径（米），无翼径时用"},
                "turns_per_wing": {"type": "integer", "default": 1},
                "wire_diameter_mm": {"type": "number", "default": 2.0},
                "with_head_model": {"type": "boolean", "default": True,
                                    "description": "三层球壳头模（脑/颅骨/头皮）；False 为空气域"},
                "current_A": {"type": "number", "default": 1.0, "description": "激励电流（A）"},
                "freq_hz": {"type": "number", "default": 3000.0, "description": "激励频率（Hz）"},
                "ref_h5": {"type": "string",
                           "description": "可选参照 Output.h5（定量锚）；shape 不一致自动降级特征判据"},
                "timeout_s": {"type": "integer", "default": 1800,
                              "description": "GUI 冷启动 3-5min + 求解数分钟，勿设太小"},
            },
            "required": [],
        },
        handler=_s4l_solve_run,
    ),
    category="simulation",
    cost_hint="expensive",
    async_capable=True,
    requires=["sim4life_gui"],
    produces_artifacts=True,
)
