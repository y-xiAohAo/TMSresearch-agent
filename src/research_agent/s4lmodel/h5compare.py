#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Output.h5 基准对比模块（B3 定稿 2026-08-03）。

设计依据（实测 + pitfalls）：
- 结构：FieldGroups/<uuid>/AllFields/<场名>/_Object/Snapshots/0/comp{0,1,2}，
  最后一维 (re, im)（pitfall #27）。
- E 场 edge-centered（三分量 shape 交错），B/H/A cell-centered——
  对比必须逐分量进行，不能拼成单一矢量场（pitfall #28）。
- 域外体素为 NaN（BMEN16 实测 ~64%）——全部统计 nan 安全（pitfall #28）。
- 弱场体素相对误差无意义——mask = ref > mask_ratio * nanmax(ref)（skill §5）。
- 复算对比双方是同一模型同一网格，shape 必然一致；不一致即链路缺陷，直接报错。

本模块不 import s4l_v1，可在任意机器解析（编排层零 license 依赖）。
"""

from __future__ import annotations

import h5py
import numpy as np

DEFAULT_FIELDS = ("EM E(x,y,z,f0)",)


def _find_all_fields(h5: "h5py.File") -> "h5py.Group":
    for grp in h5["FieldGroups"].values():
        if "AllFields" in grp:
            return grp["AllFields"]
    raise KeyError("FieldGroups 中未找到 AllFields 组")


def load_field_components(h5_path: str, field_name: str) -> dict[str, "np.ndarray"]:
    """加载指定场的各分量复数模（|re + i·im|），NaN 保留。

    返回 {"comp0": ndarray, ...}；分量 shape 可交错（edge-centered 场）。
    """
    with h5py.File(h5_path, "r") as h5:
        fields = _find_all_fields(h5)
        if field_name not in fields:
            raise KeyError(f"场不存在：{field_name}（可用：{list(fields.keys())}）")
        snaps = fields[field_name]["_Object"]["Snapshots"]["0"]
        out: dict[str, np.ndarray] = {}
        for comp in snaps.keys():
            d = np.array(snaps[comp])
            out[comp] = np.abs(d[..., 0] + 1j * d[..., 1])
        return out


def _rel_err_stats(sim: "np.ndarray", ref: "np.ndarray", mask_ratio: float) -> dict:
    """强场体素上的相对误差统计（nan 安全）。"""
    if sim.shape != ref.shape:
        raise ValueError(f"shape 不一致：sim {sim.shape} vs ref {ref.shape}（链路缺陷）")
    finite = np.isfinite(sim) & np.isfinite(ref)
    threshold = mask_ratio * float(np.nanmax(ref)) if finite.any() else 0.0
    mask = finite & (ref > threshold)
    n = int(mask.sum())
    if n == 0:
        return {"n_voxels": 0, "max_rel_err": float("nan"),
                "median_rel_err": float("nan"), "p99_rel_err": float("nan")}
    rel = np.abs(sim[mask] - ref[mask]) / ref[mask]
    return {
        "n_voxels": n,
        "max_rel_err": float(np.max(rel)),
        "median_rel_err": float(np.median(rel)),
        "p99_rel_err": float(np.percentile(rel, 99)),
    }


def compare_fields(
    sim_h5: str,
    ref_h5: str,
    field_names: tuple = DEFAULT_FIELDS,
    mask_ratio: float = 0.01,
) -> dict:
    """逐场逐分量对比两个 Output.h5。

    返回 {
      "fields": {场名: {分量: {n_voxels, max_rel_err, median_rel_err, p99_rel_err}}},
      "worst": {"max_rel_err": ..., "p99_rel_err": ...},   # 跨全部分量的最差值
    }
    """
    result: dict = {"fields": {}}
    worst_max: float | None = None
    worst_p99: float | None = None
    for name in field_names:
        sim_comps = load_field_components(sim_h5, name)
        ref_comps = load_field_components(ref_h5, name)
        stats: dict = {}
        for comp, ref in ref_comps.items():
            if comp not in sim_comps:
                raise KeyError(f"sim 缺分量：{name}/{comp}")
            s = _rel_err_stats(sim_comps[comp], ref, mask_ratio)
            stats[comp] = s
            if s["n_voxels"]:
                worst_max = s["max_rel_err"] if worst_max is None else max(worst_max, s["max_rel_err"])
                worst_p99 = s["p99_rel_err"] if worst_p99 is None else max(worst_p99, s["p99_rel_err"])
        result["fields"][name] = stats
    # 无任何强场体素时落 NaN——判定层据此 fail，而非误 pass
    result["worst"] = {
        "max_rel_err": worst_max if worst_max is not None else float("nan"),
        "p99_rel_err": worst_p99 if worst_p99 is not None else float("nan"),
    }
    return result


def compare_linearity(
    scaled_h5: str,
    ref_h5: str,
    scale: float,
    field_names: tuple = DEFAULT_FIELDS,
    mask_ratio: float = 0.01,
) -> dict:
    """方案 A 线性缩放验证：scaled 场应 ≈ scale × ref 场（强场体素）。

    返回每分量 {ratio_median, ratio_rel_err_vs_scale, n_voxels} 及 worst。
    线性区物理：B/E 与激励电流成正比（spec §0.1）。
    """
    result: dict = {"fields": {}, "scale": float(scale)}
    worst: float | None = None
    for name in field_names:
        sim_comps = load_field_components(scaled_h5, name)
        ref_comps = load_field_components(ref_h5, name)
        stats: dict = {}
        for comp, ref in ref_comps.items():
            sim = sim_comps[comp]
            if sim.shape != ref.shape:
                raise ValueError(f"shape 不一致：{name}/{comp}")
            finite = np.isfinite(sim) & np.isfinite(ref)
            threshold = mask_ratio * float(np.nanmax(ref)) if finite.any() else 0.0
            mask = finite & (ref > threshold)
            n = int(mask.sum())
            if not n:
                stats[comp] = {"n_voxels": 0, "ratio_median": float("nan"),
                               "ratio_rel_err_vs_scale": float("nan")}
                continue
            ratio = float(np.median(sim[mask] / ref[mask]))
            rel_err = abs(ratio - float(scale)) / float(scale)
            stats[comp] = {"n_voxels": n, "ratio_median": ratio,
                           "ratio_rel_err_vs_scale": rel_err}
            worst = rel_err if worst is None else max(worst, rel_err)
        result["fields"][name] = stats
    result["worst"] = {"ratio_rel_err_vs_scale":
                       worst if worst is not None else float("nan")}
    return result
