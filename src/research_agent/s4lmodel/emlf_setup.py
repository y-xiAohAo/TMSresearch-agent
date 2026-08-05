#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""emlf 仿真设置发射器（拓扑无关共享组件，B3 定稿 2026-08-03）。

API 模式以 GUI --run 全量实测（2026-08-01/02，30+ 轮）为准：
- 电流源必须挂**线框**（model.CreateCircle）；挂实体 tube 报
  "The current loops require to consist of at least one segment!"（pitfall #17）。
- 源绑定：cs = sim.AddCurrentSourceSettings([wire])（类型化添加，错误信息更明确）。
- 幅值/线径必须带单位元组：cs.Amplitude = (I, units.Amperes)；裸浮点求解器解释不同。
- 两翼反向：CurrentSourceSettings.IsDirectionReverted = True（探针确认存在）。
- 网格/传感器：sim.AddAutomaticGridSettings()；
  sim.AddOverallFieldSensorSettings().RecordHField = True。
- 频率：sim.SetupSettings.Frequency（TMS 等效 ~3kHz；B 场线性区与频率无关）。
- 求解顺序铁律（pitfall #20）：
  UpdateAllMaterials → UpdateGrid → SaveAs → AddAutomaticVoxelerSettings
  → CreateVoxels → AllSimulations.Add → RunSimulation(wait=True)。
  本模块 emit_solve 不含 SaveAs——由上游 setup.emit_save_and_report 完成，
  发射顺序必须保证 save 段在 solve 段之前。
- 场提取不走 S4L 后处理 API（pitfall #26）：编排层 h5py 直读 Output.h5
  （见 s4lmodel/h5compare.py），故本模块不含提取段。

物理声明（spec §0.1）：B 场线性区与频率无关；TMS 等效频率仅求解器需要。
诚实边界（pitfall #18）：空气域 MQS "nothing to solve"——本发射器生成的
脚本作为 GUI 可运行资产交付，数值验证走基准复算路径（tools/s4l_solve.py）。
"""

from __future__ import annotations


def _fmt(x: float) -> str:
    return f"{float(x):.9g}"


def emit_current_source_wires(rings: list[dict]) -> tuple[str, list[str]]:
    """为每匝发射 CreateCircle 电流源线框（电流源必须挂线框，pitfall #17）。

    rings: [{"name": str, "center": (x,y,z), "radius": float,
             "normal": (nx,ny,nz) 可选，默认 (0,0,1)}]
    线框实体命名 "{name}_wire"。返回 (脚本片段, 线框实体名列表)。
    与几何环同圆心同半径——圆心/半径由调用方从几何参数推导（如 analytical.wing_radii）。
    """
    lines = ["", "# --- emlf: current source wires (CreateCircle) ---"]
    names: list[str] = []
    for ring in rings:
        name = f"{ring['name']}_wire"
        cx, cy, cz = (float(v) for v in ring["center"])
        nx, ny, nz = (float(v) for v in ring.get("normal", (0.0, 0.0, 1.0)))
        r = float(ring["radius"])
        names.append(name)
        lines.append(
            f"_w = model.CreateCircle(\n"
            f"    model.Vec3({_fmt(cx)}, {_fmt(cy)}, {_fmt(cz)}),\n"
            f"    model.Vec3({_fmt(nx)}, {_fmt(ny)}, {_fmt(nz)}),\n"
            f"    {_fmt(r)},\n"
            f")\n"
            f"try:\n"
            f"    _w.Name = \"{name}\"\n"
            f"except Exception:\n"
            f"    pass"
        )
    return "\n".join(lines) + "\n", names


def emit_mqs_simulation(
    sim_name: str,
    positive_wires: list[str],
    current_A: float,
    wire_radius_m: float,
    negative_wires: list[str] | None = None,
    freq_hz: float = 3000.0,
    grid_entities: list[str] | None = None,
    max_step_m: float | None = None,
    padding_m: float | None = None,
) -> str:
    """MQS 仿真 + 电流源（正负两组线框，负组 IsDirectionReverted=True）。

    positive/negative 为线框实体名列表（运行时按名查实体绑定）。
    figure8：两翼各为一组，电流方向相反（spec §0 物理关键）。
    negative_wires 为空时不发射负组块（避免 AddCurrentSourceSettings([]) 空调用）。

    grid_entities（B4 定案，pitfall #39）：显式绑定后网格才纳入这些实体
    （网格纳入=体素化纳入；无参调用 = 什么都不纳入）。
    max_step_m：有损区细网格（ManualGridSettings.MaxStep，嵌套元组）。
    padding_m：GlobalGridSettings 手动 padding（嵌套元组）。
    """
    neg = negative_wires or []
    neg_block = ""
    if neg:
        neg_block = f'''
_cs_neg = sim.AddCurrentSourceSettings(_ents_by_names({neg!r}))
_cs_neg.Name = "coil_neg"
_cs_neg.Amplitude = ({_fmt(current_A)}, units.Amperes)
_cs_neg.Radius = ({_fmt(wire_radius_m)}, units.Meters)
_cs_neg.IsDirectionReverted = True
'''
    if grid_entities:
        grid_block = f'''
sim.AddAutomaticGridSettings(_ents_by_names({grid_entities!r}))
'''
        if max_step_m:
            grid_block += f'''_mgs = sim.AddManualGridSettings(_ents_by_names({grid_entities!r}))
_mgs.MaxStep = (({_fmt(max_step_m)}, {_fmt(max_step_m)}, {_fmt(max_step_m)}), units.Meters)
'''
        if padding_m:
            grid_block += f'''_ggs = sim.GlobalGridSettings
_ggs.ManualPadding = True
_ggs.BottomPadding = (({_fmt(padding_m)}, {_fmt(padding_m)}, {_fmt(padding_m)}), units.Meters)
_ggs.TopPadding = (({_fmt(padding_m)}, {_fmt(padding_m)}, {_fmt(padding_m)}), units.Meters)
'''
    else:
        grid_block = "\nsim.AddAutomaticGridSettings()\n"
    return f'''
# --- emlf: MQS simulation ---
import s4l_v1.units as units
import s4l_v1.simulation.emlf as emlf

def _ents_by_names(names):
    _all = {{getattr(_e, "Name", ""): _e for _e in model.AllEntities()}}
    return [_all[_n] for _n in names if _n in _all]

sim = emlf.MagnetoQuasiStaticSimulation()
sim.Name = "{sim_name}"

_cs_pos = sim.AddCurrentSourceSettings(_ents_by_names({positive_wires!r}))
_cs_pos.Name = "coil_pos"
_cs_pos.Amplitude = ({_fmt(current_A)}, units.Amperes)
_cs_pos.Radius = ({_fmt(wire_radius_m)}, units.Meters)
_cs_pos.IsDirectionReverted = False
{neg_block}{grid_block}
_sensor = sim.AddOverallFieldSensorSettings()
_sensor.RecordHField = True
sim.SetupSettings.Frequency = {_fmt(freq_hz)}
sim.UpdateAllMaterials()
'''


def emit_material_links(pairs: list[tuple[str, str]]) -> str:
    """材料数据库链接（B4 单元 1 已验证链路）。

    pairs: [(实体名, 库材料名)]，如 [("brain", "Brain (Grey Matter)")]。
    逐个 try/except，链接结果打印 REPORT|MATLINK|实体|σ 供编排层核验。
    注意：数据库模块是 s4l_v1.materials.database（不是 s4l_v1.database）；
    FindMaterial 返回列表；链接传材料对象。
    """
    lines = [
        "",
        "# --- emlf: material links (database) ---",
        "from s4l_v1.materials import database as _s4l_db",
        "",
        "def _link_material(ent_name, mat_name):",
        "    _ents = _ents_by_names([ent_name])",
        "    if not _ents:",
        "        print('REPORT|MATLINK|' + ent_name + '|FAIL:no_entity')",
        "        return",
        "    _mats = _s4l_db.FindMaterial(mat_name)",
        "    if not _mats:",
        "        print('REPORT|MATLINK|' + ent_name + '|FAIL:no_material')",
        "        return",
        "    try:",
        "        _ms = sim.AddMaterialSettings(_ents)",
        "        sim.LinkMaterialWithDatabase(_ms, _mats[0])",
        "        print('REPORT|MATLINK|' + ent_name + '|' + str(_ms.ElectricProps.Conductivity))",
        "    except Exception as _exc:",
        "        print('REPORT|MATLINK|' + ent_name + '|FAIL:' + str(_exc))",
    ]
    for ent, mat in pairs:
        lines.append(f'_link_material("{ent}", "{mat}")')
    return "\n".join(lines) + "\n"


def emit_solve(sim_var: str = "sim",
               voxeler_layers: list[tuple[str, int]] | None = None) -> str:
    """求解段：顺序铁律（pitfall #20），不含 SaveAs。

    前置约束：上游必须先完成 UpdateGrid 之前的设置与 document.SaveAs
    （setup.emit_save_and_report）——CreateVoxels 要求文档已存档。

    voxeler_layers: [(实体名, priority)]——同心/重叠结构必须逐实体独立
    settings + 递增优先级（内层高）；给定时不发射默认 voxeler 行，
    改用分层绑定 + kIntersectionVoxeler（pitfall #39：Topological 对脚本
    实体恒报 Voxels: None）。
    """
    if voxeler_layers:
        vox = ["", "# --- emlf: layered voxeler (intersection engine) ---",
               "import Voxeler as _Voxeler"]
        for name, prio in voxeler_layers:
            vox.append(
                f"_avs = {sim_var}.AddAutomaticVoxelerSettings(_ents_by_names(['{name}']))\n"
                f"_avs.Priority = {int(prio)}"
            )
        vox += [
            "_gvs = None",
            f"_alls = {sim_var}.AllSettings",
            "_alls = _alls() if callable(_alls) else _alls",
            "for _st in _alls:",
            "    if type(_st).__name__ == 'GlobalVoxelerSettings':",
            "        _gvs = _st",
            "        break",
            "if _gvs is not None:",
            "    _gvs.VoxelEngine = _Voxeler.GlobalVoxelerSettings.eVoxelEngine.kIntersectionVoxeler",
        ]
        voxeler_block = "\n".join(vox) + "\n"
    else:
        voxeler_block = f"{sim_var}.AddAutomaticVoxelerSettings()\n"

    return f'''
# --- emlf: solve (order is load-bearing, pitfall #20) ---
{sim_var}.UpdateGrid()
# NOTE: document.SaveAs 必须已在前面完成（emit_save_and_report），否则 CreateVoxels 报错
{voxeler_block}{sim_var}.CreateVoxels()
document.AllSimulations.Add({sim_var})
{sim_var}.WriteInputFile()  # 先落输入：headless 可核验网格/材料/体素，无需 license
{sim_var}.RunSimulation(wait=True, run_isolve_directly=True)  # Ares 路径报笼统错误，iSolve 子进程给真实错误
print("REPORT|SOLVE|HasResults|" + str({sim_var}.HasResults()))
print("REPORT|DONE")
'''
