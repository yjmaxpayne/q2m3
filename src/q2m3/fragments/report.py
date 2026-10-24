"""Markdown report of fragment interaction-energy results.

The report is a derived view: every number in it comes from a
:class:`~q2m3.fragments.result.FragmentInteractionResult`. An imported Ks table is only
shown next to the results; no statistic is computed from it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from q2m3.constants import HARTREE_TO_KCAL_MOL
from q2m3.fragments.result import (
    COUNTERPOISE_SUBSYSTEMS,
    ENERGY_LAYERS,
    ERROR_TERMS,
    FragmentInteractionResult,
)

FIXED_STATEMENT = "q2m3 不对 Ks 做统计结论。"
BANNED_TERMS = ("量子优势", "quantum advantage")

TITLE = "# 片段相互作用能报告"
LAYER_NAMES = {"hf": "HF", "mp2": "MP2", "casci": "CASCI", "sqd": "SQD", "ccsd_t": "CCSD(T)"}
ERROR_NAMES = {
    "algorithm": "算法误差（SQD − CASCI）",
    "active_space": "活性空间误差（CASCI − 同基组全空间参考）",
    "basis": "基组误差（同基组全空间参考 − 外部参考）",
    "model": "模型误差（活性空间 + 基组）",
}
QPE_HEADING = "### 双因子分解成本估计（QPE 资源）"
SQD_HEADING = "### SQD 工作量与内存预估"
KS_HEADING = "## 实验 Ks 值（只并列显示）"


def render_report(
    results: Sequence[FragmentInteractionResult], *, ks_table: Mapping[str, float] | None = None
) -> str:
    """Return one Markdown report for all fragment results of one call.

    Args:
        results: Fragment results to report, in report order.
        ks_table: Experimental labels mapped to imported Ks values. They are listed
            as given.

    Raises:
        ValueError: The report contains a term of ``BANNED_TERMS``.
    """
    sections = [TITLE, FIXED_STATEMENT]
    for result in results:
        sections += [_energy_section(result), _error_section(result), _resource_section(result)]
    if ks_table is not None:
        rows = [f"| {label} | {ks:g} |" for label, ks in ks_table.items()]
        sections.append("\n".join([KS_HEADING, "", "| 实验标签 | Ks |", "|---|---|", *rows]))
    text = "\n\n".join(sections) + "\n"
    folded = text.casefold()
    for term in BANNED_TERMS:
        if term.casefold() in folded:
            raise ValueError(f"report contains banned term {term!r}")
    return text


def _energy_section(result: FragmentInteractionResult) -> str:
    rows = [
        _row(LAYER_NAMES[layer], result.delta_e.get(layer), result.null_reasons, f"delta_e.{layer}")
        for layer in ENERGY_LAYERS
    ]
    return "\n".join(
        [
            f"## 片段 {result.fragment_id}（模式 {result.mode}，基组 {result.basis}）",
            "",
            "### 相互作用能 ΔE（counterpoise 校正）",
            "",
            "| 层 | ΔE (Ha) | ΔE (kcal/mol) | 空值原因 |",
            "|---|---|---|---|",
            *rows,
        ]
    )


def _error_section(result: FragmentInteractionResult) -> str:
    rows = []
    for term in ERROR_TERMS:
        value = getattr(result.errors, term)
        if value is None:
            reason = result.null_reasons.get(f"errors.{term}", "")
            rows.append(f"| {ERROR_NAMES[term]} | — | — | — | — | {reason} |")
        else:
            rows.append(
                f"| {ERROR_NAMES[term]} | {value:.6f} | {abs(value):.6f} "
                f"| {value * HARTREE_TO_KCAL_MOL:.3f} | {abs(value) * HARTREE_TO_KCAL_MOL:.3f} | |"
            )
    source = result.provenance.get("same_basis_reference_source")
    return "\n".join(
        [
            "### 误差分解",
            "",
            f"同基组全空间参考来源：{source}",
            "",
            "| 误差项 | 有符号 (Ha) | 绝对值 (Ha) | 有符号 (kcal/mol) | 绝对值 (kcal/mol) | 空值原因 |",
            "|---|---|---|---|---|---|",
            *rows,
        ]
    )


def _resource_section(result: FragmentInteractionResult) -> str:
    qpe_rows, sqd_rows = [], []
    for name in COUNTERPOISE_SUBSYSTEMS:
        sub = result.subsystems[name]
        qpe, sqd = sub.resources.get("qpe"), sub.resources.get("sqd")
        if qpe is None:
            qpe_rows.append(f"| {name} | — | — | — | {sub.null_reasons.get('resources.qpe', '')} |")
        else:
            v = qpe.values
            qpe_rows.append(
                f"| {name} | {v['hamiltonian_1norm']:.3f} | {v['logical_qubits']} "
                f"| {v['toffoli_gates']} | |"
            )
        if sqd is None:
            sqd_rows.append(
                f"| {name} | — | — | — | — | {sub.null_reasons.get('resources.sqd', '')} |"
            )
        else:
            v, rss = sqd.values, sqd.values["rss_mb"]
            sqd_rows.append(
                f"| {name} | {v['qubits']} | {v['full_ci_dim']} "
                f"| {rss['prepare']:.1f} | {rss['sample']:.1f} | |"
            )
    return "\n".join(
        [
            QPE_HEADING,
            "",
            "| 子体系 | λ (Ha) | 逻辑比特 | Toffoli 门数 | 空值原因 |",
            "|---|---|---|---|---|",
            *qpe_rows,
            "",
            SQD_HEADING,
            "",
            "| 子体系 | 量子比特 | 全 CI 维数 | prepare RSS (MB) | sample RSS (MB) | 空值原因 |",
            "|---|---|---|---|---|---|",
            *sqd_rows,
        ]
    )


def _row(label: str, value: float | None, reasons: Mapping[str, str], key: str) -> str:
    if value is None:
        return f"| {label} | — | — | {reasons.get(key, '')} |"
    return f"| {label} | {value:.6f} | {value * HARTREE_TO_KCAL_MOL:.3f} | |"
