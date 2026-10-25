"""Acceptance evidence for the fixed teaching examples, without changing energies."""

from __future__ import annotations

import math

from q2m3.constants import HARTREE_TO_KCAL_MOL

SUBSYSTEM_LIMIT_HA = 0.0005
INTERACTION_LIMIT_KCAL_MOL = 0.25
VARIATIONAL_TOLERANCE_HA = 1e-6


def _difference(value, reference):
    if value is None or reference is None:
        return None
    if not (math.isfinite(value) and math.isfinite(reference)):
        return None
    return float(value - reference)


def assess_accuracy(result) -> dict:
    """Require exact active-space references and bounded errors in every subsystem.

    These thresholds apply only to these small examples. They do not certify
    basis convergence, the active-space selection, or the total molecular energy.
    """
    records = {}
    for name in ("AB", "A", "B"):
        sub = result.subsystems[name]
        error = _difference(sub.e_sqd, sub.e_casci)
        sqd = sub.sqd
        tier = None if sqd is None else sqd.baseline_tier
        slots = None if sqd is None else sqd.n_reps
        required_slots = None
        if sqd is not None:
            electrons, orbitals = sqd.active_space
            occupied = electrons // 2
            # Each eigenvector of the occupied-virtual doubles matrix produces
            # at most two UCJ factors in ffsim's real-amplitude decomposition.
            required_slots = 2 * occupied * (orbitals - occupied)
        passed = (
            tier == "T0"
            and not sqd.baseline_untrustworthy
            and slots is not None
            and slots >= required_slots
            and error is not None
            and -VARIATIONAL_TOLERANCE_HA <= error <= SUBSYSTEM_LIMIT_HA
        )
        records[name] = dict(
            reference_tier=tier,
            error_mha=None if error is None else 1000 * error,
            ucj_factor_slots=slots,
            required_factor_slots=required_slots,
            passed=bool(passed),
        )
    delta = _difference(result.delta_e["sqd"], result.delta_e["casci"])
    delta_kcal = None if delta is None else delta * HARTREE_TO_KCAL_MOL
    return dict(
        passed=all(row["passed"] for row in records.values())
        and delta_kcal is not None
        and abs(delta_kcal) <= INTERACTION_LIMIT_KCAL_MOL,
        subsystem_limit_mha=1000 * SUBSYSTEM_LIMIT_HA,
        variational_tolerance_ha=VARIATIONAL_TOLERANCE_HA,
        interaction_limit_kcal_mol=INTERACTION_LIMIT_KCAL_MOL,
        interaction_error_kcal_mol=delta_kcal,
        subsystems=records,
    )


def accuracy_markdown(fragment_id: str, evidence: dict) -> str:
    """Display acceptance beside the unchanged library report."""
    status = "PASS" if evidence["passed"] else "FAIL"
    lines = [
        f"\n## SQD 精度验收：{fragment_id} — {status}\n",
        "要求：各子系统均有可信 T0 参考，SQD 误差不超过 0.5 mHa。",
        "相互作用能误差的绝对值不得超过 0.25 kcal/mol。",
        "变分下界允许 1e-6 Ha 数值容差。",
        "这些限值仅用于当前活性空间，不代表完整体系的精度。\n",
        "| 子系统 | 参考层级 | SQD − CASCI (mHa) | UCJ 项数槽位 | 通过 |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for name, row in evidence["subsystems"].items():
        error = row["error_mha"]
        number = "缺失" if error is None else f"{error:.6f}"
        lines.append(
            f"| {name} | {row['reference_tier']} | {number} | "
            f"{row['ucj_factor_slots']} | {row['passed']} |"
        )
    delta = evidence["interaction_error_kcal_mol"]
    number = "缺失" if delta is None else f"{delta:.6f} kcal/mol"
    lines.append(f"\n相互作用能：SQD − CASCI = {number}。\n")
    return "\n".join(lines)
