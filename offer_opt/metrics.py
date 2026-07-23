"""Reference-solution reconstruction (for validating the parser/verifier
independently of the solver) and the CPU/GPU benchmark harness."""

from __future__ import annotations

import numpy as np
import pandas as pd

from offer_opt.io.dialects import read_reference
from offer_opt import features as _features
from offer_opt import verify as _verify


def reference_selection_low(offer_table: pd.DataFrame) -> np.ndarray:
    """sample_out_low.csv has one categorical `Offer` column: 0=none,
    1..4=which of the p1..p4 slots (in source-column order) was chosen."""
    ref = read_reference("low")
    ref = ref.rename(columns={ref.columns[0]: "client_id"})
    ref["client_id"] = ref["client_id"].astype("int64")
    offer_by_client = ref.set_index("client_id")["Offer"].astype("int64")

    chosen_slot = offer_table["client_id"].map(offer_by_client)
    return (offer_table["_offer_slot"].to_numpy() == chosen_slot.to_numpy()).astype("float64")


def reference_selection_med(offer_table: pd.DataFrame) -> np.ndarray:
    """sample_out_med.csv is wide: one X1..X15 column per (channel, segment)
    combo, named e.g. "EMAIL_IFL_AA". Reconstruct via a (client_id, channel,
    segment) -> selected map, then align to the canonical long OfferTable."""
    ref = read_reference("med")
    id_col = ref.columns[1]  # "SUBJISN" (col 0 is the anonymous index)
    channels = sorted(offer_table["channel"].unique(), key=len, reverse=True)

    x_cols = [c for c in ref.columns if c.startswith("X") and c[1:].isdigit()]
    combo_cols = [c for c in ref.columns if c not in (ref.columns[0], id_col) and c not in x_cols]

    selected = pd.Series(0, index=pd.MultiIndex.from_arrays([[], [], []], names=["client_id", "channel", "segment"]), dtype="int64")
    parts = []
    for col, x_col in zip(combo_cols, x_cols):
        channel = next(ch for ch in channels if col.startswith(ch + "_"))
        segment = col[len(channel) + 1:]
        flag = (ref[x_col].astype("float64") > 0.5).astype("int64")
        parts.append(pd.DataFrame({
            "client_id": ref[id_col].astype("int64"),
            "channel": channel,
            "segment": segment,
            "selected": flag,
        }))
    long_ref = pd.concat(parts, ignore_index=True)
    long_ref = long_ref[long_ref["selected"] == 1]
    key = pd.MultiIndex.from_frame(long_ref[["client_id", "channel", "segment"]])
    selected_set = set(key)

    ot_key = list(zip(offer_table["client_id"], offer_table["channel"], offer_table["segment"]))
    return np.array([1.0 if k in selected_set else 0.0 for k in ot_key])


def reference_selection_hard(offer_table: pd.DataFrame) -> np.ndarray:
    """sample_out_hard.csv only lists rows with OPT=1 (verified: every row
    in the file has OPT=1) -- reconstruct the full 0/1 vector by treating
    any (client_id, product, channel, segment) present in the file as
    selected, everything else as not."""
    ref = read_reference("hard")
    selected_set = set(zip(
        ref["SUBJISN"].astype("int64"),
        ref["PRODUCT"].astype(str),
        ref["CHANNEL"].astype(str),
        ref["SEGMENT"].astype(str),
    ))
    ot_key = list(zip(
        offer_table["client_id"], offer_table["product"], offer_table["channel"], offer_table["segment"],
    ))
    return np.array([1.0 if k in selected_set else 0.0 for k in ot_key])


_REFERENCE_LOADERS = {
    "low": lambda ot: reference_selection_low(ot),
    "med": lambda ot: reference_selection_med(ot),
    "hard": lambda ot: reference_selection_hard(ot),
}


def load_reference(case: str, offer_table: pd.DataFrame) -> np.ndarray:
    return _REFERENCE_LOADERS[case](offer_table)


def verify_reference(case: str):
    """Reconstruct the vendor's own reference solution for `case` and run it
    through our generic verifier -- validates the constraint parser (and the
    verifier itself) independently of whether our solver works yet."""
    offer_table, constraint_set = _features.load_case(case)
    offer_table, _n_clients = _features.encode_dims(offer_table)
    selection = load_reference(case, offer_table)
    return _verify.verify(offer_table, constraint_set, selection)


def benchmark(case: str, device, n_reps: int = 5, **solve_kwargs) -> dict:
    """Median-of-`n_reps` wall-clock timing for a full solve() on `case` and
    `device` -- median avoids one-off warmup/allocator noise skewing the
    fixed-hardware comparison the grading rubric benchmarks against."""
    import time

    from offer_opt.device import synchronize
    from offer_opt.solver.lagrangian import solve as _solve

    offer_table, constraint_set = _features.load_case(case)
    offer_table, _n_clients = _features.encode_dims(offer_table)

    times = []
    last_result = None
    for _ in range(n_reps):
        synchronize(device)
        t0 = time.perf_counter()
        last_result = _solve(offer_table, constraint_set, device, **solve_kwargs)
        synchronize(device)
        times.append(time.perf_counter() - t0)

    times_sorted = sorted(times)
    median_time = times_sorted[len(times_sorted) // 2]
    report = _verify.verify(offer_table, constraint_set, last_result.selection)

    return dict(case=case, device=str(device), n_reps=n_reps, times=times, median_time=median_time,
                iterations=last_result.iterations, converged=last_result.converged,
                total_ev=last_result.total_ev, verifier_ok=report.ok,
                n_clients=int(offer_table["client_idx"].max()) + 1, n_offers=len(offer_table))
