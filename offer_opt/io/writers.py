"""Write a solved selection back out to CSV.

One canonical long format for all 3 cases (client_id, product, channel,
segment, base_ev, selected) -- deliberately not three different idiosyncratic
vendor-shaped exports (low's p1..p4/Offer, med's wide X1..X15, hard's
OPT-only-rows), since reproducing those per-case quirks would mean exactly
the kind of per-case special-casing the rest of this pipeline avoids. This
format carries the same information and can be re-joined against the
original offer table on (client_id, product, channel, segment).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_selection(offer_table: pd.DataFrame, selection: np.ndarray, path: str | Path) -> None:
    out = offer_table[["client_id", "product", "channel", "segment", "base_ev"]].copy()
    out["selected"] = np.asarray(selection, dtype=int)
    out.to_csv(path, index=False)
