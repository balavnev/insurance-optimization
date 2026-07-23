"""Case `low` is the odd one out: a wide/pivoted offer table (one row per
client, one column per metric x product x channel) plus a KEY=VALUE
constraint file, instead of the long/tidy tables med and hard use.

Everything here is data-driven off the column names actually present --
never a literal "OSG"/"KSK" string -- so it would keep working if low's
product/channel vocabulary changed.
"""

from __future__ import annotations

import re

import pandas as pd

from offer_opt.io.dialects import CASE_FILES, read_offers
from offer_opt.schema import RawConstraintRow

_METRIC_RE = re.compile(r"^(PREMIUM|PROBA|MARGIN|COST)_(.+)$")


def _discover_channels(columns: list[str]) -> set[str]:
    """COST_* columns are channel-only (broadcast across products) -- that's
    what makes them the reliable source for the channel vocabulary."""
    channels = set()
    for c in columns:
        m = _METRIC_RE.match(c)
        if m and m.group(1) == "COST":
            channels.add(m.group(2))
    return channels


def _split_suffix(suffix: str, channels: set[str]) -> tuple[str | None, str]:
    """Split e.g. "OSG_SMS" into (product="OSG", channel="SMS") using the
    known channel vocabulary as the anchor, so it works even if product or
    channel names themselves contain underscores."""
    if suffix in channels:
        return None, suffix
    for ch in channels:
        if suffix.endswith("_" + ch):
            return suffix[: -(len(ch) + 1)], ch
    raise ValueError(f"could not resolve column suffix {suffix!r} against channels {channels}")


def load_low_offers() -> pd.DataFrame:
    wide = read_offers("low")
    client_col = wide.columns[0]  # "subjisn"
    channels = _discover_channels(list(wide.columns))

    # metric -> {(product, channel): column_name}
    by_metric: dict[str, dict[tuple[str | None, str], str]] = {"PREMIUM": {}, "PROBA": {}, "MARGIN": {}, "COST": {}}
    for c in wide.columns:
        m = _METRIC_RE.match(c)
        if not m:
            continue
        metric, suffix = m.group(1), m.group(2)
        product, channel = _split_suffix(suffix, channels)
        by_metric[metric][(product, channel)] = c

    # Preserve the source column order (not sorted) so offer slots 1..N line
    # up with the vendor's own p1..p4 / Offer numbering convention.
    combos = [
        k for k in by_metric["PREMIUM"]
        if k in by_metric["PROBA"] and k in by_metric["MARGIN"]
    ]

    frames = []
    for offer_idx, (product, channel) in enumerate(combos, start=1):
        cost_col = by_metric["COST"].get((None, channel)) or by_metric["COST"].get((product, channel))
        frame = pd.DataFrame(
            {
                "client_id": wide[client_col].astype("int64"),
                "product": product,
                "channel": channel,
                "segment": f"{product}_{channel}",
                "score": wide[by_metric["PROBA"][(product, channel)]].astype("float64"),
                "premium": wide[by_metric["PREMIUM"][(product, channel)]].astype("float64"),
                "margin": wide[by_metric["MARGIN"][(product, channel)]].astype("float64"),
                "cost": wide[cost_col].astype("float64"),
                "_offer_slot": offer_idx,  # matches sample_out_low.csv's p1..p4 / Offer column order
            }
        )
        frames.append(frame)

    long_df = pd.concat(frames, ignore_index=True).sort_values(["client_id", "_offer_slot"]).reset_index(drop=True)
    long_df["offer_uid"] = long_df.index.astype("int64")
    return long_df, combos


def load_low_constraints() -> list[RawConstraintRow]:
    """`CNT_{PRODUCT}_{CHANNEL}=N` lines -> the same RawConstraintRow shape
    med/hard's tabular constraints produce, resolved as "offers_per_product"
    (a count cap scoped to one channel x product combo) so it flows through
    the identical, case-agnostic `constraints.resolve_one`."""
    path = CASE_FILES["low"]["constraints"]
    text = path.read_text(encoding="utf-8-sig")
    channels = _discover_channels(list(read_offers("low").columns))

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        key, _, value = line.partition("=")
        assert key.startswith("CNT_"), f"unexpected constraint key format: {line!r}"
        product, channel = _split_suffix(key[len("CNT_"):], channels)
        rows.append(
            RawConstraintRow(raw_type="offers_per_product", channel=channel, product=product,
                              min=None, max=float(value))
        )
    return rows
