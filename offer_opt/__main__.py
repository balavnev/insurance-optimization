"""Command-line entrypoint: `python -m offer_opt ...`.

Runs either the constraint parser alone (`--parse-only`: schema resolution +
constraint resolution, no solving -- prints every resolved ConstraintSpec/
ParameterSpec) or the full pipeline (parse + solve + verify + codegen
cross-check), against either a known case (`--case low/med/hard`) or an
arbitrary dataset (`--offers PATH --constraints PATH`).
"""

from __future__ import annotations

import argparse
import sys


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m offer_opt",
        description="Parse (and, optionally, solve/verify) a marketing-campaign offer/constraint dataset.",
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--case", choices=["low", "med", "hard"], help="one of the 3 known example cases")
    source.add_argument("--offers", type=str, help="path to an arbitrary offers file (use with --constraints)")

    p.add_argument("--constraints", type=str, help="path to an arbitrary constraints file (required with --offers)")
    p.add_argument("--llm-url", type=str, default=None,
                   help="OpenAI-compatible base URL (defaults to $LLM_BASE_URL if unset)")
    p.add_argument("--llm-key", type=str, default=None,
                   help="API key for --llm-url (defaults to $LLM_API_KEY if unset)")
    p.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="auto")
    p.add_argument("--max-iters", type=int, default=300)
    p.add_argument("--repair-every", type=int, default=10)
    p.add_argument("--parse-only", action="store_true",
                   help="run schema resolution + constraint parsing only; skip solving entirely")
    return p


def _resolve_device(name: str):
    import torch

    if name == "auto":
        from offer_opt.device import get_device
        return get_device(prefer_gpu=True)
    return torch.device(name)


def _resolve_llm_client(llm_url: str | None, llm_key: str | None):
    from offer_opt.llm.client import LLMUnavailable, VLLMOpenAIClient

    try:
        client = VLLMOpenAIClient(base_url=llm_url, api_key=llm_key)
    except LLMUnavailable:
        print("[llm] no endpoint configured (--llm-url or $LLM_BASE_URL) -- running symbolic-only", file=sys.stderr)
        return None
    healthy = client.health_check()
    print(f"[llm] endpoint {client.base_url!r} (healthy: {healthy})", file=sys.stderr)
    return client


def _paths_for(args) -> tuple[str, str]:
    if args.case:
        from offer_opt.io.dialects import CASE_FILES
        return str(CASE_FILES[args.case]["offers"]), str(CASE_FILES[args.case]["constraints"])
    if not args.constraints:
        raise SystemExit("--constraints is required when --offers is given")
    return args.offers, args.constraints


def _run_parse_only(args, llm_client) -> None:
    from offer_opt import features

    offers_path, constraints_path = _paths_for(args)
    offer_table, constraint_set, dim_names = features.load_from_paths(
        offers_path, constraints_path, llm_client=llm_client)

    print(f"discovered dimensions: {dim_names}")
    print(f"offer rows: {len(offer_table)}")
    print(f"resolved {len(constraint_set.constraints)} constraint(s), "
          f"{len(constraint_set.parameters)} parameter(s):\n")
    for c in constraint_set.constraints:
        print(f"  [constraint] {c.id}\n"
              f"      raw_type={c.raw_type!r} scope={c.scope} measure={c.measure} "
              f"min={c.min} max={c.max} per_client={c.per_client}")
    for p in constraint_set.parameters:
        print(f"  [parameter]  kind={p.kind} scope={p.scope} value={p.value}")


def _run_full_pipeline(args, device, llm_client) -> None:
    if args.case:
        from offer_opt.pipeline import run_case
        result = run_case(args.case, device, max_iters=args.max_iters, repair_every=args.repair_every)
        print(result.verification)
        print(f"reference EV (vendor's own solution): {result.reference_ev:,.2f}")
        return

    from offer_opt.pipeline import run_dataset
    offers_path, constraints_path = _paths_for(args)
    result = run_dataset(offers_path, constraints_path, device, llm_client=llm_client,
                          max_iters=args.max_iters, repair_every=args.repair_every)

    print(f"discovered dimensions: {result.dims}")
    if result.conflicts:
        print(f"{len(result.conflicts)} constraint conflict(s) detected (deeper constraint wins, "
              f"see system_design_overview.md Section 3):")
        for c in result.conflicts:
            print(f"  - {c.reason}")
    print(result.verification)
    print(f"generated verifier code agrees with verify.py: {result.codegen_agrees}")


def main(argv: list[str] | None = None) -> None:
    from offer_opt.codegen.generate import CodegenError
    from offer_opt.constraints import UnresolvedConstraintError

    args = build_arg_parser().parse_args(argv)
    if args.offers and not args.constraints:
        build_arg_parser().error("--constraints is required when --offers is given")

    device = _resolve_device(args.device)
    print(f"[device] {device}", file=sys.stderr)
    llm_client = _resolve_llm_client(args.llm_url, args.llm_key)

    try:
        if args.parse_only:
            _run_parse_only(args, llm_client)
        else:
            _run_full_pipeline(args, device, llm_client)
    except UnresolvedConstraintError as exc:
        print(f"\n[unresolved] {exc}\n"
              f"(pass --llm-url, or set $LLM_BASE_URL, so the LLM fallback can classify it)", file=sys.stderr)
        raise SystemExit(1)
    except CodegenError as exc:
        print(f"\n[codegen] {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
