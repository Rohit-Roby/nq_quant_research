"""Command-line entry point: ``nqqr <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import date
from pathlib import Path

from quant_research.config import load_settings
from quant_research.pipeline import Pipeline

STAGES = ("ingest", "validate", "gold", "backtest", "all")


def _date(s: str) -> date:
    return date.fromisoformat(s)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nqqr", description="NQ quant research pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run one stage or the whole pipeline")
    run.add_argument("stage", choices=STAGES)
    run.add_argument("--start", type=_date)
    run.add_argument("--end", type=_date)
    run.add_argument("--source", choices=["synthetic", "csv"])
    run.add_argument("--csv-path")
    run.add_argument("--lake-root")
    run.add_argument("--config-dir")
    run.add_argument("--no-dq-gate", action="store_true", help="do not fail on DQ breaches")

    sub.add_parser("dashboard", help="launch the Streamlit dashboard")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "dashboard":
        app = Path(__file__).resolve().parents[2] / "dashboard" / "app.py"
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])

    settings = load_settings(
        args.config_dir, source=args.source, csv_path=args.csv_path, lake_root=args.lake_root
    )
    pipe = Pipeline(settings)
    start, end = args.start, args.end
    if args.stage == "all":
        results = [
            pipe.ingest(start, end),
            pipe.validate(start, end, enforce_gate=not args.no_dq_gate),
            pipe.build_gold(start, end),
            pipe.backtest(start, end),
        ]
    elif args.stage == "validate":
        results = [pipe.validate(start, end, enforce_gate=not args.no_dq_gate)]
    else:
        stage = {"ingest": pipe.ingest, "gold": pipe.build_gold, "backtest": pipe.backtest}
        results = [stage[args.stage](start, end)]
    for r in results:
        print(json.dumps(r.to_dict(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
