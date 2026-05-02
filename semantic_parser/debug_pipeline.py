"""
debug_pipeline.py

Standalone script to run the semantic parser pipeline without ROS2.

Usage
─────
    python debug_pipeline.py
    python debug_pipeline.py --input /path/to/commands.txt --output results.txt
    python debug_pipeline.py --limit 20   # only first 20 commands

Output format (results.txt)
────────────────────────────
    ══════════════════════════════════════════
    [001] Go to the kitchen and find a cup.
    ──────────────────────────────────────────
    [
      {"primitive": "PTRANS", "actor": "robot", ...},
      {"primitive": "ATTEND", "actor": "robot", ...}
    ]
    ══════════════════════════════════════════

BioRobotics Laboratory, Facultad de Ingeniería, UNAM
"""

import argparse
import json
import sys
import time
from pathlib import Path

# ── Make the package importable without installing it ────────────────────────
# Adjust this path if your workspace layout differs.
_WORKSPACE = Path(__file__).resolve().parent
_SRC = _WORKSPACE / "src" / "semantic_parser"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
# ─────────────────────────────────────────────────────────────────────────────

from semantic_parser.nlp.spacy_pipeline import SpacyPipelineFactory
from semantic_parser.nlp.spacy_srl import SpacySRL
from semantic_parser.pipeline.handlers import (
    AttendNormalizationHandler,
    MappingHandler,
    QueryExpansionHandler,
    SerializationHandler,
    SRLHandler,
    load_verb_map,
)

_SPACY_MODEL = "en_core_web_trf"
_SEP_THICK = "═" * 60
_SEP_THIN = "─" * 60


def build_pipeline() -> SRLHandler:
    print(f"Loading spaCy model ({_SPACY_MODEL})...", flush=True)
    t0 = time.perf_counter()
    nlp = SpacyPipelineFactory.build(_SPACY_MODEL)
    print(f"  Model ready in {(time.perf_counter() - t0)*1000:.0f} ms", flush=True)

    srl = SpacySRL(nlp)
    verb_map = load_verb_map()

    first = SRLHandler(srl)
    (
        first.set_next(MappingHandler(verb_map))
        .set_next(QueryExpansionHandler())
        .set_next(AttendNormalizationHandler())
        .set_next(SerializationHandler())
    )
    return first


def run(input_path: Path, output_path: Path, limit: int) -> None:
    commands = [
        line.strip()
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    if limit > 0:
        commands = commands[:limit]

    total = len(commands)
    print(f"Processing {total} commands → {output_path}\n", flush=True)

    pipeline = build_pipeline()
    print()

    lines_out = []
    errors = 0
    total_ms = 0.0

    for idx, cmd in enumerate(commands, start=1):
        t0 = time.perf_counter()
        try:
            result = pipeline.handle({"command": cmd})
            cd_actions = result.get("cd_actions", [])
            output = json.dumps(cd_actions, indent=2, ensure_ascii=False)
            status = f"{len(cd_actions)} action(s)"
        except Exception as exc:
            output = f"ERROR: {exc}"
            status = "ERROR"
            errors += 1

        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_ms += elapsed_ms

        # Console progress
        print(f"[{idx:03d}/{total}] ({elapsed_ms:6.1f} ms)  {status:<20}  {cmd[:70]}")

        # Output block
        lines_out.append(_SEP_THICK)
        lines_out.append(f"[{idx:03d}] {cmd}")
        lines_out.append(_SEP_THIN)
        lines_out.append(output)
        lines_out.append("")

    lines_out.append(_SEP_THICK)
    lines_out.append(
        f"SUMMARY: {total} commands | {errors} errors | "
        f"avg {total_ms/total:.1f} ms | total {total_ms/1000:.2f} s"
    )
    lines_out.append(_SEP_THICK)

    output_path.write_text("\n".join(lines_out), encoding="utf-8")

    print(f"\nDone. {total - errors}/{total} OK | avg {total_ms/total:.1f} ms")
    print(f"Results saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the semantic parser pipeline on a list of commands."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("/home/fidel/CommandGenerator/CommandGenerator/gpsr_commands.txt"),
        help="Path to the input commands file (one command per line).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("pipeline_results.txt"),
        help="Path to the output results file.",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=0,
        help="Only process the first N commands (0 = all).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}")
        sys.exit(1)

    run(args.input, args.output, args.limit)
