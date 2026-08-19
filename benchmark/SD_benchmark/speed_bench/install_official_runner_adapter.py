#!/usr/bin/env python3
"""Explicitly install the SGLANG_REMOTE adapter into an official checkout.

This script only modifies the checkout passed on the command line. It copies
the adjacent ``sglang_remote.py`` and makes the explicit model import, engine
registry, max-sequence mapping, CLI, and constructor edits required by the
official runner.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ADAPTER_SOURCE = SCRIPT_DIR / "sglang_remote.py"


def replace_once(path: Path, old: str, new: str) -> bool:
    content = path.read_text(encoding="utf-8")
    if new in content:
        return False
    if old not in content:
        raise RuntimeError(f"Expected text not found in {path}:\n{old}")
    path.write_text(content.replace(old, new, 1), encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Install the explicit SGLANG_REMOTE adapter into an external NVIDIA "
            "Model Optimizer SpecDec-Bench checkout."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python SD_benchmark/speed_bench/install_official_runner_adapter.py "
            "/path/to/model-optimizer-specbench/examples/specdec_bench --check\n\n"
            "Run again without --check to modify only the supplied external checkout."
        ),
    )
    parser.add_argument("official_root", type=Path, help="Official examples/specdec_bench checkout.")
    parser.add_argument("--check", action="store_true", help="Report required changes without writing files.")
    args = parser.parse_args()

    root = args.official_root.expanduser().resolve()
    package = root / "specdec_bench"
    models = package / "models"
    run_py = root / "run.py"
    init_py = models / "__init__.py"
    if not run_py.is_file() or not init_py.is_file():
        raise FileNotFoundError(
            "Expected official layout: <root>/run.py and <root>/specdec_bench/models/__init__.py"
        )

    # Each replacement is guarded by its desired text, making the installer
    # idempotent and safe to rerun after updating this adapter.
    changes: list[str] = []
    adapter_target = models / "sglang_remote.py"
    if not adapter_target.exists() or adapter_target.read_bytes() != ADAPTER_SOURCE.read_bytes():
        changes.append(str(adapter_target))
        if not args.check:
            shutil.copyfile(ADAPTER_SOURCE, adapter_target)

    init_old = "from .sglang import SGLANGModel\n"
    init_new = "from .sglang import SGLANGModel\nfrom .sglang_remote import SGLANGRemoteModel\n"
    if init_new not in init_py.read_text(encoding="utf-8"):
        changes.append(str(init_py))
        if not args.check:
            replace_once(init_py, init_old, init_new)

    run_text = run_py.read_text(encoding="utf-8")
    registry_old = '    "SGLANG": models.SGLANGModel,\n'
    registry_new = '    "SGLANG": models.SGLANGModel,\n    "SGLANG_REMOTE": models.SGLANGRemoteModel,\n'
    if registry_new not in run_text:
        changes.append(str(run_py))
        if not args.check:
            replace_once(run_py, registry_old, registry_new)

    run_text = run_py.read_text(encoding="utf-8")
    max_key_old = '    "SGLANG": "context_length",\n'
    max_key_new = '    "SGLANG": "context_length",\n    "SGLANG_REMOTE": "context_length",\n'
    if max_key_new not in run_text:
        changes.append(str(run_py))
        if not args.check:
            replace_once(run_py, max_key_old, max_key_new)

    run_text = run_py.read_text(encoding="utf-8")
    server_arg = '''    parser.add_argument(\n        "--server_url",\n        type=str,\n        default=None,\n        help="Remote SGLang gateway base URL; required for --engine SGLANG_REMOTE.",\n    )\n'''
    anchor = '    parser.add_argument("--model_dir", type=str, required=True, help="Path to the model directory")\n'
    if server_arg not in run_text:
        changes.append(str(run_py))
        if not args.check:
            replace_once(run_py, anchor, anchor + server_arg)

    run_text = run_py.read_text(encoding="utf-8") if not args.check else run_text
    server_kwarg = '        server_url=args.server_url,\n'
    constructor_anchor = '        tokenizer_path=args.tokenizer,\n'
    if server_kwarg not in run_text:
        changes.append(str(run_py))
        if not args.check:
            replace_once(run_py, constructor_anchor, constructor_anchor + server_kwarg)

    if args.check:
        print("Would modify:" if changes else "Already installed:")
    else:
        print("Modified:" if changes else "Already installed:")
    for change in dict.fromkeys(changes):
        print(f"- {change}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
