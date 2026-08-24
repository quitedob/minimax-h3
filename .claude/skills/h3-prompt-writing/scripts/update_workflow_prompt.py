#!/usr/bin/env python3
"""Read or replace the prompt inside a MiniMax H3 ComfyUI workflow JSON.

MiniMax H3 has no dedicated "Text to Video" node: text-to-video is done with
``MiniMaxH3ImageToVideo`` while leaving ``first_frame`` / ``last_frame`` empty.

The prompt lives in different places depending on the workflow shape:

* **Subgraph templates** (``t2v``, ``i2v``): the prompt is exposed on the
  subgraph-instance node (``widgets_values[0]``) and mirrored in the inner
  ``MiniMaxH3ImageToVideo`` node's ``widgets_values[0]``.
* **Flattened templates** (``r2v``): the prompt lives in a
  ``PrimitiveStringMultiline`` node that links into ``MiniMaxH3ReferenceToVideo``.

``set`` writes every prompt-bearing slot so the workflow stays consistent.

Stdlib only; always reads/writes UTF-8 (Windows-safe).
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
_WORKFLOW_DIR = (
    REPO_ROOT
    / "ComfyUI_windows_portable"
    / "ComfyUI"
    / "user"
    / "default"
    / "workflows"
)

PRESETS = {
    "t2v": _WORKFLOW_DIR / "video_minimax_h3_t2v.json",
    "i2v": _WORKFLOW_DIR / "video_minimax_h3_i2v.json",
    "r2v": _WORKFLOW_DIR / "video_minimax_h3_r2v.json",
}

DEFAULT_WORKFLOW = PRESETS["t2v"]


def _is_video_node(node_type: str) -> bool:
    return (
        isinstance(node_type, str)
        and "MiniMaxH3" in node_type
        and "ToVideo" in node_type
    )


def _is_prompt_slot(widgets) -> bool:
    return bool(widgets) and isinstance(widgets[0], str)


def _targets(data):
    """Yield (node, widgets_list) for every prompt-bearing widget found."""
    subgraphs = data.get("definitions", {}).get("subgraphs") or []
    if subgraphs:
        subgraph_ids = {s["id"] for s in subgraphs}
        # Exposed prompt on the subgraph-instance node at the top level.
        for node in data.get("nodes", []):
            if node.get("type") in subgraph_ids and _is_prompt_slot(node.get("widgets_values")):
                yield node, node["widgets_values"]
        # Inner MiniMaxH3*ToVideo defaults.
        for sub in subgraphs:
            for node in sub.get("nodes", []):
                if _is_video_node(node.get("type", "")) and _is_prompt_slot(node.get("widgets_values")):
                    yield node, node["widgets_values"]
    else:
        # Flattened workflow (r2v): the prompt is held in a multiline string
        # node; otherwise fall back to a MiniMaxH3*ToVideo node's own widget.
        string_nodes = [
            n for n in data.get("nodes", [])
            if n.get("type") == "PrimitiveStringMultiline"
            and _is_prompt_slot(n.get("widgets_values"))
        ]
        if string_nodes:
            for node in string_nodes:
                yield node, node["widgets_values"]
        else:
            for node in data.get("nodes", []):
                if _is_video_node(node.get("type", "")) and _is_prompt_slot(node.get("widgets_values")):
                    yield node, node["widgets_values"]


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path: Path, data):
    # ensure_ascii=True keeps the file pure ASCII (non-ASCII becomes \uXXXX),
    # so no encoding can ever misread it — Windows cp936/gbk included.
    path.write_text(
        json.dumps(data, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def cmd_get(path: Path) -> int:
    data = _load(path)
    prompts = [w[0] for _, w in _targets(data)]
    if not prompts:
        print("No prompt-bearing node found.", file=sys.stderr)
        return 1
    # All prompt slots should hold the same text; print the exposed one.
    print(prompts[0])
    return 0


def cmd_set(path: Path, prompt: str) -> int:
    data = _load(path)
    changed = 0
    for node, widgets in _targets(data):
        if widgets[0] != prompt:
            widgets[0] = prompt
            changed += 1
    if changed == 0:
        print("Prompt already matches; nothing changed.", file=sys.stderr)
    _save(path, data)
    print(f"Updated {changed} prompt slot(s) in {path.name}")
    return 0


def _read_prompt(args) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    if args.prompt is not None:
        return args.prompt
    if not sys.stdin.isatty():
        return sys.stdin.read()
    print("No prompt provided (use --prompt-file, --prompt, or stdin).",
          file=sys.stderr)
    sys.exit(2)


def _resolve_workflow(value: str) -> Path:
    if value in PRESETS:
        return PRESETS[value]
    return Path(value)


def main() -> int:
    # Force UTF-8 so printing prompts with non-ASCII or U+2028/U+2029 chars does
    # not crash on Windows consoles that default to cp936/gbk.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("get", "set"),
        help="'get' prints the current prompt; 'set' replaces it.",
    )
    parser.add_argument(
        "--workflow",
        default=str(DEFAULT_WORKFLOW),
        help=(
            "Workflow to edit: a preset name (t2v | i2v | r2v) or a path. "
            f"Default: t2v ({DEFAULT_WORKFLOW})"
        ),
    )
    parser.add_argument("--prompt", help="Prompt text for 'set' (short prompts only).")
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="File containing the prompt text for 'set' (recommended for multiline).",
    )
    args = parser.parse_args()

    path = _resolve_workflow(args.workflow)
    if not path.exists():
        print(f"Workflow not found: {path}", file=sys.stderr)
        return 1

    if args.command == "get":
        return cmd_get(path)
    return cmd_set(path, _read_prompt(args))


if __name__ == "__main__":
    sys.exit(main())
