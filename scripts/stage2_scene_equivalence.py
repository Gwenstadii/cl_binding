"""Verify two canonical scene XMLs differ only by the repository root in asset paths, and record the evidence.

A git worktree that reaches the shared venv through a symlink renders robosuite mesh paths under its own root,
so the scene-XML hash changes while the physical scene does not. This check is the only accepted basis for a
scene-hash alias in stage2_compare.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main(args):
    a, b = args.xml_a.read_text(), args.xml_b.read_text()
    norm_a, norm_b = a.replace(args.root_a, "<ROOT>"), b.replace(args.root_b, "<ROOT>")
    changed = [line for line in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0)
               if line.startswith(("-", "+")) and not line.startswith(("---", "+++"))]
    report = {"xml_a": str(args.xml_a), "xml_b": str(args.xml_b), "root_a": args.root_a, "root_b": args.root_b,
              "sha256_a": digest(a), "sha256_b": digest(b),
              "identical_after_root_normalization": norm_a == norm_b, "normalized_sha256": digest(norm_a),
              "changed_lines_each_side": len(changed) // 2,
              "every_changed_line_mentions_a_root": all(args.root_a in line or args.root_b in line for line in changed),
              "examples": changed[:2]}
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("sha256_a", "sha256_b", "identical_after_root_normalization",
                                             "changed_lines_each_side", "every_changed_line_mentions_a_root")}))
    if not (report["identical_after_root_normalization"] and report["every_changed_line_mentions_a_root"]):
        raise SystemExit("Scenes differ beyond the repository root; no alias allowed")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--xml-a", type=Path, required=True)
    p.add_argument("--root-a", required=True)
    p.add_argument("--xml-b", type=Path, required=True)
    p.add_argument("--root-b", required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
