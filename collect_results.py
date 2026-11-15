"""Collect every experiment artifact needed for the paper into one folder.

Usage
-----
    python collect_results.py                     # -> data/paper_results/<timestamp>/
    python collect_results.py --with-ckpts        # also copy .pt checkpoints (~235 MB each)
    python collect_results.py --out my_results    # custom output directory
    python collect_results.py --extra-dir ~/afpsm_logs   # console logs / calibration files

What is collected (read-only; nothing in the repo is modified or deleted)
------------------------------------------------------------------------
  runs/<run_name>/train.log      per-log-step metrics of every training run under logs/
  runs/<run_name>/config.py      the config snapshot written by train.py
  runs/<run_name>/images/        sample grids saved during training
  runs/<run_name>/ckpts/         checkpoints, only with --with-ckpts
  fid_results.jsonl              all FID rows collected so far
  fid_table.md                   paper-ready FID table (NFE column, deduplicated)
  eval/<dir>/                    top-level eval_* sample grids
  console/                       *.out console logs + extracted [override] lines
  calibration/                   real-data statistics (e.g. real_mnist_class_std.txt)
  MANIFEST.md                    run summary + FID table + file inventory + git commit

The output directory defaults to <repo>/data/paper_results/..., which is git-ignored,
so the snapshot can be large without polluting the repository.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METRIC_RE = re.compile(r"([A-Za-z_]+):\s*([-\d.eE+]+)")
DEFAULT_EXTRA = Path.home() / "afpsm_logs"


# --------------------------------------------------------------- log parsing
def parse_train_log(path):
    """Return (first_block, last_block) dicts of metrics, one block per log_step."""
    first, last, cur = {}, {}, {}
    for line in open(path, errors="ignore"):
        if line.startswith("Step:"):
            cur = {k: v for k, v in METRIC_RE.findall(line)}
        elif line.startswith("Loss:"):
            cur.update({k: v for k, v in METRIC_RE.findall(line)})
            if not first:
                first = dict(cur)
            last = dict(cur)
    return first, last


def fmt(d, key, spec=".4f"):
    try:
        return format(float(d.get(key, "nan")), spec)
    except (TypeError, ValueError):
        return "-"


# ------------------------------------------------------------------- FID table
def parse_mode(mode):
    s = str(mode).strip().lower()
    if s == "adaptive":
        return 2, True
    if s.endswith("a"):
        return int(s[:-1]), True
    return int(s), False


def row_nfe(row):
    if row.get("nfe") is not None:
        return int(row["nfe"])
    steps, adaptive = parse_mode(row.get("mode", "1"))
    if adaptive and "AFPSM" in str(row.get("sampler", "")):
        return steps * 3
    return steps


def load_fid(path):
    rows = []
    if not path.exists():
        return rows
    for line in open(path, errors="ignore"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def render_fid_table(rows):
    """Markdown table, deduplicated: last row wins per (name, mode, NFE)."""
    for r in rows:
        r["nfe"] = row_nfe(r)
    uniq = {}
    for r in rows:
        uniq[(r.get("name", "?"), str(r.get("mode", "?")), r["nfe"])] = r
    rows = sorted(uniq.values(),
                  key=lambda r: (str(r.get("name", "?")), r["nfe"], str(r.get("mode"))))
    lines = ["| model | mode | NFE | n | FID |", "|---|---|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r.get('name', '?')} | {r.get('mode', '?')} | {r['nfe']} | "
                     f"{r.get('n', '?')} | {float(r['fid']):.2f} |")
    return lines


# ---------------------------------------------------------------------- helper
def copy_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.stat().st_size


def copy_dir_files(src_dir, dst_dir, suffix=None):
    """Copy (non-recursively) files of src_dir into dst_dir; returns (count, bytes)."""
    count = size = 0
    if not src_dir.exists():
        return 0, 0
    for f in sorted(src_dir.iterdir()):
        if not f.is_file():
            continue
        if suffix is not None and f.suffix != suffix:
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        size += copy_file(f, dst_dir / f.name)
        count += 1
    return count, size


def git_commit():
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--with-ckpts", action="store_true", help="also copy *.pt checkpoints")
    ap.add_argument("--extra-dir", default=str(DEFAULT_EXTRA),
                    help="directory with console *.out logs and calibration files")
    ap.add_argument("--no-console", action="store_true", help="skip the extra-dir scan")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out).resolve() if args.out else ROOT / "data" / "paper_results" / stamp
    out.mkdir(parents=True, exist_ok=True)
    extra = Path(args.extra_dir).expanduser() if not args.no_console else None

    inventory = []          # (relative path, bytes)
    run_rows = []           # (run, first, last)
    ckpt_files = ckpt_bytes = 0

    # ---- 1) training runs -------------------------------------------------
    logs_dir = ROOT / "logs"
    for run_dir in sorted(p for p in logs_dir.glob("*") if p.is_dir()):
        log = run_dir / "train.log"
        if not log.exists():
            continue
        first, last = parse_train_log(log)
        dst = out / "runs" / run_dir.name
        inventory.append((str(dst.relative_to(out) / "train.log"),
                          copy_file(log, dst / "train.log")))
        if (run_dir / "config.py").exists():
            inventory.append((str(dst.relative_to(out) / "config.py"),
                              copy_file(run_dir / "config.py", dst / "config.py")))
        n_img, b_img = copy_dir_files(run_dir / "images", dst / "images", suffix=".png")
        if n_img:
            inventory.append((f"runs/{run_dir.name}/images/ ({n_img} png)", b_img))
        if args.with_ckpts:
            for ck in sorted((run_dir / "ckpts").glob("*.pt")):
                inventory.append((f"runs/{run_dir.name}/ckpts/{ck.name}",
                                  copy_file(ck, dst / "ckpts" / ck.name)))
                ckpt_files += 1
                ckpt_bytes += ck.stat().st_size
        run_rows.append((run_dir.name, first, last, n_img))

    # ---- 2) FID results ---------------------------------------------------
    fid_src = ROOT / "fid_results.jsonl"
    fid_rows = load_fid(fid_src)
    if fid_src.exists():
        inventory.append(("fid_results.jsonl", copy_file(fid_src, out / "fid_results.jsonl")))
    fid_lines = render_fid_table(fid_rows)

    # ---- 3) evaluation sample grids --------------------------------------
    for ev in sorted(p for p in ROOT.glob("eval*") if p.is_dir()):
        n, b = copy_dir_files(ev, out / "eval" / ev.name, suffix=".png")
        if n:
            inventory.append((f"eval/{ev.name}/ ({n} png)", b))

    # ---- 4) console logs + [override] lines -------------------------------
    override_lines = []
    if extra is not None and extra.exists():
        for f in sorted(extra.glob("*.out")):
            try:
                head = f.read_text(errors="ignore").splitlines()[:80]
            except OSError:
                continue
            for line in head:
                if "[override]" in line or "[data]" in line or "[seed]" in line:
                    override_lines.append(f"{f.name}: {line.strip()}")
            copy_file(f, out / "console" / f.name)
        for cal in sorted(extra.glob("*.txt")):
            copy_file(cal, out / "calibration" / cal.name)
        if override_lines:
            (out / "console" / "overrides.txt").write_text(
                "\n".join(override_lines) + "\n", encoding="utf-8")
            inventory.append(("console/overrides.txt", 0))

    # ---- 5) MANIFEST ------------------------------------------------------
    total_bytes = sum(b for _, b in inventory)
    lines = [
        "# AF-PSM paper results snapshot",
        "",
        f"- generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- repo: `{ROOT}`",
        f"- git commit: `{git_commit()}`",
        f"- python: {sys.version.split()[0]}",
        f"- checkpoint copies: {ckpt_files} files, {ckpt_bytes / 2**20:.1f} MiB"
        + ("" if args.with_ckpts else "  (use --with-ckpts to include)"),
        "",
        "## Training runs",
        "",
        "| run | steps | loss | grad_norm | step_time | mem_peak | sample grids |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, first, last, n_img in run_rows:
        lines.append(
            f"| {name} | {first.get('Step', last.get('Step', '?'))} -> {last.get('Step', '?')} "
            f"| {fmt(last, 'Loss')} | {fmt(last, 'Grad_Norm')} "
            f"| {fmt(last, 'Step_time', '.3f')} | {fmt(last, 'Mem_peak', '.2f')} | {n_img} |"
        )

    lines += ["", "## FID results", ""] + fid_lines
    lines += [
        "",
        "Notes:",
        "- NFE = network forward passes per sample. AF-PSM `2a` costs 3 forwards per interval",
        "  (NFE = 3 x steps), so compare it against an NFE-matched baseline run.",
        "- FID on 32x32 data goes through the Inception resize to 299 and is only comparable",
        "  *within* this table (same detector / reference set / sample count).",
        "",
        "## Reproduction records ([override] lines)",
        "",
        "```",
    ] + (override_lines or ["(no console logs found; pass --extra-dir)"]) + [
        "```",
        "",
        "## File inventory",
        "",
        f"Total size: {total_bytes / 2**20:.1f} MiB",
        "",
        "| path | size |",
        "|---|---:|",
    ]
    for rel, size in inventory:
        lines.append(f"| {rel} | {size / 2**20:.2f} MiB |")
    lines += ["", f"_generated by `{Path(__file__).name}`_", ""]

    manifest = out / "MANIFEST.md"
    manifest.write_text("\n".join(lines), encoding="utf-8")
    (out / "fid_table.md").write_text("\n".join(fid_lines) + "\n", encoding="utf-8")

    print(f"collected {len(run_rows)} runs, {len(fid_rows)} FID rows, "
          f"{len(inventory)} entries")
    print(f"total size: {total_bytes / 2**20:.1f} MiB")
    if ckpt_files:
        print(f"checkpoints: {ckpt_files} files / {ckpt_bytes / 2**20:.1f} MiB")
    print(f"snapshot: {out}")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
