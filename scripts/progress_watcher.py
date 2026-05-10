#!/usr/bin/env python3
"""
Progress watcher — runs alongside training and updates progress.txt
every UPDATE_INTERVAL_SECS with a live progress bar + current metrics.

Usage:
    python scripts/progress_watcher.py <experiment_dir> [--interval 120]
"""
import argparse
import datetime
import os
import re
import sys
import time


UPDATE_INTERVAL_SECS = 120   # write to progress.txt every 2 minutes


def parse_latest_batch(log_detail_path: str):
    """Parse the latest tqdm line from log_detail.txt."""
    if not os.path.exists(log_detail_path):
        return None
    try:
        with open(log_detail_path, "rb") as f:
            # Read last 64 KB — enough to find the latest tqdm state
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 65536))
            content = f.read().decode("utf-8", errors="replace")
    except Exception:
        return None

    lines = re.split(r"[\r\n]", content)
    for line in reversed(lines):
        m = re.search(
            r"Epoch (\d+)/(\d+).*\|\s*(\d+)/(\d+)\s*\[(\S+)<(\S+),\s*(\S+)\].*loss=(\S+)",
            line,
        )
        if m:
            epoch, total_e, batch, total_b, elapsed, eta, speed, loss = m.groups()
            return {
                "epoch": int(epoch),
                "total_epochs": int(total_e),
                "batch": int(batch),
                "total_batches": int(total_b),
                "elapsed": elapsed,
                "eta_epoch": eta,
                "speed": speed,
                "loss": loss.rstrip(","),
            }
    return None


def parse_result_rows(result_path: str):
    """Parse completed epoch rows from result.txt into list of dicts."""
    if not os.path.exists(result_path):
        return []
    rows = []
    with open(result_path) as f:
        for line in f:
            line = line.rstrip()
            if not line.startswith("Epoch"):
                continue
            # Format: "Epoch  N [TAG] | Accuracy=X% | AUC=X% | Real_Accuracy=X% | Fake_Accuracy=X% | Loss=X"
            ep_m = re.search(r"Epoch\s+(\d+)", line)
            if not ep_m:
                continue
            epoch_num = int(ep_m.group(1))
            tags = re.findall(r"\[([^\]]+)\]", line)
            parts = {}
            for seg in line.split("|")[1:]:
                seg = seg.strip()
                if "=" in seg:
                    k, v = seg.split("=", 1)
                    parts[k.strip()] = v.strip().rstrip("%")
            rows.append({
                "epoch": epoch_num,
                "tags": tags,
                "accuracy":      parts.get("Accuracy",      "?"),
                "auc":           parts.get("AUC",           "?"),
                "real_accuracy": parts.get("Real_Accuracy", "?"),
                "fake_accuracy": parts.get("Fake_Accuracy", "?"),
                "loss":          parts.get("Loss",          "?"),
            })
    return rows


def make_progress_block(info: dict, result_rows: list, exp_dir: str) -> str:
    """Build the progress block string to write into progress.txt."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    epoch = info["epoch"]
    total_epochs = info["total_epochs"]
    batch = info["batch"]
    total_batches = info["total_batches"]

    epoch_pct  = (epoch - 1 + batch / total_batches) / total_epochs * 100
    batch_pct  = batch / total_batches * 100
    bar_full   = int(epoch_pct / 5)
    bar_str    = "█" * bar_full + "░" * (20 - bar_full)

    # Estimate overall ETA
    try:
        def parse_hms(s):
            """Parse H:MM:SS or MM:SS or HH:MM:SS into seconds."""
            parts = s.strip().split(":")
            parts = [int(float(p)) for p in parts]
            if len(parts) == 3:
                return parts[0] * 3600 + parts[1] * 60 + parts[2]
            elif len(parts) == 2:
                return parts[0] * 60 + parts[1]
            return int(parts[0])

        eta_epoch_secs       = parse_hms(info["eta_epoch"])
        elapsed_secs         = parse_hms(info["elapsed"])
        secs_per_batch       = elapsed_secs / max(batch, 1)
        epoch_total_secs     = secs_per_batch * total_batches
        remaining_full_epochs = total_epochs - epoch          # epochs after current
        overall_eta_secs     = eta_epoch_secs + remaining_full_epochs * epoch_total_secs
        overall_eta = str(datetime.timedelta(seconds=int(overall_eta_secs)))
    except Exception:
        overall_eta = "calculating..."

    block = (
        f"\n{'─'*70}\n"
        f"LIVE TRAINING PROGRESS  (updated {now})\n"
        f"{'─'*70}\n"
        f"  Overall  [{bar_str}] {epoch_pct:.1f}%  "
        f"Epoch {epoch}/{total_epochs}  |  Overall ETA: {overall_eta}\n"
        f"\n"
        f"  Current epoch {epoch}/{total_epochs}:\n"
        f"    Batch   : {batch:,} / {total_batches:,}  ({batch_pct:.1f}%)\n"
        f"    Speed   : {info['speed']}\n"
        f"    Elapsed : {info['elapsed']} into epoch\n"
        f"    ETA     : {info['eta_epoch']} to finish epoch {epoch}\n"
        f"    Loss    : {info['loss']}\n"
    )

    if result_rows:
        block += f"\n  Completed epochs ({len(result_rows)} / {total_epochs}):\n"
        block += f"  {'─'*72}\n"
        block += f"  {'Ep':>3}  {'Accuracy':>9}  {'AUC':>8}  {'Real_Acc':>9}  {'Fake_Acc':>9}  {'Loss':>7}  Flags\n"
        block += f"  {'─'*72}\n"
        for r in result_rows:
            tag_str = " ".join(f"[{t}]" for t in r["tags"]) if r["tags"] else ""
            acc  = f"{r['accuracy']}%"  if r['accuracy']  != "?" else "?"
            auc  = f"{r['auc']}%"       if r['auc']       != "?" else "?"
            racc = f"{r['real_accuracy']}%" if r['real_accuracy'] != "?" else "?"
            facc = f"{r['fake_accuracy']}%" if r['fake_accuracy'] != "?" else "?"
            block += (
                f"  {r['epoch']:>3}  {acc:>9}  {auc:>8}  {racc:>9}  {facc:>9}"
                f"  {r['loss']:>7}  {tag_str}\n"
            )
        block += f"  {'─'*72}\n"
        # Best stats line
        best = max(result_rows, key=lambda r: float(r['auc']) if r['auc'] != '?' else 0)
        block += f"  Best so far → Epoch {best['epoch']}  AUC={best['auc']}%  Acc={best['accuracy']}%  Loss={best['loss']}\n"
    else:
        block += f"\n  No epochs completed yet — first results appear after epoch 1.\n"

    block += f"{'─'*70}\n"
    return block


SENTINEL_START = "##PROGRESS_START##"
SENTINEL_END   = "##PROGRESS_END##"


def update_progress_txt(progress_path: str, block: str):
    """Replace the LIVE TRAINING PROGRESS section using unique sentinels."""
    wrapped = f"{SENTINEL_START}\n{block}\n{SENTINEL_END}\n"

    if not os.path.exists(progress_path):
        with open(progress_path, "w") as f:
            f.write(wrapped)
        return

    with open(progress_path, "r") as f:
        content = f.read()

    if SENTINEL_START in content:
        start = content.index(SENTINEL_START)
        end   = content.index(SENTINEL_END, start) + len(SENTINEL_END) + 1
        content = content[:start] + wrapped + content[end:]
    else:
        content = content.rstrip("\n") + "\n" + wrapped

    with open(progress_path, "w") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", help="Experiment output directory")
    parser.add_argument("--interval", type=int, default=UPDATE_INTERVAL_SECS)
    args = parser.parse_args()

    exp_dir      = args.exp_dir
    log_detail   = os.path.join(exp_dir, "log_detail.txt")
    progress_txt = os.path.join(exp_dir, "progress.txt")
    result_txt   = os.path.join(exp_dir, "result.txt")

    print(f"[progress_watcher] Watching: {exp_dir}")
    print(f"[progress_watcher] Updating progress.txt every {args.interval}s")

    while True:
        info = parse_latest_batch(log_detail)
        if info:
            result_rows = parse_result_rows(result_txt)
            block = make_progress_block(info, result_rows, exp_dir)
            update_progress_txt(progress_txt, block)
            print(
                f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
                f"Updated — Epoch {info['epoch']}/{info['total_epochs']} "
                f"batch {info['batch']:,}/{info['total_batches']:,} "
                f"({info['batch']/info['total_batches']*100:.1f}%) "
                f"loss={info['loss']} eta={info['eta_epoch']}"
            )
        else:
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Waiting for training to start...")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
