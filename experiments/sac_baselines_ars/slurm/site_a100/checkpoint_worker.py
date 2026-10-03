#!/usr/bin/env python3
"""Asynchronously offload one job's rolling checkpoint to the rclone remote (CHECKPOINT_REMOTE)."""
import argparse, os, shutil, subprocess, time
from pathlib import Path

def _dir_signature(d: Path):
    items = []
    if d.is_dir():
        for q in sorted(d.iterdir()):
            if q.is_file():
                st = q.stat()
                items.append((q.name, st.st_size, st.st_mtime_ns))
    return tuple(items)


def upload_once(root: Path, remote_root: str) -> bool:
    """Upload the rolling checkpoint; race-safe against trainer rotation.

    A 19 GB upload takes minutes; the trainer may rotate latest->prev and
    write a NEW latest mid-upload.  Uploading the live dir would then ship a
    torn mix and the final rmtree would delete the newer checkpoint we never
    uploaded.  So: hardlink-snapshot latest (instant, same fs; the trainer
    never modifies checkpoint files in place — it writes a tmp dir and
    renames), upload the snapshot, and delete the real latest only if it is
    UNCHANGED since the snapshot.
    """
    latest = root / "latest"
    if not latest.is_dir() or not (latest / "meta.json").is_file():
        return False
    group, name = root.parent.name, root.name
    remote = f"{remote_root.rstrip('/')}/{group}/{name}/latest"
    timeout = int(os.environ.get("CHECKPOINT_RCLONE_TIMEOUT", "21600"))
    snap = root / ".upload_snap"
    try:
        if snap.is_dir():
            shutil.rmtree(snap)
        sig_before = _dir_signature(latest)
        shutil.copytree(latest, snap, copy_function=os.link)
        subprocess.run(["rclone", "copy", str(snap), remote, "--transfers", "1", "--checkers", "2", "--retries", "3"], check=True, timeout=timeout)
        prev = root / "prev"
        if prev.is_dir():
            subprocess.run(["rclone", "copy", str(prev), f"{remote_root.rstrip('/')}/{group}/{name}/prev", "--transfers", "1", "--checkers", "2", "--retries", "3"], check=True, timeout=timeout)
            if prev.is_dir():
                shutil.rmtree(prev)
        if _dir_signature(latest) == sig_before:
            shutil.rmtree(latest)
            print(f"[checkpoint-worker] uploaded {remote}; local latest removed", flush=True)
        else:
            print(f"[checkpoint-worker] uploaded {remote}; latest rotated mid-upload, kept for next pass", flush=True)
        return True
    except Exception as e:
        print(f"[checkpoint-worker] upload failed ({e}); retaining local", flush=True)
        return False
    finally:
        if snap.is_dir():
            shutil.rmtree(snap, ignore_errors=True)

def checkpoint_signature(root: Path):
    """Return a cheap bounded signature of the rolling checkpoint trees.
    The initial signature protects a checkpoint restored before the worker starts.
    """
    items = []
    for dirname in ("latest", "prev"):
        d = root / dirname
        if not d.is_dir():
            continue
        for q in sorted(d.iterdir()):
            if q.is_file():
                st = q.stat()
                items.append((dirname, q.name, st.st_size, st.st_mtime_ns))
    return tuple(items)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("remote"); ap.add_argument("--interval",type=int,default=30); ap.add_argument("--once",action="store_true"); a=ap.parse_args()
    root=Path(a.root)
    if a.once: upload_once(root,a.remote); return
    # Do not upload/delete a checkpoint restored before the worker started.
    # A later trainer write changes the signature and is eligible for upload.
    baseline = checkpoint_signature(root)
    while True:
        current = checkpoint_signature(root)
        if current != baseline and upload_once(root, a.remote):
            baseline = checkpoint_signature(root)
        time.sleep(a.interval)
if __name__ == "__main__": main()
