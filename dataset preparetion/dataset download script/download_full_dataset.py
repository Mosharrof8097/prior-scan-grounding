#!/usr/bin/env python3
"""
Full Dataset Downloader for 3D-RAD (NeurIPS 2025)
Repository: Tang-xiaoxiao/3D-RAD on Hugging Face
Total Size: ~20.6 GB compressed (~25 GB uncompressed)

Features:
- Live terminal progress bar (Percentage, Downloaded GB, Speed, ETA)
- Automatic RESUME support (if interrupted by Ctrl+C or network disconnect, it continues where it left off)
- Downloads all QA CSV files (train & test tasks) + Images/test.zip (20.6 GB)
- Optional automatic unzipping with --unzip flag
"""

import os
import sys
import time
import shutil
import zipfile
import argparse
import urllib.request

REPO_ID = "Tang-xiaoxiao/3D-RAD"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "3D-RAD_Dataset")


def check_disk_space():
    total, used, free = shutil.disk_usage(BASE_DIR)
    free_gb = free / (1024 ** 3)
    print(f"[DISK] Available disk space: {free_gb:.2f} GB")
    if free_gb < 25:
        print("[WARNING] You have less than 25 GB free. The full dataset is ~20.6 GB compressed.")
        print("          If you plan to unzip, you will need at least ~45-50 GB free.")
        resp = input("Do you wish to continue anyway? (y/n): ").strip().lower()
        if resp != "y":
            print("Aborted by user.")
            sys.exit(0)
    else:
        print(f"[OK] Sufficient disk space detected ({free_gb:.1f} GB free).\n")


def download_with_huggingface_hub(target_dir):
    """
    Downloads the full repository using official Hugging Face Hub library.
    Includes built-in multi-threading, live progress bar, auto-retry, and resume.
    """
    try:
        from huggingface_hub import snapshot_download
        print("=" * 70)
        print("     STARTING FULL DATASET DOWNLOAD VIA HUGGING FACE HUB")
        print(f"     Repository: {REPO_ID}")
        print(f"     Save Path:  {target_dir}")
        print("=" * 70)
        print("[INFO] Features: Live progress bar, auto-resume, hash verification.")
        print("[INFO] Total download size is ~20.6 GB. Please keep internet active.\n")

        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4
        )
        print("\n[SUCCESS] Download completed successfully via huggingface_hub!")
        return True
    except Exception as e:
        print(f"\n[NOTICE] huggingface_hub download encountered an issue: {e}")
        print("[INFO] Falling back to direct HTTP resumable stream downloader...\n")
        return False


def format_size(bytes_val):
    if bytes_val >= 1024 ** 3:
        return f"{bytes_val / (1024 ** 3):.2f} GB"
    elif bytes_val >= 1024 ** 2:
        return f"{bytes_val / (1024 ** 2):.2f} MB"
    elif bytes_val >= 1024:
        return f"{bytes_val / 1024:.2f} KB"
    return f"{bytes_val} B"


def download_file_resumable(url, dest_path, desc="Downloading"):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    existing_bytes = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0

    # Get total remote size
    req_head = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req_head) as resp:
        total_remote_size = int(resp.headers.get("Content-Length", 0))

    if existing_bytes >= total_remote_size and total_remote_size > 0:
        print(f"  [ALREADY COMPLETE] {desc} ({format_size(existing_bytes)})")
        return

    headers = {"User-Agent": "Mozilla/5.0"}
    mode = "wb"
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"
        mode = "ab"
        print(f"  [RESUMING] {desc} from {format_size(existing_bytes)} / {format_size(total_remote_size)}...")
    else:
        print(f"  [STARTING] {desc} ({format_size(total_remote_size)})...")

    req = urllib.request.Request(url, headers=headers)
    chunk_size = 256 * 1024  # 256 KB chunks
    downloaded = existing_bytes
    t_start = time.time()
    t_last_update = t_start

    with urllib.request.urlopen(req) as resp:
        with open(dest_path, mode) as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                if now - t_last_update >= 0.2 or downloaded == total_remote_size:
                    t_last_update = now
                    elapsed = now - t_start
                    speed = (downloaded - existing_bytes) / elapsed if elapsed > 0 else 0
                    speed_mb = speed / (1024 * 1024)
                    pct = (downloaded / total_remote_size) * 100 if total_remote_size > 0 else 0
                    eta_sec = (total_remote_size - downloaded) / speed if speed > 0 else 0
                    eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60)}s" if eta_sec > 0 else "--"

                    bar_len = 30
                    filled = int(bar_len * downloaded // total_remote_size) if total_remote_size > 0 else 0
                    bar = "=" * filled + (">" if filled < bar_len else "") + " " * (bar_len - filled - 1)
                    if filled == bar_len:
                        bar = "=" * bar_len

                    sys.stdout.write(
                        f"\r  [{bar}] {pct:5.1f}% | {format_size(downloaded)} / {format_size(total_remote_size)} | {speed_mb:5.2f} MB/s | ETA: {eta_str}  "
                    )
                    sys.stdout.flush()

    sys.stdout.write("\n")
    print(f"  [OK] Completed: {os.path.basename(dest_path)}")


def unzip_archive(zip_path, extract_to):
    print(f"\n[UNZIP] Extracting {zip_path} to {extract_to}...")
    print("        This may take several minutes as it contains ~3,000 3D volumes...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        total_files = len(members)
        for i, member in enumerate(members, 1):
            zf.extract(member, extract_to)
            if i % 100 == 0 or i == total_files:
                pct = (i / total_files) * 100
                sys.stdout.write(f"\r  Extracting: [{pct:5.1f}%] ({i}/{total_files} files)")
                sys.stdout.flush()
    sys.stdout.write("\n")
    print("[OK] Extraction complete!")


def main():
    parser = argparse.ArgumentParser(description="Download full 3D-RAD dataset (NeurIPS 2025)")
    parser.add_argument("--dest", type=str, default=DATASET_DIR, help="Destination directory")
    parser.add_argument("--unzip", action="store_true", help="Automatically unzip Images/test.zip after download")
    parser.add_argument("--direct", action="store_true", help="Force direct HTTP download instead of huggingface_hub")
    args = parser.parse_args()

    print("=" * 70)
    print("      3D-RAD (NeurIPS 2025) - FULL DATASET DOWNLOADER")
    print("=" * 70)
    check_disk_space()

    target_dir = args.dest
    os.makedirs(target_dir, exist_ok=True)

    success = False
    if not args.direct:
        success = download_with_huggingface_hub(target_dir)

    if not success:
        # Fallback to direct HTTP with resume
        print("[HTTP] Downloading main image archive with resume support...")
        zip_url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/Images/test.zip"
        zip_dest = os.path.join(target_dir, "Images", "test.zip")
        download_file_resumable(zip_url, zip_dest, desc="Images/test.zip (20.6 GB)")

    zip_file = os.path.join(target_dir, "Images", "test.zip")
    if args.unzip and os.path.exists(zip_file):
        extract_dir = os.path.join(target_dir, "Images")
        unzip_archive(zip_file, extract_dir)

    print("\n" + "=" * 70)
    print("ALL DONE!")
    print(f"Dataset location: {target_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
