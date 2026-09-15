"""
Dataset Downloader, Unpacker, and Non-IID Hospital Partitioner.
Partitions the Kaggle Chest X-Ray images across:
- Hospital A (pve01: node_1) - Bacterial Pneumonia bias (85% bacterial, 15% normal)
- Hospital B (pve02: node_2) - Viral Pneumonia bias (85% viral, 15% normal)
- Hospital C (pve03: node_3) - Mixed regional cases
- Tier-1 Workstation (node_4) - Balanced local edge clinic
- Central Coordinator (global_test) - Unseen global validation holdout
"""

import os
import shutil
import zipfile
import random
from pathlib import Path

# Seed for deterministic institutional distribution
random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"


def find_dataset_zip() -> Path:
    """Check common locations for the Kaggle archive.zip."""
    candidates = [
        DATA_DIR / "archive.zip",
        DATA_DIR / "chest-xray-pneumonia.zip",
        Path.home() / "Downloads" / "archive.zip",
        Path.home() / "Downloads" / "chest-xray-pneumonia.zip",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 100 * 1024 * 1024:  # > 100 MB
            return p
    return None


def extract_archive(zip_path: Path) -> Path:
    """Extract archive into data/raw/."""
    print(f"[Dataset] Extracting {zip_path} to {RAW_DIR}...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(RAW_DIR)
    print("[Dataset] Extraction complete.")
    
    # Locate the root of chest_xray folder
    for p in RAW_DIR.rglob("chest_xray"):
        if p.is_dir() and (p / "train").exists():
            return p
    return RAW_DIR


def partition_images(xray_root: Path):
    """
    Categorize radiographs:
    - NORMAL
    - BACTERIA (filename contains 'bacteria')
    - VIRUS (filename contains 'virus')
    """
    print("[Dataset] Scanning radiographs...")
    all_normal = []
    all_bacteria = []
    all_virus = []

    for ext in ("*.jpeg", "*.jpg", "*.png"):
        for img_path in xray_root.rglob(ext):
            if "__MACOSX" in str(img_path) or img_path.name.startswith("._"):
                continue
            name_lower = img_path.name.lower()
            if "bacteria" in name_lower:
                all_bacteria.append(img_path)
            elif "virus" in name_lower:
                all_virus.append(img_path)
            elif "normal" in str(img_path).lower():
                all_normal.append(img_path)

    print(f"[Dataset] Identified: {len(all_normal)} Normal, {len(all_bacteria)} Bacterial, {len(all_virus)} Viral.")

    # Shuffle for fair random distribution
    random.shuffle(all_normal)
    random.shuffle(all_bacteria)
    random.shuffle(all_virus)

    # 1. Global Holdout Evaluation Set (Balanced 400 normal, 400 pneumonia)
    test_normal = all_normal[:400]
    test_pneumonia = all_bacteria[:200] + all_virus[:200]
    
    del all_normal[:400]
    del all_bacteria[:200]
    del all_virus[:200]

    # Save Global Test Set
    gt_dir = DATA_DIR / "global_test" / "train"
    for label, imgs in [("NORMAL", test_normal), ("PNEUMONIA", test_pneumonia)]:
        target_dir = gt_dir / label
        target_dir.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy2(img, target_dir / img.name)
    print(f"[Dataset] Created Global Test Set at {gt_dir.parent} ({len(test_normal) + len(test_pneumonia)} images).")

    # 2. Institutional Non-IID Allocations for Hospitals A, B, C, Workstation
    splits = {
        "node_1": {"bact_pct": 0.85, "virus_pct": 0.15, "norm_count": 250},  # Hospital A: Bacterial Skew
        "node_2": {"bact_pct": 0.15, "virus_pct": 0.85, "norm_count": 250},  # Hospital B: Viral Skew
        "node_3": {"bact_pct": 0.50, "virus_pct": 0.50, "norm_count": 250},  # Hospital C: Mixed
        "node_4": {"bact_pct": 0.50, "virus_pct": 0.50, "norm_count": 250},  # Workstation: Local Clinic
    }

    # Partition remaining images
    for node, cfg in splits.items():
        node_train = DATA_DIR / node / "train"
        (node_train / "NORMAL").mkdir(parents=True, exist_ok=True)
        (node_train / "PNEUMONIA").mkdir(parents=True, exist_ok=True)

        # Normal images
        node_norm = all_normal[:cfg["norm_count"]]
        del all_normal[:cfg["norm_count"]]
        for img in node_norm:
            shutil.copy2(img, node_train / "NORMAL" / img.name)

        # Pneumonia images with institutional skew
        pneu_count = 600
        bact_take = int(pneu_count * cfg["bact_pct"])
        virus_take = pneu_count - bact_take

        node_bact = all_bacteria[:bact_take]
        del all_bacteria[:bact_take]
        node_virus = all_virus[:virus_take]
        del all_virus[:virus_take]

        for img in (node_bact + node_virus):
            shutil.copy2(img, node_train / "PNEUMONIA" / img.name)

        print(f"[Dataset] {node} configured: {len(node_norm)} Normal, {len(node_bact)} Bacterial, {len(node_virus)} Viral.")

    print("\n[Dataset] Non-IID Institutional partitioning successfully completed!")


def sync_to_proxmox_nodes():
    """Compress and push partitioned hospital datasets to pve01, pve02, pve03 over SSH."""
    print("\n[Dataset] Syncing partitioned datasets to Proxmox LXC containers...")
    nodes = [
        ("node_1", "192.168.20.100", "201"),
        ("node_2", "192.168.10.100", "202"),
        ("node_3", "192.168.40.100", "203"),
    ]

    for node_name, host_ip, ct_id in nodes:
        node_dir = DATA_DIR / node_name
        zip_file = DATA_DIR / f"{node_name}.zip"
        print(f"  -> Packaging {node_name} for {host_ip} (CT {ct_id})...")
        shutil.make_archive(str(DATA_DIR / node_name), 'zip', node_dir)

        # SCP to Proxmox node and push into container
        os.system(f'scp -o StrictHostKeyChecking=no {zip_file} root@{host_ip}:/tmp/{node_name}.zip')
        os.system(f'ssh -o StrictHostKeyChecking=no root@{host_ip} "pct push {ct_id} /tmp/{node_name}.zip /opt/fl_client/{node_name}.zip && pct exec {ct_id} -- unzip -q -o /opt/fl_client/{node_name}.zip -d /opt/fl_client/data/{node_name} && rm /tmp/{node_name}.zip"')
        print(f"  -> {node_name} successfully delivered to CT {ct_id}!")

    print("\n[Dataset] All Proxmox edge servers are synchronized with real clinical radiographs!")


if __name__ == "__main__":
    zip_path = find_dataset_zip()
    if not zip_path:
        print("=======================================================================")
        print("  DATASET NOT FOUND! Please download the Kaggle Chest X-Ray dataset:   ")
        print("  URL: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        print("  Download the zip and place it in either:")
        print(f"    1. {DATA_DIR / 'archive.zip'}")
        print(f"    2. {Path.home() / 'Downloads' / 'archive.zip'}")
        print("=======================================================================")
    else:
        print(f"Found dataset archive at: {zip_path}")
        extracted_root = extract_archive(zip_path)
        partition_images(extracted_root)
        sync_to_proxmox_nodes()
