"""
preprocess.py
─────────────
Reads labels from adni_csv.csv (columns: Subject, Group),
matches them to .nii files in adni_clean/ by subject ID,
preprocesses each volume, and writes df_train/val/test.csv.

Run once before training:
    python data/preprocess.py --data_root ./adni_clean --csv ./data/adni_csv.csv --out_dir .
"""

import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split


LABEL_MAP = {"cn": 0, "mci": 1, "ad": 2}


# ── Preprocessing helpers ──────────────────────────────────────────────────────

def load_nifti(path):
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    zooms = img.header.get_zooms()[:3]
    return data, zooms


def resample_and_resize(volume, orig_spacing, target_spacing=1.75, target_size=128):
    zoom_factors = [s / target_spacing for s in orig_spacing]
    resampled = zoom(volume, zoom_factors, order=1)
    final_zoom = [target_size / s for s in resampled.shape]
    resized = zoom(resampled, final_zoom, order=1)
    return resized.astype(np.float32)


def zscore_norm(volume):
    mu, sigma = volume.mean(), volume.std()
    if sigma < 1e-8:
        return volume - mu
    return (volume - mu) / sigma


def simple_skull_strip(volume):
    threshold = 0.1 * volume.max()
    mask = (volume > threshold).astype(np.float32)
    return volume * mask


def process_one(path):
    volume, spacing = load_nifti(path)
    volume = simple_skull_strip(volume)
    volume = resample_and_resize(volume, spacing)
    volume = zscore_norm(volume)
    return volume   # (128, 128, 128)


# ── Label CSV loading ──────────────────────────────────────────────────────────

def load_label_map(csv_path):
    """
    Read adni_csv.csv and return a dict:
        { subject_id_normalised : label_int }

    Subject IDs look like '051_S_1331' in the CSV.
    We normalise to lowercase for matching.
    """
    df = pd.read_csv(csv_path)

    # Flexible column detection
    subject_col = None
    group_col   = None
    for col in df.columns:
        cl = col.strip().lower()
        if cl in ("subject", "subject id", "ptid", "subject_id"):
            subject_col = col
        if cl in ("group", "dx", "diagnosis", "dx_bl"):
            group_col = col

    if subject_col is None or group_col is None:
        raise ValueError(
            f"Could not find Subject/Group columns in {csv_path}.\n"
            f"Columns found: {list(df.columns)}"
        )

    label_dict = {}
    skipped = 0
    for _, row in df.iterrows():
        subj  = str(row[subject_col]).strip().lower().replace("-", "_")
        group = str(row[group_col]).strip().lower()
        if group not in LABEL_MAP:
            skipped += 1
            continue
        label_dict[subj] = LABEL_MAP[group]

    print(f"  Loaded {len(label_dict)} labelled subjects "
          f"({skipped} rows skipped — unknown group).")
    return label_dict


def extract_subject_id(filename):
    """
    Pull subject ID from ADNI filename.
    ADNI_051_S_1331_MR_...nii  →  '051_s_1331'
    """
    base  = os.path.basename(filename).replace(".nii.gz", "").replace(".nii", "")
    parts = base.split("_")
    try:
        adni_idx = next(i for i, p in enumerate(parts) if p.upper() == "ADNI")
        # subject ID = next 3 tokens: e.g. ['051', 'S', '1331']
        subject_id = "_".join(parts[adni_idx + 1 : adni_idx + 4]).lower()
        return subject_id
    except (StopIteration, IndexError):
        return None


# ── File list builder ──────────────────────────────────────────────────────────

def build_file_list(data_root, label_dict):
    records = []
    skipped = []

    for fname in sorted(os.listdir(data_root)):
        if not (fname.endswith(".nii") or fname.endswith(".nii.gz")):
            continue

        fpath = os.path.join(data_root, fname)
        subj  = extract_subject_id(fname)

        if subj is None or subj not in label_dict:
            skipped.append(fname)
            continue

        records.append({
            "source_path": fpath,
            "label":       label_dict[subj],
            "subject":     subj,
        })

    df = pd.DataFrame(records)
    print(f"  Matched {len(df)} files | Skipped {len(skipped)} (no label)")
    if skipped:
        print(f"  Example skipped: {skipped[0]}")
        print(f"  Extracted ID from it: {extract_subject_id(skipped[0])}")
    return df


# ── Preprocessing + saving ─────────────────────────────────────────────────────

def preprocess_and_save(df, out_dir):
    npy_dir   = os.path.join(out_dir, "adni_processed")
    os.makedirs(npy_dir, exist_ok=True)
    new_paths = []

    for i, (_, row) in enumerate(df.iterrows()):
        src  = row["source_path"]
        stem = os.path.basename(src).replace(".nii.gz", "").replace(".nii", "")
        dst  = os.path.join(npy_dir, f"{stem}.npy")

        if os.path.exists(dst):
            print(f"  [{i+1}/{len(df)}] CACHED   {stem[:60]}")
        else:
            print(f"  [{i+1}/{len(df)}] PROCESS  {stem[:60]}")
            vol = process_one(src)
            np.save(dst, vol)

        new_paths.append(dst)

    df = df.copy()
    df["filepath"] = new_paths
    return df[["filepath", "label", "subject"]]


# ── Splits ─────────────────────────────────────────────────────────────────────

def make_splits(df, train_size=0.70, val_size=0.15, seed=42):
    """70 / 15 / 15 subject-level stratified split."""
    train_df, temp_df = train_test_split(
        df, test_size=round(1 - train_size, 4),
        stratify=df["label"], random_state=seed
    )
    relative_val = val_size / (1 - train_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=round(1 - relative_val, 4),
        stratify=temp_df["label"], random_state=seed
    )
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./adni_clean")
    parser.add_argument("--csv",       type=str, default="./data/adni_csv.csv")
    parser.add_argument("--out_dir",   type=str, default=".")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    print("── Step 1: Loading label CSV ──")
    label_dict = load_label_map(args.csv)

    print("\n── Step 2: Scanning NIfTI files ──")
    df = build_file_list(args.data_root, label_dict)

    if len(df) == 0:
        print("\n[ERROR] No files matched.")
        print("  Make sure adni_csv.csv is in ./data/ and subject IDs match filenames.")
        print("  Example filename: ADNI_051_S_1331_MR_...")
        print("  Expected CSV subject ID: 051_S_1331")
        return

    label_names = {0: "CN", 1: "MCI", 2: "AD"}
    counts = df["label"].value_counts().sort_index().to_dict()
    print(f"  Distribution: { {label_names[k]: v for k,v in counts.items()} }")

    print("\n── Step 3: Preprocessing ──")
    df = preprocess_and_save(df, args.out_dir)

    print("\n── Step 4: Splits ──")
    train_df, val_df, test_df = make_splits(df, seed=args.seed)

    for name, split in [("df_train", train_df), ("df_val", val_df), ("df_test", test_df)]:
        path = os.path.join(args.out_dir, f"{name}.csv")
        split.to_csv(path, index=False)
        c = split["label"].value_counts().sort_index().to_dict()
        print(f"  {name}: {len(split)} files | { {label_names[k]:v for k,v in c.items()} }")

    print("\nDone. Run train.py next.")


if __name__ == "__main__":
    main()