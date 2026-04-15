"""
generate_augmented.py
─────────────────────
Pre-generates augmented copies of training scans on disk.
Turns ~400 training scans into ~2800 (6 augmented versions each).

Run ONCE after preprocess.py:
    python data/generate_augmented.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from data.augument import get_train_transforms

COPIES = 6
SEED   = 42


def augment_and_save(vol_path, out_dir, n_copies, transform):
    vol  = np.load(vol_path, allow_pickle=False).astype(np.float32)
    t    = torch.from_numpy(vol).unsqueeze(0)   # (1, D, H, W)
    paths = []

    stem = os.path.splitext(os.path.basename(vol_path))[0]
    for i in range(n_copies):
        aug      = transform(t).squeeze(0).numpy()
        out_path = os.path.join(out_dir, f"{stem}_aug{i}.npy")
        np.save(out_path, aug)
        paths.append(out_path)
    return paths


def main():
    df_train = pd.read_csv("./df_train.csv")
    out_dir  = "./adni_processed/augmented"
    os.makedirs(out_dir, exist_ok=True)

    transform = get_train_transforms()
    new_rows  = []

    print(f"Augmenting {len(df_train)} training scans x {COPIES} copies...")
    print(f"This will create {len(df_train) * COPIES} new files.\n")

    for i, (_, row) in enumerate(df_train.iterrows()):
        src = row["filepath"]
        if not os.path.exists(src):
            print(f"  [SKIP] Missing: {src}")
            continue
        print(f"  [{i+1}/{len(df_train)}] {os.path.basename(src)[:60]}")
        aug_paths = augment_and_save(src, out_dir, COPIES, transform)
        for p in aug_paths:
            new_rows.append({
                "filepath": p,
                "label":    row["label"],
                "subject":  row.get("subject", "aug"),
            })

    # Combine originals + augmented and shuffle
    aug_df   = pd.DataFrame(new_rows)
    combined = pd.concat([df_train, aug_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=SEED).reset_index(drop=True)
    combined.to_csv("./df_train.csv", index=False)

    label_names = {0: "CN", 1: "MCI", 2: "AD"}
    counts = combined["label"].value_counts().sort_index().to_dict()
    print(f"\nDone!")
    print(f"df_train.csv now has {len(combined)} rows")
    print(f"  Original: {len(df_train)} | Augmented: {len(aug_df)}")
    print(f"  Labels: { {label_names[k]: v for k,v in counts.items()} }")


if __name__ == "__main__":
    main()