# config.py

microbifpn_config = {
    # ── Input ─────────────────────────────
    "input_shape": (3, 224, 224),   # 2D RGB image
    "num_classes": 3,              

    # ── Model ────────────────────────────
    "dropout": 0.3,

    # ── Training ─────────────────────────
    "epochs": 30,
    "batch_size": 16,
    "val_batch_size": 64,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,

    # ── Focal Loss ───────────────────────
    "focal_alpha": 1.0,
    "focal_gamma": 2.0,

    # ── LR Scheduler ─────────────────────
    "lr_step_size": 10,
    "lr_gamma": 0.5,

    # ── Paths ────────────────────────────
    "train_csv": "./df_train.csv",
    "val_csv": "./df_val.csv",
    "test_csv": "./df_test.csv",

    # ── Misc ─────────────────────────────
    "num_workers": 0,
    "pin_memory": False,
    "weights_dir": "./weights_finalized",
    "plots_dir": "./training_plots",
}