#!/usr/bin/env python3
"""
Pre-training environment and checkpoint verification.
Run before training: python scripts/verify_setup.py
"""

import os
import sys

PYTHON = "/data/awais/anaconda/envs/d3/bin/python"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.join(REPO_ROOT, "fsvfm"))
sys.path.insert(0, os.path.join(REPO_ROOT, "fsvfm", "finetune", "cross_dataset_DFD_and_DiFF"))

PASS = "  [PASS]"
FAIL = "  [FAIL]"
WARN = "  [WARN]"


def check(label, fn):
    try:
        result = fn()
        print(f"{PASS} {label}" + (f": {result}" if result else ""))
        return True
    except Exception as e:
        print(f"{FAIL} {label}: {e}")
        return False


def main():
    failures = 0
    print("\n" + "=" * 60)
    print("FSVFM Setup Verification")
    print("=" * 60)

    # ----- Python + PyTorch -----
    print("\n[1] Python / PyTorch / CUDA")

    def check_torch():
        import torch
        return f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}"

    if not check("torch", check_torch):
        failures += 1

    def check_gpus():
        import torch
        assert torch.cuda.is_available(), "CUDA not available"
        n = torch.cuda.device_count()
        assert n >= 1, f"No GPUs visible"
        names = [torch.cuda.get_device_name(i) for i in range(n)]
        mems = [torch.cuda.get_device_properties(i).total_memory / 1e9 for i in range(n)]
        return f"{n} GPU(s): " + ", ".join(f"{n} ({m:.0f}GB)" for n, m in zip(names, mems))

    if not check("GPUs visible", check_gpus):
        failures += 1

    # ----- Key packages -----
    print("\n[2] Required packages")

    for pkg, attr in [
        ("timm", "__version__"),
        ("torch.distributed", None),
        ("torchvision", "__version__"),
        ("sklearn", "__version__"),
        ("tensorboard", "__version__"),
    ]:
        def _check(p=pkg, a=attr):
            import importlib
            m = importlib.import_module(p)
            return getattr(m, a, "ok") if a else "ok"
        if not check(pkg, _check):
            failures += 1

    # ----- timm compatibility -----
    print("\n[3] timm 1.0.x compatibility")

    def check_trunc_normal():
        try:
            from timm.layers import trunc_normal_
            return "timm.layers.trunc_normal_ (1.0.x)"
        except ImportError:
            from timm.models.layers import trunc_normal_
            return "timm.models.layers.trunc_normal_ (legacy)"

    check("trunc_normal_", check_trunc_normal)

    def check_model_vit():
        sys.path.insert(0, os.path.join(REPO_ROOT, "fsvfm"))
        import models_vit
        import torch
        model = models_vit.vit_large_patch16(num_classes=2, drop_path_rate=0.1, global_pool=True)
        dummy = torch.zeros(1, 3, 224, 224)
        out = model(dummy)
        return f"ViT-L output shape: {tuple(out.shape)}"

    if not check("models_vit.vit_large_patch16", check_model_vit):
        failures += 1

    # ----- Checkpoints -----
    print("\n[4] Checkpoints")

    ckpt_paths = [
        "/data/awais/projects/GenD_NeSy/weights/FS-VFM/FS-VFM-ViT-L.pth",
        "/data/awais/projects/GenD_NeSy/weights/FS-VFM/FS-VFM-ViT-L-Adapter.pth",
    ]
    for path in ckpt_paths:
        def _check_ckpt(p=path):
            assert os.path.exists(p), f"Not found: {p}"
            size_mb = os.path.getsize(p) / 1e6
            return f"{size_mb:.0f} MB"
        if not check(os.path.basename(path), _check_ckpt):
            failures += 1

    def check_load_ckpt():
        import torch, sys, os
        sys.path.insert(0, os.path.join(REPO_ROOT, "fsvfm"))
        import models_vit
        path = "/data/awais/projects/GenD_NeSy/weights/FS-VFM/FS-VFM-ViT-L.pth"
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = models_vit.vit_large_patch16(num_classes=2, drop_path_rate=0.1, global_pool=True)
        msg = model.load_state_dict(ckpt["model"], strict=False)
        missing = [k for k in msg.missing_keys if "head" not in k and "fc_norm" not in k]
        return f"missing={missing if missing else 'none (expected)'}"

    if not check("FS-VFM-ViT-L checkpoint loads", check_load_ckpt):
        failures += 1

    # ----- DDP -----
    print("\n[5] DDP / distributed")

    def check_ddp():
        import torch.distributed
        return "torch.distributed available"

    check("torch.distributed", check_ddp)

    # ----- Composite dataset -----
    print("\n[6] Composite dataset loader")

    def check_composite_import():
        sys.path.insert(0, os.path.join(REPO_ROOT, "fsvfm", "util"))
        sys.path.insert(0, os.path.join(REPO_ROOT, "fsvfm"))
        from util.composite_dataset import CompositeDeepfakeDataset, SingleRootDataset
        return "import ok"

    if not check("composite_dataset imports", check_composite_import):
        failures += 1

    # ----- Directory structure -----
    print("\n[7] Output directories")

    for d in ["experiments", "checkpoints", "logs", "configs", "outputs"]:
        path = os.path.join(REPO_ROOT, d)
        def _check_dir(p=path):
            assert os.path.isdir(p), f"Missing: {p}"
            return p
        check(d, _check_dir)

    # ----- Summary -----
    print("\n" + "=" * 60)
    if failures == 0:
        print("ALL CHECKS PASSED — ready for training.")
    else:
        print(f"{failures} check(s) FAILED — fix above before training.")
    print("=" * 60 + "\n")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
