import torch
from mmcv import Config
from pyskl.models import build_model
from fvcore.nn import FlopCountAnalysis, parameter_count

def analyze(config_path, name):
    cfg = Config.fromfile(config_path)
    model = build_model(cfg.model)
    model.eval()

    params = parameter_count(model)['']
    dummy_input = torch.randn(1, 2, 100, 17, 3)
    flops = FlopCountAnalysis(model, dummy_input)

    print(f"--- {name} ---")
    print(f"Parameters: {params:,}")
    print(f"FLOPs: {flops.total():,}")
    print()

analyze('/content/pyskl/configs/stgcn/stgcn_lite_hmdb51_hrnet/j.py', 'Lite-STGCN (dwstcn)')
analyze('/content/pyskl/configs/stgcn/stgcn_baseline_hmdb51_hrnet/j.py', 'Baseline STGCN (unit_tcn)')