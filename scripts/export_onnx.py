import torch
import torch.nn as nn
from mmcv import Config
from pyskl.models import build_model

# --- Load your trained lite model ---
cfg = Config.fromfile('/content/pyskl/configs/stgcn/stgcn_lite_hmdb51_hrnet/j.py')
model = build_model(cfg.model)

checkpoint = torch.load(
    '/content/drive/MyDrive/wes237b_project/work_dirs/lite_stgcn_hmdb51/best_top1_acc_epoch_14.pth',
    map_location='cpu'
)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# --- Minimal wrapper: backbone + cls_head only, no training/multi-clip logic ---
class InferenceWrapper(nn.Module):
    def __init__(self, backbone, cls_head):
        super().__init__()
        self.backbone = backbone
        self.cls_head = cls_head

    def forward(self, keypoint):
        # keypoint: (N, M, T, V, C)
        x = self.backbone(keypoint)
        return self.cls_head(x)

wrapper = InferenceWrapper(model.backbone, model.cls_head)
wrapper.eval()

# --- Dummy input matching your actual pipeline's per-sample shape ---
dummy_input = torch.randn(1, 2, 100, 17, 3)  # (N, M, T, V, C)

# --- Sanity check: does the wrapper run at all before exporting? ---
with torch.no_grad():
    out = wrapper(dummy_input)
print("Forward pass output shape:", out.shape)  # expect (1, 51) — one score per HMDB51 class

# --- Export ---
torch.onnx.export(
    wrapper,
    dummy_input,
    '/content/drive/MyDrive/wes237b_project/lite_stgcn_hmdb51.onnx',
    input_names=['keypoint'],
    output_names=['action_scores'],
    opset_version=12,
    dynamic_axes=None  # fixed shape — matches your RB3 pipeline's fixed clip_len=100
)
print("Export complete.")
