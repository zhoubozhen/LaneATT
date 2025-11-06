import os
import sys

# ✅ 自动把上一级（项目根目录）加入到 Python 模块搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import torch
from lib.config import Config

CFG_PATH = "cfgs/laneatt_tusimple_split_resnet18.yml"

cfg = Config(CFG_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = cfg.get_model().to(device)
model.eval()

# 伪造随机输入 (batch_size=1)
dummy_input = torch.randn(1, 3, 360, 640).to(device)
n_warmup, n_test = 20, 200

print("🚀 Measuring pure inference FPS...")
times = []
with torch.no_grad():
    for i in range(n_warmup + n_test):
        if i < n_warmup:
            _ = model(dummy_input)
            continue
        start = time.time()
        _ = model(dummy_input)
        torch.cuda.synchronize()
        times.append(time.time() - start)

avg_time = sum(times) / len(times)
fps = 1.0 / avg_time
print(f"✅ Pure model FPS: {fps:.2f}")
