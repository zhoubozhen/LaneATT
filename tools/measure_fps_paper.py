import os
import sys

# ✅ 自动把上一级（项目根目录）加入到 Python 模块搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import torch
import argparse
from lib.config import Config

def parse_args():
    parser = argparse.ArgumentParser(description="Measure LaneATT FPS (as defined in the paper)")
    parser.add_argument("--cfg", required=True, help="Config file")
    parser.add_argument("--model_path", required=True, help="Trained model checkpoint")
    parser.add_argument("--iters", type=int, default=500, help="Number of iterations to average over")
    parser.add_argument("--no_cuda", action="store_true", help="Use CPU instead of GPU")
    return parser.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")

    # === Load model ===
    cfg = Config(args.cfg)
    model = cfg.get_model().to(device)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # === Prepare constant input ===
    h, w = cfg["datasets"]["test"]["parameters"]["img_size"]
    x = torch.zeros((1, 3, h, w)).to(device)

    # === Warmup ===
    for _ in range(50):
        _ = model(x, **cfg.get_test_parameters())

    # === Measure FPS ===
    total = 0.0
    iters = args.iters
    print(f"🚀 Measuring FPS (forward only, constant input) for {iters} iterations...")
    for _ in range(iters):
        t1 = time.time()
        _ = model(x, **cfg.get_test_parameters())
        t2 = time.time()
        total += (t2 - t1)

    avg_time = total / iters
    fps = 1.0 / avg_time
    print(f"\n✅ LaneATT paper-style FPS (forward only): {fps:.2f}")
    print(f"⏱️ Average latency per frame: {avg_time*1000:.2f} ms")

if __name__ == "__main__":
    main()
