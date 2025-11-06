import os
import sys

# ✅ 自动把上一级（项目根目录）加入到 Python 模块搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import time
import torch
import argparse
from tqdm import tqdm

from lib.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Measure end-to-end inference FPS (LaneATT paper style)")
    parser.add_argument("--cfg", required=True, help="Path to config file")
    parser.add_argument("--model_path", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--iters", type=int, default=200, help="Number of samples to measure (default: 200)")
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU (for debug)")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")

    cfg = Config(args.cfg)
    model = cfg.get_model().to(device)
    test_params = cfg.get_test_parameters()

    print(f"🔹 Loading model from: {args.model_path}")
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dataset = cfg.get_dataset("val")  # 验证集测速度
    print(f"✅ Dataset loaded: {len(dataset)} samples")

    # warmup GPU
    dummy = torch.zeros((1, 3, cfg['datasets']['val']['parameters']['img_size'][0],
                         cfg['datasets']['val']['parameters']['img_size'][1])).to(device)
    for _ in range(20):
        model(dummy, **test_params)

    total_time = 0.0
    count = min(args.iters, len(dataset))
    print(f"🚀 Measuring full pipeline FPS on {device} ... ({count} samples)")

    for i in tqdm(range(count)):
        # ===== 数据加载 =====
        t0 = time.time()
        sample = dataset[i]
        image = sample[0].unsqueeze(0).to(device)  # ✅ tuple结构: (image, label, meta)

        # ===== 前向推理 =====
        t1 = time.time()
        output = model(image, **test_params)

        # ===== 后处理 =====
        _ = model.decode(output, as_lanes=True)
        t2 = time.time()

        total_time += (t2 - t0)

    avg_time = total_time / count
    fps = 1.0 / avg_time
    print(f"\n✅ End-to-end FPS (with I/O + decode + NMS): {fps:.2f}")
    print(f"⏱️ Average latency per frame: {avg_time * 1000:.2f} ms")


if __name__ == "__main__":
    main()
