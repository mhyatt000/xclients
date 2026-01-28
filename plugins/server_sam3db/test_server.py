import numpy as np
import cv2
from pathlib import Path
import sys
import os
import time

from plugins.server_sam3db.server import Sam3dBodyPolicy

os.environ["SAM3DB_ROOT"] = "/data/home/lliao2/sam3db/sam-3d-body"

policy = Sam3dBodyPolicy()

# load episode
data = np.load("/data/home/lliao2/25dec10/ep0.npz")

for view in ["over", "side", "low"]:
    frames = data[view]

    joints_seq=[]
    inference_time=[]
    fail_count=0
    total_frames=len(frames)

    for i in range(total_frames):
        img=frames[i]
        payload={"image": img}

        start_time=time.time()

        try:
            out=policy.step(payload, render=True)
            elapsed=time.time() - start_time
            inference_time.append(elapsed)

            if out["render"] is None:
                fail_count +=1
                continue
            
            # joints 3D data from server output
            mesh_3d = out.get("mesh_3d", None)
            if mesh_3d is None:
                fail_count += 1
                continue

            joints_3d = mesh_3d["joints_3d"]
            joints_seq.append(joints_3d)
    
        except Exception as e:
            fail_count += 1
            print(f"[{view}] Frame {i} exception:", e)

    joints_seq = np.array(joints_seq)  # (T, J, 3)
    
    # 相邻的joints的差距
    diffs = np.linalg.norm(
        joints_seq[1:] - joints_seq[:-1],
        axis=2
    )  # shape: (T-1, J)

    jitter_per_frame = diffs.mean(axis=1)  # 每帧平均 jitter

    jitter_mean = float(jitter_per_frame.mean())
    jitter_max  = float(jitter_per_frame.max())
    jitter_std  = float(jitter_per_frame.std())

    # failure rate
    failure_rate = fail_count / total_frames

    # 长序列退化
    T = len(jitter_per_frame)
    early = jitter_per_frame[: T // 3].mean()
    mid   = jitter_per_frame[T // 3 : 2 * T // 3].mean()
    late  = jitter_per_frame[2 * T // 3 :].mean()

    # 汇总
    all_metrics[view] = {
        "jitter_mean": jitter_mean,
        "jitter_max": jitter_max,
        "jitter_std": jitter_std,
        "failure_rate": failure_rate,
        "jitter_early": float(early),
        "jitter_mid": float(mid),
        "jitter_late": float(late),
        "avg_inference_time": float(np.mean(inference_time)),
    }

    print("\n===== SAM3D Stability Report =====")
    for view, m in all_metrics.items():
        print(f"\nView: {view}")
        for k, v in m.items():
            print(f"  {k:20s}: {v:.4f}")


