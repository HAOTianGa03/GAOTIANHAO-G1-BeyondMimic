#!/usr/bin/env python3
"""Visualize BeyondMimic npz motion files as 3D skeleton animation → MP4.

不依赖 Isaac Sim，纯 Python + matplotlib + ffmpeg，可 headless 运行。

Usage:
    # 可视化单个 npz 并导出 mp4
    python scripts/beyond_mimic/visualize_npz.py \
        --motion_file source/whole_body_tracking/data/motion_npz/Kobe_level1/motion.npz

    # 指定输出路径和 FPS
    python scripts/beyond_mimic/visualize_npz.py \
        --motion_file source/whole_body_tracking/data/motion_npz/walk_level1/motion.npz \
        --output /tmp/walk_level1.mp4 \
        --playback_fps 25

    # 批量可视化目录下所有 npz
    python scripts/beyond_mimic/visualize_npz.py \
        --motion_dir source/whole_body_tracking/data/motion_npz/
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless backend, no display needed
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ---------------------------------------------------------------------------
# G1 body index → name (30 bodies in Isaac Sim order)
# Obtained from kinematic replay, the URDF body order is:
# ---------------------------------------------------------------------------
G1_BODY_NAMES = [
    "pelvis",                    # 0
    "left_hip_pitch_link",       # 1
    "left_hip_roll_link",        # 2
    "left_hip_yaw_link",         # 3
    "left_knee_link",            # 4
    "left_ankle_pitch_link",     # 5
    "left_ankle_roll_link",      # 6
    "right_hip_pitch_link",      # 7
    "right_hip_roll_link",       # 8
    "right_hip_yaw_link",        # 9
    "right_knee_link",           # 10
    "right_ankle_pitch_link",    # 11
    "right_ankle_roll_link",     # 12
    "waist_yaw_link",            # 13
    "waist_roll_link",           # 14
    "torso_link",                # 15
    "left_shoulder_pitch_link",  # 16
    "left_shoulder_roll_link",   # 17
    "left_shoulder_yaw_link",    # 18
    "left_elbow_link",           # 19
    "left_wrist_roll_link",      # 20
    "left_wrist_pitch_link",     # 21
    "left_wrist_yaw_link",       # 22
    "right_shoulder_pitch_link", # 23
    "right_shoulder_roll_link",  # 24
    "right_shoulder_yaw_link",   # 25
    "right_elbow_link",          # 26
    "right_wrist_roll_link",     # 27
    "right_wrist_pitch_link",    # 28
    "right_wrist_yaw_link",      # 29
]

# Skeleton bone connections: list of (parent_body_idx, child_body_idx)
# Based on URDF parent-child relationships
SKELETON_BONES = [
    # Left leg:  pelvis → hip_pitch → hip_roll → hip_yaw → knee → ankle_pitch → ankle_roll
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
    # Right leg: pelvis → hip_pitch → hip_roll → hip_yaw → knee → ankle_pitch → ankle_roll
    (0, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12),
    # Torso:     pelvis → waist_yaw → waist_roll → torso
    (0, 13), (13, 14), (14, 15),
    # Left arm:  torso → shoulder_pitch → shoulder_roll → shoulder_yaw → elbow → wrist_roll → wrist_pitch → wrist_yaw
    (15, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22),
    # Right arm: torso → shoulder_pitch → shoulder_roll → shoulder_yaw → elbow → wrist_roll → wrist_pitch → wrist_yaw
    (15, 23), (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29),
]

# Color scheme: different colors for body parts
BONE_COLORS = {}
for b in [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6)]:
    BONE_COLORS[b] = "#2196F3"   # left leg: blue
for b in [(0,7),(7,8),(8,9),(9,10),(10,11),(11,12)]:
    BONE_COLORS[b] = "#4CAF50"   # right leg: green
for b in [(0,13),(13,14),(14,15)]:
    BONE_COLORS[b] = "#FF9800"   # torso: orange
for b in [(15,16),(16,17),(17,18),(18,19),(19,20),(20,21),(21,22)]:
    BONE_COLORS[b] = "#E91E63"   # left arm: pink
for b in [(15,23),(23,24),(24,25),(25,26),(26,27),(27,28),(28,29)]:
    BONE_COLORS[b] = "#9C27B0"   # right arm: purple


def load_motion(motion_file: str) -> dict:
    """Load an npz motion file and return its data."""
    assert os.path.isfile(motion_file), f"File not found: {motion_file}"
    data = np.load(motion_file)
    fps = int(data["fps"])
    body_pos = data["body_pos_w"]   # (T, N_bodies, 3)
    body_quat = data["body_quat_w"] # (T, N_bodies, 4) wxyz
    T, N, _ = body_pos.shape
    duration = T / fps
    print(f"Loaded: {motion_file}")
    print(f"  frames={T}, bodies={N}, fps={fps}, duration={duration:.2f}s")
    return {
        "fps": fps,
        "body_pos": body_pos,
        "body_quat": body_quat,
        "T": T,
        "N": N,
        "duration": duration,
        "filename": os.path.basename(os.path.dirname(motion_file)),
    }


def render_motion_to_mp4(
    motion: dict,
    output_path: str,
    playback_fps: int = 25,
    figsize: tuple = (10, 8),
    elev: float = 15.0,
    azim: float = 45.0,
    dpi: int = 100,
):
    """Render a 3D skeleton animation from body_pos_w and save as mp4."""
    body_pos = motion["body_pos"]  # (T, N, 3)
    fps = motion["fps"]
    T = motion["T"]
    name = motion["filename"]

    # Subsample frames to match playback_fps
    step = max(1, fps // playback_fps)
    frame_indices = list(range(0, T, step))
    n_frames = len(frame_indices)

    # Compute scene bounds (with some padding)
    all_pos = body_pos[frame_indices]  # (n_frames, N, 3)
    x_min, x_max = all_pos[:, :, 0].min() - 0.3, all_pos[:, :, 0].max() + 0.3
    y_min, y_max = all_pos[:, :, 1].min() - 0.3, all_pos[:, :, 1].max() + 0.3
    z_min, z_max = -0.05, all_pos[:, :, 2].max() + 0.3

    # Make axes equal aspect ratio
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2
    mid_x = (x_max + x_min) / 2
    mid_y = (y_max + y_min) / 2
    mid_z = (z_max + z_min) / 2

    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    def _draw_frame(fi):
        ax.cla()
        idx = frame_indices[fi]
        pos = body_pos[idx]  # (N, 3)
        t = idx / fps

        # Draw ground plane
        ax.plot_surface(
            np.array([[x_min, x_max], [x_min, x_max]]),
            np.array([[y_min, y_min], [y_max, y_max]]),
            np.array([[0, 0], [0, 0]]),
            alpha=0.1, color="gray",
        )

        # Draw bones
        for (p, c) in SKELETON_BONES:
            if p < pos.shape[0] and c < pos.shape[0]:
                xs = [pos[p, 0], pos[c, 0]]
                ys = [pos[p, 1], pos[c, 1]]
                zs = [pos[p, 2], pos[c, 2]]
                color = BONE_COLORS.get((p, c), "#333333")
                ax.plot(xs, ys, zs, color=color, linewidth=2.5, solid_capstyle="round")

        # Draw joints as dots
        ax.scatter(
            pos[:, 0], pos[:, 1], pos[:, 2],
            c="black", s=15, depthshade=True, zorder=5,
        )

        # Highlight key joints
        for ki, kc, ks in [(0, "red", 40), (6, "#2196F3", 30), (12, "#4CAF50", 30)]:
            if ki < pos.shape[0]:
                ax.scatter([pos[ki, 0]], [pos[ki, 1]], [pos[ki, 2]], c=kc, s=ks, zorder=10)

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(z_min, z_min + 2 * max_range)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"{name}  t={t:.2f}s  (frame {idx}/{T})", fontsize=14)
        ax.view_init(elev=elev, azim=azim + fi * 0.3)  # slow rotation

    print(f"Rendering {n_frames} frames → {output_path} ...")
    anim = animation.FuncAnimation(fig, _draw_frame, frames=n_frames, interval=1000 / playback_fps)

    writer = animation.FFMpegWriter(fps=playback_fps, bitrate=2000)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)

    file_size = os.path.getsize(output_path) / 1024
    print(f"Saved: {output_path} ({file_size:.0f} KB, {n_frames} frames @ {playback_fps} fps)")


def main():
    parser = argparse.ArgumentParser(description="Visualize BeyondMimic npz as 3D skeleton MP4 (no Isaac Sim).")
    parser.add_argument("--motion_file", type=str, default=None, help="Path to a single motion.npz file.")
    parser.add_argument("--motion_dir", type=str, default=None, help="Directory of motion_npz/ subfolders to batch-render.")
    parser.add_argument("--output", type=str, default=None, help="Output mp4 path (for single file mode).")
    parser.add_argument("--output_dir", type=str, default="/tmp/motion_videos", help="Output directory (for batch mode).")
    parser.add_argument("--playback_fps", type=int, default=25, help="Playback FPS of the output video (default: 25).")
    parser.add_argument("--dpi", type=int, default=100, help="DPI for rendering (default: 100).")
    args = parser.parse_args()

    if args.motion_file is None and args.motion_dir is None:
        parser.error("Must provide either --motion_file or --motion_dir.")

    if args.motion_file is not None:
        # Single file mode
        motion = load_motion(args.motion_file)
        if args.output is None:
            name = motion["filename"]
            args.output = f"/tmp/motion_videos/{name}.mp4"
        render_motion_to_mp4(motion, args.output, playback_fps=args.playback_fps, dpi=args.dpi)
    else:
        # Batch mode
        subdirs = sorted(
            d for d in os.listdir(args.motion_dir)
            if os.path.isfile(os.path.join(args.motion_dir, d, "motion.npz"))
        )
        print(f"Found {len(subdirs)} motions in {args.motion_dir}")
        for i, d in enumerate(subdirs):
            npz_path = os.path.join(args.motion_dir, d, "motion.npz")
            out_path = os.path.join(args.output_dir, f"{d}.mp4")
            if os.path.isfile(out_path):
                print(f"[{i+1}/{len(subdirs)}] SKIP {d} (already exists)")
                continue
            print(f"[{i+1}/{len(subdirs)}] {d}")
            motion = load_motion(npz_path)
            render_motion_to_mp4(motion, out_path, playback_fps=args.playback_fps, dpi=args.dpi)
        print(f"=== All {len(subdirs)} videos saved to {args.output_dir} ===")


if __name__ == "__main__":
    main()
