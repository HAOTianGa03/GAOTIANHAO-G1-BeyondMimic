"""Replay a BeyondMimic npz motion in Isaac Sim (kinematic playback).

Supports both GUI mode and headless video recording.

.. code-block:: bash

    # GUI 回放（需要显示器）
    /isaac-sim/python.sh scripts/beyond_mimic/replay_npz.py \
        --motion_file source/whole_body_tracking/data/motion_npz/Kobe_level1/motion.npz

    # 无头录制 MP4（不需要显示器，适合远程服务器）
    /isaac-sim/python.sh scripts/beyond_mimic/replay_npz.py \
        --motion_file source/whole_body_tracking/data/motion_npz/Kobe_level1/motion.npz \
        --video --video_output /tmp/Kobe_level1.mp4 \
        --headless --enable_cameras

    # 从 WandB Registry 加载
    /isaac-sim/python.sh scripts/beyond_mimic/replay_npz.py \
        --registry_name {org}-org/wandb-registry-motions/{name}
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument(
    "--registry_name", type=str, default=None,
    help="WandB registry name, e.g. {org}-org/wandb-registry-motions/{name}.",
)
parser.add_argument(
    "--motion_file", type=str, default=None,
    help="Path to a local motion.npz file (alternative to --registry_name).",
)
parser.add_argument(
    "--video", action="store_true", default=False,
    help="Record an MP4 video of the replay (requires --headless --enable_cameras).",
)
parser.add_argument(
    "--video_output", type=str, default=None,
    help="Output path for the MP4 video. Default: /tmp/{motion_name}.mp4.",
)
parser.add_argument(
    "--video_fps", type=int, default=30,
    help="FPS of the output video (default: 30).",
)
parser.add_argument(
    "--video_resolution",
    nargs=2, type=int, metavar=("W", "H"), default=[1280, 720],
    help="Video resolution as W H (default: 1280 720).",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# If recording video in headless mode, need to enable cameras for offscreen rendering
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.mdp import MotionLoader


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _resolve_motion_file() -> str:
    """Resolve the motion file path from CLI arguments."""
    if args_cli.motion_file is not None:
        return args_cli.motion_file
    elif args_cli.registry_name is not None:
        registry_name = args_cli.registry_name
        if ":" not in registry_name:
            registry_name += ":latest"
        import pathlib
        import wandb
        api = wandb.Api()
        artifact = api.artifact(registry_name)
        return str(pathlib.Path(artifact.download()) / "motion.npz")
    else:
        raise ValueError("Must provide either --motion_file or --registry_name.")


def _setup_video_recorder(sim: SimulationContext, resolution: tuple):
    """Set up offscreen RGB annotator for video recording."""
    import omni.replicator.core as rep

    # Create render product from the default viewport camera
    render_product = rep.create.render_product("/OmniverseKit_Persp", resolution)
    # Create RGB annotator
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    rgb_annotator.attach([render_product])
    return rgb_annotator


def _open_ffmpeg_pipe(output_path: str, width: int, height: int, fps: int):
    """Open an ffmpeg subprocess pipe for streaming frame-by-frame encoding."""
    import subprocess

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "fast",
        output_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()

    # Load motion
    motion_file = _resolve_motion_file()
    print(f"[INFO] Loading motion: {motion_file}")

    motion = MotionLoader(
        motion_file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    # Video recording setup
    recording = args_cli.video
    rgb_annotator = None
    ffmpeg_proc = None
    frames_written = 0
    total_frames = int(motion.time_step_total)

    if recording:
        resolution = tuple(args_cli.video_resolution)
        rgb_annotator = _setup_video_recorder(sim, resolution)
        print(f"[INFO] Recording {total_frames} frames @ {resolution[0]}x{resolution[1]}...")
        # Warm up renderer (first few frames may be blank)
        for _ in range(5):
            sim.render()

    # Determine output path
    if recording and args_cli.video_output is None:
        # Derive name from motion file path
        motion_name = os.path.basename(os.path.dirname(motion_file))
        if not motion_name or motion_name == ".":
            motion_name = os.path.splitext(os.path.basename(motion_file))[0]
        args_cli.video_output = f"/tmp/{motion_name}.mp4"

    frame_count = 0

    # Simulation loop
    while simulation_app.is_running():
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins[:, None, :]
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
        scene.write_data_to_sim()
        sim.render()  # No physics stepping, just rendering
        scene.update(sim_dt)

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        # Capture frame for video
        if recording:
            rgb_data = rgb_annotator.get_data()
            if rgb_data is not None and rgb_data.size > 0:
                frame = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
                # Lazy-open ffmpeg pipe on first valid frame
                if ffmpeg_proc is None:
                    h, w = rgb_data.shape[:2]
                    ffmpeg_proc = _open_ffmpeg_pipe(
                        args_cli.video_output, w, h, args_cli.video_fps
                    )
                ffmpeg_proc.stdin.write(frame[:, :, :3].astype(np.uint8).tobytes())
                frames_written += 1

            frame_count += 1
            if frame_count % 50 == 0:
                print(f"  Recording: {frame_count}/{total_frames} frames (written={frames_written})")

            # Stop after one full playback
            if frame_count >= total_frames:
                if ffmpeg_proc is not None:
                    ffmpeg_proc.stdin.close()
                    ffmpeg_proc.wait()
                    file_size_kb = os.path.getsize(args_cli.video_output) / 1024
                    w, h = args_cli.video_resolution
                    print(f"[INFO] Saved video → {args_cli.video_output} "
                          f"({frames_written} frames, {w}x{h}, {args_cli.video_fps}fps, {file_size_kb:.0f}KB)")
                else:
                    print("[WARN] No valid frames were captured. Check --enable_cameras and --headless flags.")
                # Force exit (simulation_app.close() blocks in headless mode)
                os._exit(0)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO] Setup complete, starting replay...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
