"""Convert ASAP retargeted motion (joblib pkl) to BeyondMimic npz format.

ASAP pkl 格式 (joblib):
  root_trans_offset: (T, 3)   — root 平移，世界坐标
  root_rot:          (T, 4)   — root 旋转，xyzw 四元数
  dof:               (T, 23)  — 关节角度（G1 去掉 6 个手腕关节）
  fps:               int      — 原始帧率（通常 30）

ASAP dof[23] → G1 joint 映射（手腕 6 个关节固定为 0）:
  [0..11]  -> 双腿 12 关节 (hip_pitch/roll/yaw, knee, ankle_pitch/roll) x2
  [12..14] -> 腰部 3 关节 (waist_yaw/roll/pitch)
  [15..18] -> 左臂 4 关节 (shoulder_pitch/roll/yaw, elbow)
  [19..22] -> 右臂 4 关节 (shoulder_pitch/roll/yaw, elbow)
  (left/right wrist_roll/pitch/yaw 共 6 个 -> 0)

输出 npz 格式 (BeyondMimic):
  fps:           scalar
  joint_pos:     (T, 29)
  joint_vel:     (T, 29)
  body_pos_w:    (T, N_bodies, 3)
  body_quat_w:   (T, N_bodies, 4)  wxyz
  body_lin_vel_w:(T, N_bodies, 3)
  body_ang_vel_w:(T, N_bodies, 3)

Usage:
    # 转换单个 pkl（单个 motion key）并上传到 WandB Registry
    /isaac-sim/python.sh scripts/beyond_mimic/asap_to_npz.py \\
        --input_file source/whole_body_tracking/data/ASAP/0-motions_raw_tairantestbed_smpl_video_walk_level1_filter_amass.pkl \\
        --output_name walk_level1 \\
        --output_fps 50 \\
        --headless

    # 仅保存本地 npz，不上传
    /isaac-sim/python.sh scripts/beyond_mimic/asap_to_npz.py \\
        --input_file ... --output_name walk_level1 --no_upload --headless

    # 指定 motion key（pkl 中包含多个 motion 时）
    /isaac-sim/python.sh scripts/beyond_mimic/asap_to_npz.py \\
        --input_file ... --motion_key my_motion_key --output_name my_motion --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert ASAP pkl motion to BeyondMimic npz format.")
parser.add_argument("--input_file", type=str, required=True, help="Path to the ASAP joblib pkl file.")
parser.add_argument(
    "--motion_key",
    type=str,
    default=None,
    help="Key of the motion inside the pkl dict. If None, uses the first key.",
)
parser.add_argument("--output_name", type=str, required=True, help="Output motion name (used as WandB artifact name).")
parser.add_argument("--output_fps", type=int, default=50, help="Target FPS for the output npz (default: 50).")
parser.add_argument(
    "--output_dir",
    type=str,
    default="/tmp",
    help="Local directory to save the npz file before uploading (default: /tmp).",
)
parser.add_argument(
    "--no_upload",
    action="store_true",
    default=False,
    help="Skip WandB upload, only save npz locally.",
)
parser.add_argument(
    "--wandb_project",
    type=str,
    default="asap_to_npz",
    help="WandB project name for upload (default: asap_to_npz).",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG

# ---------------------------------------------------------------------------
# G1 完整 29 关节顺序（与 csv_to_npz.py 保持一致）
# ---------------------------------------------------------------------------
G1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",   # ASAP 中无，固定为 0
    "left_wrist_pitch_joint",  # ASAP 中无，固定为 0
    "left_wrist_yaw_joint",    # ASAP 中无，固定为 0
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",  # ASAP 中无，固定为 0
    "right_wrist_pitch_joint", # ASAP 中无，固定为 0
    "right_wrist_yaw_joint",   # ASAP 中无，固定为 0
]

# ASAP dof[23] 在 G1_JOINT_NAMES[29] 中的索引（手腕 6 个不在其中）
ASAP_DOF_TO_G1_INDEX = [
    0,  1,  2,  3,  4,  5,   # 左腿
    6,  7,  8,  9,  10, 11,  # 右腿
    12, 13, 14,              # 腰部
    15, 16, 17, 18,          # 左臂 (到 elbow)
    22, 23, 24, 25,          # 右臂 (到 elbow)  注意跳过了左手腕 19/20/21
]


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class AsapMotionLoader:
    """Loads an ASAP pkl motion, expands dof to G1 29 joints, interpolates to target FPS."""

    def __init__(
        self,
        motion_file: str,
        motion_key: str | None,
        output_fps: int,
        device: torch.device,
    ):
        self.output_fps = output_fps
        self.output_dt = 1.0 / output_fps
        self.device = device
        self.current_idx = 0
        self._load(motion_file, motion_key)
        self._interpolate()
        self._compute_velocities()

    # ------------------------------------------------------------------
    def _load(self, motion_file: str, motion_key: str | None):
        import joblib

        assert os.path.isfile(motion_file), f"File not found: {motion_file}"
        raw = joblib.load(motion_file)

        if motion_key is None:
            motion_key = list(raw.keys())[0]
            print(f"[INFO] Using first motion key: '{motion_key}'")
        assert motion_key in raw, f"Key '{motion_key}' not found. Available: {list(raw.keys())}"

        m = raw[motion_key]
        input_fps = int(m["fps"])
        self.input_fps = input_fps
        self.input_dt = 1.0 / input_fps

        root_trans = torch.tensor(m["root_trans_offset"], dtype=torch.float32)  # (T, 3)
        root_rot_xyzw = torch.tensor(m["root_rot"], dtype=torch.float32)        # (T, 4) xyzw
        dof_23 = torch.tensor(m["dof"], dtype=torch.float32)                    # (T, 23)

        T = root_trans.shape[0]
        self.input_frames = T
        self.duration = (T - 1) * self.input_dt

        # xyzw -> wxyz
        root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]

        # Expand dof 23 -> 29 (fill wrist joints with 0)
        dof_29 = torch.zeros(T, 29, dtype=torch.float32)
        for asap_idx, g1_idx in enumerate(ASAP_DOF_TO_G1_INDEX):
            dof_29[:, g1_idx] = dof_23[:, asap_idx]

        self.input_root_trans = root_trans.to(self.device)
        self.input_root_rot = root_rot_wxyz.to(self.device)
        self.input_dof = dof_29.to(self.device)

        print(
            f"[INFO] Loaded '{motion_key}': {T} frames @ {input_fps} FPS "
            f"({self.duration:.2f}s), expanding dof 23→29"
        )

    # ------------------------------------------------------------------
    def _interpolate(self):
        times = torch.arange(0, self.duration, self.output_dt, dtype=torch.float32, device=self.device)
        self.output_frames = times.shape[0]

        idx0, idx1, blend = self._frame_blend(times)

        self.root_trans = self._lerp(self.input_root_trans[idx0], self.input_root_trans[idx1], blend.unsqueeze(1))
        self.root_rot = self._slerp(self.input_root_rot[idx0], self.input_root_rot[idx1], blend)
        self.dof_pos = self._lerp(self.input_dof[idx0], self.input_dof[idx1], blend.unsqueeze(1))

        print(
            f"[INFO] Interpolated: {self.input_frames}f@{self.input_fps}fps → "
            f"{self.output_frames}f@{self.output_fps}fps"
        )

    def _frame_blend(self, times: torch.Tensor):
        phase = times / self.duration
        idx0 = (phase * (self.input_frames - 1)).floor().long()
        idx1 = torch.clamp(idx0 + 1, max=self.input_frames - 1)
        blend = phase * (self.input_frames - 1) - idx0.float()
        return idx0, idx1, blend

    def _lerp(self, a, b, blend):
        return a * (1 - blend) + b * blend

    def _slerp(self, a, b, blend):
        """Batched quaternion SLERP. a, b: (N, 4) wxyz, blend: (N,)."""
        # dot product
        dot = (a * b).sum(dim=-1, keepdim=True)  # (N, 1)
        # ensure shortest path
        b = torch.where(dot < 0, -b, b)
        dot = dot.abs()
        dot = dot.clamp(-1.0, 1.0)
        # for near-identical quaternions use linear interp
        linear = a * (1 - blend.unsqueeze(1)) + b * blend.unsqueeze(1)
        linear = linear / linear.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        angle = torch.acos(dot)
        sin_angle = torch.sin(angle).clamp(min=1e-10)
        coeff_a = torch.sin((1 - blend.unsqueeze(1)) * angle) / sin_angle
        coeff_b = torch.sin(blend.unsqueeze(1) * angle) / sin_angle
        slerp = coeff_a * a + coeff_b * b
        slerp = slerp / slerp.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        # use linear interp where angle is near zero
        mask = (dot > 1.0 - 1e-10).squeeze(-1)
        out = torch.where(mask.unsqueeze(1), linear, slerp)
        return out

    # ------------------------------------------------------------------
    def _compute_velocities(self):
        self.root_lin_vel = torch.gradient(self.root_trans, spacing=self.output_dt, dim=0)[0]
        self.dof_vel = torch.gradient(self.dof_pos, spacing=self.output_dt, dim=0)[0]

        q_prev = self.root_rot[:-2]
        q_next = self.root_rot[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * self.output_dt)
        self.root_ang_vel = torch.cat([omega[:1], omega, omega[-1:]], dim=0)

    # ------------------------------------------------------------------
    def get_next_state(self):
        i = self.current_idx
        state = (
            self.root_trans[i : i + 1],
            self.root_rot[i : i + 1],
            self.root_lin_vel[i : i + 1],
            self.root_ang_vel[i : i + 1],
            self.dof_pos[i : i + 1],
            self.dof_vel[i : i + 1],
        )
        self.current_idx += 1
        done = self.current_idx >= self.output_frames
        if done:
            self.current_idx = 0
        return state, done


# ---------------------------------------------------------------------------

def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    """Kinematic replay loop: drive robot joints frame-by-frame, record body states."""

    motion = AsapMotionLoader(
        motion_file=args_cli.input_file,
        motion_key=args_cli.motion_key,
        output_fps=args_cli.output_fps,
        device=sim.device,
    )

    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(G1_JOINT_NAMES, preserve_order=True)[0]

    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    file_saved = False

    while simulation_app.is_running():
        (
            base_pos,
            base_rot,
            base_lin_vel,
            base_ang_vel,
            dof_pos,
            dof_vel,
        ), done = motion.get_next_state()

        # --- set root state ---
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] = base_pos
        root_state[:, :2] += scene.env_origins[:, :2]
        root_state[:, 3:7] = base_rot
        root_state[:, 7:10] = base_lin_vel
        root_state[:, 10:] = base_ang_vel
        robot.write_root_state_to_sim(root_state)

        # --- set joint state ---
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = dof_pos
        joint_vel[:, robot_joint_indexes] = dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.render()
        scene.update(sim.get_physics_dt())

        # camera follow
        sim.set_camera_view(
            base_pos[0].cpu().numpy() + np.array([2.0, 2.0, 0.5]),
            base_pos[0].cpu().numpy(),
        )

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0].cpu().numpy().copy())

        if done and not file_saved:
            file_saved = True
            for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                log[k] = np.stack(log[k], axis=0)

            os.makedirs(args_cli.output_dir, exist_ok=True)
            out_path = os.path.join(args_cli.output_dir, "motion.npz")
            np.savez(out_path, **log)
            print(f"[INFO] Saved npz → {out_path}")
            print(f"       joint_pos:  {log['joint_pos'].shape}")
            print(f"       body_pos_w: {log['body_pos_w'].shape}")

            if not args_cli.no_upload:
                import wandb

                run = wandb.init(project=args_cli.wandb_project, name=args_cli.output_name)
                print(f"[INFO] Uploading to WandB Registry as '{args_cli.output_name}'...")
                REGISTRY = "motions"
                artifact = run.log_artifact(artifact_or_path=out_path, name=args_cli.output_name, type=REGISTRY)
                run.link_artifact(artifact=artifact, target_path=f"wandb-registry-{REGISTRY}/{args_cli.output_name}")
                print(f"[INFO] Uploaded to wandb-registry-{REGISTRY}/{args_cli.output_name}")
            else:
                print("[INFO] Skipping WandB upload (--no_upload).")

            # 强制退出：Isaac Sim headless 模式下 simulation_app.close() 会阻塞，
            # 直接用 os._exit(0) 跳过所有析构强制终止进程。
            print("[INFO] Conversion complete, exiting.")
            os._exit(0)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO] Setup complete, starting kinematic replay...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
