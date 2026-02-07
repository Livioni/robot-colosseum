"""
批量采集所有robot-colosseum任务的腕关节相机数据
参考collect_data.py和colosseum/tools/collect_demo.py
"""
import os
import pickle
import numpy as np
import json
from typing import Dict, Optional, Tuple

from omegaconf import OmegaConf
from PIL import Image
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend import const

# PyRep wrist camera
try:
    from pyrep.objects.vision_sensor import VisionSensor

    PYREP_AVAILABLE = True
except Exception:
    VisionSensor = None
    PYREP_AVAILABLE = False

from colosseum import ASSETS_CONFIGS_FOLDER, TASKS_PY_FOLDER, TASKS_TTM_FOLDER
from colosseum.rlbench.extensions.environment import EnvironmentExt
from colosseum.rlbench.utils import (
    ObservationConfigExt,
    check_and_make,
    name_to_class,
)

OmegaConf.register_new_resolver("eval", eval)

# 相机偏移配置（相对于cam_wrist的父对象坐标系）
CAMERA_OFFSET: Dict[str, float] = {
    "x": 0.06,
    "y": 0.0,
    "z": -0.05,
    "alpha": 0.0,  # deg
    "beta": 0.0,  # deg
    "gamma": 0.0,  # deg
}


class CameraPositionManager:
    """管理相机位置：在整个demo采集过程中保持固定的offset。"""

    def __init__(self, wrist_cam, config: Dict[str, float]):
        self.wrist_cam = wrist_cam
        self.config = config
        self.pyrep_available = PYREP_AVAILABLE
        self.parent = wrist_cam.get_parent() if self.pyrep_available else None
        self.original_pos = None
        self.original_orient = None

    def save_original(self) -> None:
        if not self.pyrep_available:
            return
        self.original_pos = self.wrist_cam.get_position(relative_to=self.parent)
        self.original_orient = self.wrist_cam.get_orientation(
            relative_to=self.parent
        )

    def apply_offset(self) -> Optional[Tuple[list, list]]:
        """应用相机偏移（位置是直接设置为config里给的相对坐标）。"""
        if not self.pyrep_available:
            return None
        if self.original_pos is None or self.original_orient is None:
            self.save_original()
        if self.original_orient is None:
            return None

        new_pos = [self.config["x"], self.config["y"], self.config["z"]]
        new_orient = [
            self.original_orient[0] + np.deg2rad(self.config["alpha"]),
            self.original_orient[1] + np.deg2rad(self.config["beta"]),
            self.original_orient[2] + np.deg2rad(self.config["gamma"]),
        ]
        self.wrist_cam.set_position(new_pos, relative_to=self.parent)
        self.wrist_cam.set_orientation(new_orient, relative_to=self.parent)
        return new_pos, new_orient

    def restore_original(self) -> None:
        if (
            not self.pyrep_available
            or self.original_pos is None
            or self.original_orient is None
        ):
            return
        self.wrist_cam.set_position(self.original_pos, relative_to=self.parent)
        self.wrist_cam.set_orientation(
            self.original_orient, relative_to=self.parent
        )


def save_episode_wrist_data(
    episode_idx: int, demo, save_dir: str, task_name: str
) -> None:
    """
    逐帧保存 wrist RGB、depth(float)、内参、外参

    目录结构:
      {save_dir}/{task_name}/episode_XXXX/{images,depth,pose,intrinsic}/frame_XXXX.*
    """
    episode_dir = os.path.join(save_dir, task_name, f"episode_{episode_idx:04d}")
    images_dir = os.path.join(episode_dir, "images")
    depth_dir = os.path.join(episode_dir, "depth")
    pose_dir = os.path.join(episode_dir, "pose")
    intrinsic_dir = os.path.join(episode_dir, "intrinsic")
    for d in (images_dir, depth_dir, pose_dir, intrinsic_dir):
        os.makedirs(d, exist_ok=True)

    for frame_idx, obs in enumerate(demo):
        # RGB
        rgb = obs.wrist_rgb
        if rgb is not None:
            Image.fromarray(rgb.astype(np.uint8)).save(
                os.path.join(images_dir, f"frame_{frame_idx:04d}.png")
            )

        # Depth (float, in meters if depth_in_meters=True)
        depth = obs.wrist_depth
        if depth is not None:
            np.save(os.path.join(depth_dir, f"frame_{frame_idx:04d}.npy"), depth)

        # Extrinsics / Intrinsics from misc
        extr = None
        intr = None
        if getattr(obs, "misc", None) is not None:
            extr = obs.misc.get("wrist_camera_extrinsics", None)
            intr = obs.misc.get("wrist_camera_intrinsics", None)
        if extr is not None:
            np.save(os.path.join(pose_dir, f"frame_{frame_idx:04d}.npy"), extr)
        if intr is not None:
            np.save(
                os.path.join(intrinsic_dir, f"frame_{frame_idx:04d}.npy"), intr
            )

    metadata = {
        "episode_idx": episode_idx,
        "num_frames": len(demo),
        "task": task_name,
        "camera_offset": CAMERA_OFFSET,
        "pyrep_available": PYREP_AVAILABLE,
    }
    with open(os.path.join(episode_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def collect_single_task(task_name: str, save_path: str, episodes_per_task: int = 10):
    """
    采集单个任务的数据
    """
    # 构造配置文件路径
    config_path = os.path.join(ASSETS_CONFIGS_FOLDER, f"{task_name}.yaml")

    # 加载配置
    cfg = OmegaConf.load(config_path)
    cfg.data.save_path = save_path
    cfg.data.episodes_per_task = episodes_per_task

    check_and_make(cfg.data.save_path)

    np.random.seed(cfg.env.seed)

    data_cfg, env_cfg = cfg.data, cfg.env

    task_class = name_to_class(cfg.env.task_name, TASKS_PY_FOLDER)
    assert (
        task_class is not None
    ), f"Can't get task-class for task {cfg.env.task_name}"

    rlbench_env = EnvironmentExt(
        action_mode=MoveArmThenGripper(
            arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()
        ),
        obs_config=ObservationConfigExt(data_cfg),
        headless=True,
        path_task_ttms=TASKS_TTM_FOLDER,
        env_config=env_cfg,
    )

    rlbench_env.launch()

    # Setup wrist camera pose manager
    cam_manager = None
    if PYREP_AVAILABLE and VisionSensor is not None:
        # try:
        #     wrist_cam = VisionSensor("cam_wrist")
        #     cam_manager = CameraPositionManager(wrist_cam, CAMERA_OFFSET)
        #     # cam_manager.save_original()
        #     # cam_manager.apply_offset()
        # except Exception:
        cam_manager = None

    task_env = rlbench_env.get_task(task_class)

    descriptions, _ = task_env.reset()

    # Save RLBench-style variation descriptions (optional, for traceability)
    task_root = os.path.join(data_cfg.save_path, task_env.get_name())
    check_and_make(task_root)
    with open(os.path.join(task_root, const.VARIATION_DESCRIPTIONS), "wb") as f:
        pickle.dump(descriptions, f)

    collected_episodes = 0
    for ex_idx in range(data_cfg.episodes_per_task):
        print(f"Task: {task_env.get_name()} // Demo: {ex_idx}")

        attempts = 10
        demo = None
        while attempts > 0:
            try:
                (demo,) = task_env.get_demos(
                    amount=1, live_demos=True,
                )
                break
            except Exception:
                attempts -= 1
                if attempts <= 0:
                    print(f"Failed with task {task_env.get_name()}, sample: {ex_idx}")
        if demo is None:
            continue

        # 保存 wrist 数据（RGB/Depth/内外参）
        save_episode_wrist_data(ex_idx, demo, data_cfg.save_path, task_env.get_name())
        collected_episodes += 1

    if cam_manager is not None:
        cam_manager.restore_original()

    rlbench_env.shutdown()

    return collected_episodes


def main():
    # Configuration
    episodes_per_task = 100  # Number of episodes to collect per task
    # 建议用单独目录保存 wrist 的 RGB/Depth/内外参
    save_dir = "datasets/colosseum_wrist_data"
    os.makedirs(save_dir, exist_ok=True)

    print("="*60)
    print("批量数据采集配置:")
    print(f"每个任务采集 {episodes_per_task} 个episodes")
    print(f"保存目录: {save_dir}")
    print("="*60)

    # 任务列表（对应配置文件名）
    task_names = [
        "basketball_in_hoop",
        "close_box",
        "close_laptop_lid",
        "empty_dishwasher",
        "get_ice_from_fridge",
        "hockey",
        "insert_onto_square_peg",
        "meat_on_grill",
        "move_hanger",
        "open_drawer",
        "place_wine_at_rack_location",
        "put_money_in_safe",
        "reach_and_drag",
        "scoop_with_spatula",
        "setup_chess",
        "slide_block_to_target",
        "stack_cups",
        "straighten_rope",
        "turn_oven_on",
        "wipe_desk"
    ]

    total_tasks = len(task_names)
    total_episodes = total_tasks * episodes_per_task

    print(f"\n开始批量数据采集...")
    print(f"任务数: {total_tasks}")
    print(f"预计总episodes数: {total_episodes}")
    print(f"保存目录: {os.path.abspath(save_dir)}\n")
    print(f"PyRep可用: {PYREP_AVAILABLE}")
    print(f"CAMERA_OFFSET: {CAMERA_OFFSET}")

    task_success_count = 0
    total_episodes_collected = 0

    # 遍历所有任务
    for task_idx, task_name in enumerate(task_names):
        print(f"\n{'='*80}")
        print(f"任务 {task_idx + 1}/{total_tasks}: {task_name}")
        print(f"{'='*80}")

        try:
            collected_episodes = collect_single_task(task_name, save_dir, episodes_per_task)
            total_episodes_collected += collected_episodes

            if collected_episodes > 0:
                task_success_count += 1
                print(f"\n✓ 任务 {task_name} 完成: {collected_episodes}/{episodes_per_task} episodes")
            else:
                print(f"\n✗ 任务 {task_name} 失败: 0/{episodes_per_task} episodes")

        except Exception as e:
            print(f"✗ 任务 {task_name} 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print(f"批量数据采集完成!")
    print(f"成功任务数: {task_success_count}/{total_tasks}")
    print(f"总采集episodes数: {total_episodes_collected}/{total_episodes}")
    print(f"数据保存目录: {os.path.abspath(save_dir)}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
