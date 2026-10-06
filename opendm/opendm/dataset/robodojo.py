"""RoboDojo-Sim dataset registration."""

from opendm.constants.robot import ROBOT_STATE_DESCS, RobotType
from opendm.dataset.register import register_dataset

register_dataset(
    {
        "sim_cover_blocks": {
            "jsonl_dir": "./data/robodojo_sim/jsonl/cover_blocks",
            "image_dir": "./data/robodojo_sim/video",
            "image_keys": ["images_1", "images_2", "images_3"],
            "image_prompts": ["Head", "Left wrist", "Right wrist"],
            "robot_type": RobotType.ALOHA,
            "state_desc": ROBOT_STATE_DESCS[RobotType.ALOHA],
            "fps": 25,
            "speed": "0.5",
        },
    },
    prefix="robodojo",
)

# The public RoboDojo-Sim archive has per-task JSONL/video directories.
# Point at the shared extracted archive and index every available task.
import os

_all_root = os.environ.get("ROBODOJO_DM05_DATA_ROOT", "./data/robodojo_sim")
register_dataset(
    {
        "sim_all": {
            "jsonl_dir": os.path.join(_all_root, "jsonl"),
            "image_dir": os.path.join(_all_root, "video"),
            "image_keys": ["images_1", "images_2", "images_3"],
            "image_prompts": ["Head", "Left wrist", "Right wrist"],
            "robot_type": RobotType.ALOHA,
            "state_desc": ROBOT_STATE_DESCS[RobotType.ALOHA],
            "fps": 25,
            "speed": "0.5",
        },
    },
    prefix="robodojo",
)
