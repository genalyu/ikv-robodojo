import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_aloha_example() -> dict:
    """Creates a random input example for the Aloha policy."""
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class AlohaInputs(transforms.DataTransformFn):
    """Inputs for the Aloha policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [14]
    - actions: [action_horizon, 14]
    """

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that base image always exists.
        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add the extra images.
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AlohaIKVRGBInputs(AlohaInputs):
    """Aloha/RoboDojo input adapter with chronological head-camera history.

    ``history_images`` contains up to ``max_history_images`` RGB frames in
    chronological order, each CHW or HWC. The current camera views continue
    through the standard Aloha mapping.
    """

    max_history_images: int = 20
    ikv_dino_model_path: str | None = None
    ikv_require_dino: bool = False

    def __call__(self, data: dict) -> dict:
        history_value = data.get("history_images")
        history = list(history_value) if history_value is not None else []
        history = history[-self.max_history_images :]
        # LeRobot conversions differ on whether image features are CHW or HWC.
        # The inherited Aloha decoder expects CHW, so normalize current views
        # without modifying the input object owned by the dataset pipeline.
        normalized = dict(data)
        normalized["images"] = {}
        for name, raw in data["images"].items():
            image = np.asarray(raw)
            if image.ndim != 3:
                raise ValueError("Current RGB views must be CHW or HWC")
            if image.shape[0] != 3 and image.shape[-1] == 3:
                image = einops.rearrange(image, "h w c -> c h w")
            if image.shape[0] != 3:
                raise ValueError("Current RGB views must have three channels")
            if np.issubdtype(image.dtype, np.floating):
                if np.min(image) < 0:
                    image = (image + 1) / 2
                if np.max(image) <= 1:
                    image = image * 255
                image = np.clip(image, 0, 255).astype(np.uint8)
            normalized["images"][name] = image
        output = super().__call__(normalized)
        first_valid = self.max_history_images - len(history)
        actual_images = []
        for index in range(self.max_history_images):
            valid = index >= first_valid
            if valid:
                image = np.asarray(history[index - first_valid])
                if image.ndim != 3:
                    raise ValueError("RGB history frames must be CHW or HWC")
                if image.shape[0] == 3 and image.shape[-1] != 3:
                    image = einops.rearrange(image, "c h w -> h w c")
                if image.shape[-1] != 3:
                    raise ValueError("RGB history frames must have three channels")
                if np.issubdtype(image.dtype, np.floating):
                    if np.min(image) < 0:
                        image = (image + 1) / 2
                    if np.max(image) <= 1:
                        image = image * 255
                    image = np.clip(image, 0, 255).astype(np.uint8)
            else:
                image = np.zeros_like(output["image"]["base_0_rgb"])
            if image.shape != output["image"]["base_0_rgb"].shape:
                raise ValueError("RGB history frame shape must match the head camera")
            key = f"memory_{index:03d}_rgb"
            output["image"][key] = image
            output["image_mask"][key] = np.bool_(valid)
            if valid:
                actual_images.append(image)
        features = data.get("ikv_dino_features")
        reference = data.get("ikv_reference_dino")
        if (features is None or reference is None) and self.ikv_dino_model_path and actual_images:
            from openpi.models.ikv_dino import encode_dino_grid
            from openpi.shared.image_tools import resize_with_pad_torch
            import torch

            aligned = resize_with_pad_torch(
                torch.as_tensor(np.stack([*actual_images, output["image"]["base_0_rgb"]])), 224, 224
            )

            encoded = encode_dino_grid(
                aligned, checkpoint=self.ikv_dino_model_path, grid=(16, 16)
            ).flatten(1, 2).numpy()
            if features is None:
                features = encoded[:-1]
            if reference is None:
                reference = encoded[-1]
        if (features is None or reference is None) and actual_images and self.ikv_require_dino:
            raise ValueError("RGB IKV requires history and current DINO indexes or a local DINOv2 checkpoint")
        if features is not None:
            features = np.asarray(features, dtype=np.float32)
            if features.ndim != 3 or features.shape[1] != 256 or features.shape[0] not in (
                len(actual_images), self.max_history_images
            ):
                raise ValueError("IKV DINO features must be [history_frames,256,feature_dim]")
            if features.shape[0] != self.max_history_images:
                padded = np.zeros((self.max_history_images, *features.shape[1:]), dtype=np.float32)
                padded[first_valid:] = features
                features = padded
            output["ikv_dino_features"] = features
        if reference is not None:
            reference = np.asarray(reference, dtype=np.float32)
            if reference.ndim != 2 or reference.shape[0] != 256:
                raise ValueError("current DINO reference must be [256,feature_dim]")
            if features is not None and reference.shape[-1] != features.shape[-1]:
                raise ValueError("history and current DINO feature widths differ")
            output["ikv_reference_dino"] = reference
        return output


@dataclasses.dataclass(frozen=True)
class AlohaOutputs(transforms.DataTransformFn):
    """Outputs for the Aloha policy."""

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims.
        actions = np.asarray(data["actions"][:, :14])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}


def _joint_flip_mask() -> np.ndarray:
    """Used to convert between aloha and pi joint angles."""
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # pi0 gripper data is normalized (0, 1) between encoder counts (2405, 3110).
    # There are 4096 total encoder counts and aloha uses a zero of 2048.
    # Converting this to radians means that the normalized inputs are between (0.5476, 1.6296)
    return _normalize(value, min_val=0.5476, max_val=1.6296)


def _gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # We do not scale the output since the trossen model predictions are already in radians.
    # See the comment in _gripper_to_angular for a derivation of the constant
    value = value + 0.5476

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return value - 0.5476


def _decode_aloha(data: dict, *, adapt_to_pi: bool = False) -> dict:
    # state is [left_arm_joint_angles, left_arm_gripper, right_arm_joint_angles, right_arm_gripper]
    # dim sizes: [6, 1, 6, 1]
    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi)

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        state = _joint_flip_mask() * state
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        state[[6, 13]] = _gripper_to_angular(state[[6, 13]])
    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])
    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular_inv(actions[:, [6, 13]])
    return actions
