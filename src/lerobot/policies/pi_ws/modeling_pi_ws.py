from collections import deque
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch
from torch import Tensor

from lerobot.optim.optimizers import OptimizerParams
from lerobot.policies.chunk import ChunkPolicySpec, ExecutionMode
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.robots.bi_piperx_follower.config_bi_piperx_follower import ACTION_KEYS
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.import_utils import _openpi_client_available, _websockets_available

from .configuration_pi_ws import PiWSConfig

if TYPE_CHECKING:
    from websockets.sync.client import ClientConnection


class PiWSPolicy(PreTrainedPolicy):
    """Stateless D4RT requests with rollout-owned training-time RTC scheduling."""

    name = "pi_ws"
    config_class = PiWSConfig

    def __init__(self, config: PiWSConfig) -> None:
        super().__init__(config)
        self.config = config
        self._connection: ClientConnection | None = None
        self._camera_map: dict[str, str] = {}
        self._action_queue: deque[Tensor] = deque()

    def _validate_metadata(self, metadata: Any) -> dict[str, str]:
        if not isinstance(metadata, dict):
            raise ValueError("Expected D4RT server metadata.")
        metadata = metadata.get("metadata", metadata)
        if not isinstance(metadata, dict):
            raise ValueError("Expected D4RT server metadata.")
        for key, value in {
            "action_dim": len(ACTION_KEYS),
            "action_horizon": self.config.chunk_size,
            "gripper_unit": "radian",
        }.items():
            if metadata.get(key) != value:
                raise ValueError(f"D4RT metadata {key}={metadata.get(key)!r}; expected {value!r}.")
        if metadata.get("action_space") not in (
            "absolute",
            "absolute joint positions (delta-trained, unnormalized on the server)",
        ):
            raise ValueError("D4RT server must return absolute joint position targets.")
        if self.config.rtc_training_max_delay:
            delay = metadata.get("rtc_max_delay")
            if (
                metadata.get("backend", "native") != "native"
                or metadata.get("rtc_enabled") is not True
                or type(delay) is not int
                or not self.config.rtc_training_max_delay <= delay < self.config.chunk_size
            ):
                raise ValueError("D4RT server RTC metadata does not cover rtc_training_max_delay.")
        cameras = metadata.get("required_cameras", metadata.get("camera_names"))
        sources = {target: source for source, target in self.config.camera_name_map.items()}
        if (
            not isinstance(cameras, (list, tuple))
            or not cameras
            or len(set(cameras)) != len(cameras)
            or set(cameras) - sources.keys()
            or tuple(metadata.get("camera_names", cameras)) != tuple(cameras)
        ):
            raise ValueError(f"Invalid D4RT camera metadata: {cameras!r}.")
        return {sources[camera]: camera for camera in cameras}

    def connect(self) -> None:
        if self._connection is not None:
            return
        if not (_openpi_client_available and _websockets_available):
            raise ImportError(
                "Install OpenPI Client and websockets using the Split-ALOHA setup in README.md."
            )
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect

        timeout = self.config.request_timeout_s
        connection = connect(self.config.endpoint, open_timeout=timeout, close_timeout=1, compression=None)
        try:
            self._camera_map = self._validate_metadata(msgpack_numpy.unpackb(connection.recv(timeout)))
        except BaseException:
            connection.close()
            raise
        self._connection = connection

    def close(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            connection.close()

    def supports_rtc(self) -> bool:
        return self.config.rtc_training_max_delay > 0

    def chunk_inference_spec(self) -> ChunkPolicySpec:
        spec = super().chunk_inference_spec()
        return replace(spec, modes=tuple(mode for mode in spec.modes if mode is not ExecutionMode.RTC_GUIDED))

    def reset(self) -> None:
        self._action_queue.clear()

    def get_optim_params(self) -> OptimizerParams:
        raise NotImplementedError("PiWS is an inference client.")

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise NotImplementedError("PiWS is an inference client.")

    @staticmethod
    def _encode_image(value: Tensor) -> np.ndarray:
        rgb = value.detach().cpu().numpy()
        if rgb.ndim != 4 or rgb.shape[:2] != (1, 3) or not np.isfinite(rgb).all():
            raise ValueError("D4RT images must be finite single-batch RGB tensors in BCHW order.")
        rgb = rgb[0].transpose(1, 2, 0)
        if rgb.dtype != np.uint8:
            rgb = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        height, width = rgb.shape[:2]
        scale = 224 / max(height, width)
        h, w = max(1, round(height * scale)), max(1, round(width * scale))
        padded = np.zeros((224, 224, 3), dtype=np.uint8)
        top, left = (224 - h) // 2, (224 - w) // 2
        padded[top : top + h, left : left + w] = cv2.resize(rgb, (w, h))
        return np.ascontiguousarray(padded.transpose(2, 0, 1))

    def predict_action_chunk(
        self,
        batch: dict[str, Any],
        *,
        inference_delay: int = 0,
        prev_chunk_left_over: Tensor | None = None,
    ) -> Tensor:
        self.connect()
        from openpi_client import msgpack_numpy

        state = batch[OBS_STATE].detach().cpu().numpy().astype(np.float32)
        if state.shape != (1, 14) or not np.isfinite(state).all():
            raise ValueError("D4RT state must contain one finite joint14 observation.")
        prompt = batch.get("task")
        if isinstance(prompt, (list, tuple)) and len(prompt) == 1:
            prompt = prompt[0]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("D4RT requires a task prompt.")
        payload = {
            "state": state[0],
            "prompt": prompt,
            "images": {
                target: self._encode_image(batch[f"observation.images.{source}"])
                for source, target in self._camera_map.items()
            },
        }
        if not 0 <= inference_delay <= self.config.rtc_training_max_delay:
            raise ValueError("D4RT RTC delay exceeds the configured training limit.")
        if inference_delay:
            if prev_chunk_left_over is None or len(prev_chunk_left_over) < inference_delay:
                raise ValueError("D4RT RTC requires a committed joint14 prefix.")
            prefix = prev_chunk_left_over[:inference_delay].detach().cpu().numpy().astype(np.float32)
            if prefix.shape != (inference_delay, 14) or not np.isfinite(prefix).all():
                raise ValueError("D4RT RTC prefix must contain finite absolute joint14 actions.")
            payload.update(action_prefix=prefix, rtc_delay=inference_delay)

        assert self._connection is not None
        try:
            self._connection.send(msgpack_numpy.packb(payload))
            response = self._connection.recv(self.config.request_timeout_s)
            if isinstance(response, str):
                raise RuntimeError(f"D4RT inference failed: {response}")
            decoded = msgpack_numpy.unpackb(response)
            actions = np.asarray(decoded["actions"], dtype=np.float32)
            if actions.shape != (self.config.chunk_size, 14) or not np.isfinite(actions).all():
                raise ValueError("D4RT response must contain a finite absolute joint14 chunk.")
        except Exception:
            self.close()
            raise
        return torch.from_numpy(actions.copy()).unsqueeze(0).to(batch[OBS_STATE].device)

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        if not self._action_queue:
            chunk = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(chunk.unbind(dim=1))
        return self._action_queue.popleft()
