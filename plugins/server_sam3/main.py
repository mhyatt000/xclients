from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

import torch
from PIL import Image
from pydantic import Field, TypeAdapter
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
from webpolicy.base_policy import BasePolicy
from webpolicy.server import Server


@dataclass
class Config:
    host: str = "0.0.0.0"
    port: int = 8080
    device: str | None = None


@dataclass
class ImageRequest:
    type: Literal["image"]
    image_path: Path
    text: str


@dataclass
class VideoRequest:
    type: Literal["video"]
    resource_path: Path
    text: str
    frame_index: int = 0


@dataclass
class StartStreamRequest:
    type: Literal["start_stream"]
    text: str


@dataclass
class StreamFrameRequest:
    type: Literal["stream_frame"]
    session_id: str
    frame_path: Path
    reverse: bool = False


RequestPayload = Annotated[
    ImageRequest | VideoRequest | StartStreamRequest | StreamFrameRequest,
    Field(discriminator="type"),
]


class Sam3Policy(BasePolicy):
    def __init__(self, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self.model = build_sam3_image_model().to(self.device)
        self.model.eval()
        self.processor = Sam3Processor(self.model)
        self.video_predictor = build_sam3_video_predictor()

        self._adapter = TypeAdapter(RequestPayload)
        self._streaming_sessions: dict[str, object] = {}

    def reset(self, *args, **kwargs) -> None:  # noqa: ANN001, D401
        """Clear any cached streaming sessions."""

        self._streaming_sessions.clear()

    def step(self, obs: dict) -> dict:  # noqa: D401
        """Handle an image, batched video, or streaming request."""

        request = self._adapter.validate_python(obs)
        if isinstance(request, ImageRequest):
            return self._run_image(request)
        if isinstance(request, VideoRequest):
            return self._run_video_predictor(request)
        if isinstance(request, StartStreamRequest):
            return self._start_stream_session(request)
        return self._stream_frame(request)

    def _run_image(self, request: ImageRequest) -> dict:
        image = Image.open(request.image_path).convert("RGB")
        state = self.processor.set_image(image)
        outputs = self.processor.set_text_prompt(state=state, prompt=request.text)

        masks = outputs["masks"].detach().cpu().numpy().tolist()
        boxes = outputs["boxes"].detach().cpu().tolist()
        scores = outputs["scores"].detach().cpu().tolist()

        return {"masks": masks, "boxes": boxes, "scores": scores}

    def _run_video_predictor(self, request: VideoRequest) -> dict:
        session = self.video_predictor.handle_request(
            {"type": "start_session", "resource_path": str(request.resource_path)}
        )
        response = self.video_predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session["session_id"],
                "frame_index": request.frame_index,
                "text": request.text,
            }
        )
        return {"session_id": session["session_id"], "outputs": response.get("outputs", {})}

    def _start_stream_session(self, request: StartStreamRequest) -> dict:
        inference_session = self.processor.init_video_session(
            inference_device=self.device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=self.dtype,
        )
        inference_session = self.processor.add_text_prompt(
            inference_session=inference_session, text=request.text
        )

        session_id = str(uuid4())
        self._streaming_sessions[session_id] = inference_session
        return {"session_id": session_id}

    def _stream_frame(self, request: StreamFrameRequest) -> dict:
        inference_session = self._streaming_sessions[request.session_id]

        frame = Image.open(request.frame_path).convert("RGB")
        inputs = self.processor(images=frame, device=self.device, return_tensors="pt")

        model_outputs = self.model(
            inference_session=inference_session,
            frame=inputs.pixel_values[0],
            reverse=request.reverse,
        )
        processed = self.processor.postprocess_outputs(
            inference_session, model_outputs, original_sizes=inputs.original_sizes
        )

        boxes = processed.get("boxes")
        masks = processed.get("masks")
        object_ids = processed.get("object_ids")

        return {
            "boxes": boxes.detach().cpu().tolist() if boxes is not None else None,
            "masks": masks.detach().cpu().numpy().tolist() if masks is not None else None,
            "object_ids": object_ids.detach().cpu().tolist() if object_ids is not None else None,
        }


def main(cfg: Config) -> None:
    policy = Sam3Policy(cfg.device)
    server = Server(policy, cfg.host, cfg.port)
    server.serve()


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Config))
