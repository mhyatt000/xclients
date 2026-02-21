from __future__ import annotations

# from huggingface_hub.utils import HF_HUB_CACHE
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

import numpy as np
import torch

# from transformers import AutoProcessor
# from transformers import Sam3Processor, Sam3Model
import tyro
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
from webpolicy.base_policy import BasePolicy
from webpolicy.server import Server


@dataclass
class Config:
    host: str = "0.0.0.0"
    port: int = 8080
    device: str | None = None
    confidence: float = 0.5


class Schema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ImageRequest(Schema):
    type: Literal["image"]
    image: np.ndarray
    text: str
    confidence: float = 0.5


class VideoRequest(Schema):
    type: Literal["video"]
    resource_path: Path
    text: str
    frame_index: int = 0


class StartStreamRequest(Schema):
    type: Literal["start_stream"]
    text: str


class StreamFrameRequest(Schema):
    type: Literal["stream_frame"]
    session_id: str
    frame_path: Path
    reverse: bool = False


RequestPayload = Annotated[
    ImageRequest | VideoRequest | StartStreamRequest | StreamFrameRequest,
    Field(discriminator="type"),
]


class Sam3Policy(BasePolicy):
    def __init__(self, device: str | None = None, confidence: float = 0.5):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self.model = build_sam3_image_model(
            compile=True,
        ).to(self.device)
        self.model.eval()
        self.confidence = confidence
        self.processor = Sam3Processor(self.model, confidence_threshold=self.confidence)

        # self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
        # self.processor = Sam3Processor.from_pretrained("facebook/sam3")

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
        if self.confidence != request.confidence:
            self.confidence = request.confidence
            self.processor = Sam3Processor(self.model, confidence_threshold=self.confidence)

        img_pil = Image.fromarray(request.image, mode="RGB")
        state = self.processor.set_image(img_pil)
        outputs = self.processor.set_text_prompt(state=state, prompt=request.text)

        # inputs = self.processor(images=request.image, text=request.text, return_tensors="pt").to(self.device)
        # with torch.no_grad():
        # outputs = self.model(**inputs)
        # results = self.processor.post_process_instance_segmentation(
        # outputs,
        # threshold=0.5,
        # mask_threshold=0.5,
        # target_sizes=inputs.get("original_sizes").tolist()
        # )[0]

        masks = outputs["masks"].detach().cpu().numpy()
        boxes = outputs["boxes"].detach().cpu().half().numpy()
        scores = outputs["scores"].detach().cpu().half().numpy()

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
    policy = Sam3Policy(device=cfg.device, confidence=cfg.confidence)
    server = Server(policy, cfg.host, cfg.port)
    server.serve()


if __name__ == "__main__":
    main(tyro.cli(Config))
