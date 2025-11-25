# SAM3 WebPolicy server

A minimal webpolicy server that exposes SAM3 for single-image prompting or streaming video.

## Usage

```bash
python plugins/server_sam3do/main.py --host 0.0.0.0 --port 8080 --device cuda
```

### Request shapes
- **Image prompt**
  ```json
  {
    "type": "image",
    "image_path": "/path/to/image.jpg",
    "text": "person"
  }
  ```
- **One-shot video** (runs a prompt through the bundled video predictor)
  ```json
  {
    "type": "video",
    "resource_path": "/path/to/video_or_frame_dir",
    "frame_index": 0,
    "text": "cat"
  }
  ```
- **Streaming video**
  1. Start a streaming session to cache the text prompt:
     ```json
     {"type": "start_stream", "text": "bicycle"}
     ```
  2. Send frames one by one:
     ```json
     {
       "type": "stream_frame",
       "session_id": "<returned_id>",
       "frame_path": "/path/to/frame.png",
       "reverse": false
     }
     ```

See `plugins/server_sam3do/main.py` for the full policy logic.
