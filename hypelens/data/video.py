"""Video keyframe extraction helper for Instagram Reels and short-form video clips.

Uses headless OpenCV to sample timestamped keyframes with sensible limits,
graceful error handling for missing/corrupted files, and optional downsampling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ExtractedFrame:
    """Individual extracted video keyframe."""
    frame_index: int
    timestamp_sec: float
    frame_rgb: np.ndarray
    width: int
    height: int
    saved_path: Optional[str] = None


@dataclass
class VideoExtractionResult:
    """Outcome of video frame extraction process."""
    success: bool
    video_path: str
    total_frames: int = 0
    duration_sec: float = 0.0
    fps: float = 0.0
    frames: List[ExtractedFrame] = field(default_factory=list)
    error: Optional[str] = None


def extract_keyframes(
    video_path: str | Path,
    max_frames: int = 5,
    output_dir: Optional[str | Path] = None,
    max_dimension: int = 720,
) -> VideoExtractionResult:
    """Extract evenly spaced keyframes from a video clip.

    Args:
        video_path: Path to video file.
        max_frames: Maximum number of keyframes to extract (1 to 30).
        output_dir: Optional directory to save extracted frames as PNGs.
        max_dimension: Maximum width or height for extracted frames (resizes if exceeded).

    Returns:
        VideoExtractionResult containing extracted frame arrays and metadata.
    """
    path_str = str(video_path)
    if not os.path.exists(path_str):
        return VideoExtractionResult(
            success=False,
            video_path=path_str,
            error=f"Video file does not exist: {path_str}",
        )

    # Enforce bounds on max_frames
    max_frames = max(1, min(max_frames, 30))

    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        return VideoExtractionResult(
            success=False,
            video_path=path_str,
            error=f"Failed to open video file (unsupported or corrupted format): {path_str}",
        )

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or total_frames <= 0:
            return VideoExtractionResult(
                success=False,
                video_path=path_str,
                error="Video file contains no readable frames or invalid framerate.",
            )

        duration_sec = total_frames / fps

        # Compute sampling frame indices (evenly distributed)
        step = max(1, total_frames // max_frames)
        sample_indices = [min(i * step, total_frames - 1) for i in range(max_frames)]
        # Deduplicate while preserving order
        sample_indices = sorted(list(set(sample_indices)))

        extracted: List[ExtractedFrame] = []
        out_path_dir = Path(output_dir) if output_dir else None
        if out_path_dir:
            out_path_dir.mkdir(parents=True, exist_ok=True)

        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                continue

            h, w = frame_bgr.shape[:2]
            # Downsample if exceeding max_dimension
            if max(h, w) > max_dimension:
                scale = max_dimension / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
                h, w = new_h, new_w

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            ts_sec = round(idx / fps, 3)

            saved_path_str: Optional[str] = None
            if out_path_dir:
                stem = Path(path_str).stem
                dest = out_path_dir / f"{stem}_frame_{idx:04d}.png"
                cv2.imwrite(str(dest), frame_bgr)
                saved_path_str = str(dest)

            extracted.append(
                ExtractedFrame(
                    frame_index=idx,
                    timestamp_sec=ts_sec,
                    frame_rgb=frame_rgb,
                    width=w,
                    height=h,
                    saved_path=saved_path_str,
                )
            )

        if not extracted:
            return VideoExtractionResult(
                success=False,
                video_path=path_str,
                total_frames=total_frames,
                duration_sec=duration_sec,
                fps=fps,
                error="Could not decode any valid frames from stream.",
            )

        return VideoExtractionResult(
            success=True,
            video_path=path_str,
            total_frames=total_frames,
            duration_sec=duration_sec,
            fps=fps,
            frames=extracted,
        )

    except Exception as e:
        return VideoExtractionResult(
            success=False,
            video_path=path_str,
            error=f"Unexpected error during frame extraction: {e}",
        )
    finally:
        cap.release()
