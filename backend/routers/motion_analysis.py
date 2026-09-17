"""
motion_analysis.py — Module 7: Motion & Flow Analysis
Covers Dense Optical Flow (Farneback), Sparse Optical Flow (Lucas-Kanade), and Background Subtraction (MOG2/KNN).
"""
import os
import tempfile
import time
import math
from typing import List, Dict, Any
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from services.cv_utils import encode_image_to_base64, resize_for_display, to_grayscale

router = APIRouter()


@router.post("/motion")
async def analyze_video_motion(
    file: UploadFile = File(...),
    method: str = Form("farneback"),
    frame_step: int = Form(3),
    max_output_frames: int = Form(6),
    # Lucas-Kanade parameters
    max_corners: int = Form(100),
    # MOG2 parameters
    history: int = Form(500),
    var_threshold: float = Form(16.0),
):
    """
    Execute Motion Analysis, Optical Flow, or Background Subtraction across video frames.
    """
    t0 = time.time()
    contents = await file.read()
    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    sample_frames_b64: List[str] = []
    motion_metrics: Dict[str, Any] = {}
    method = method.lower()

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot decode video file stream.")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_step = max(1, frame_step)
        sample_interval = max(1, total_frames // (frame_step * max(max_output_frames, 1)))

        if method == "farneback":
            # 8-bin directional motion accumulator (N, NE, E, SE, S, SW, W, NW)
            direction_bins = [0.0] * 8
            prev_gray = None
            frame_idx = 0
            output_count = 0

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if frame_idx % frame_step == 0:
                    small = resize_for_display(frame, max_dim=480)
                    gray = to_grayscale(small)

                    if prev_gray is not None:
                        flow = cv2.calcOpticalFlowFarneback(
                            prev_gray,
                            gray,
                            None,
                            pyr_scale=0.5,
                            levels=3,
                            winsize=15,
                            iterations=3,
                            poly_n=5,
                            poly_sigma=1.2,
                            flags=0,
                        )

                        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                        # Accumulate directional motion energy
                        # ang in radians [0, 2*pi]
                        bin_idx = np.floor((ang / (2 * np.pi)) * 8).astype(int) % 8
                        valid_mask = mag > 1.0  # filter out noise
                        for b in range(8):
                            b_mask = valid_mask & (bin_idx == b)
                            direction_bins[b] += float(np.sum(mag[b_mask]))

                        # Create HSV color representation for output sample
                        if (
                            output_count < max_output_frames
                            and (frame_idx // frame_step) % sample_interval == 0
                        ):
                            hsv = np.zeros_like(small)
                            hsv[..., 1] = 255
                            hsv[..., 0] = ang * 180 / np.pi / 2
                            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                            flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                            # Blend flow with original frame for visual aesthetics
                            vis = cv2.addWeighted(small, 0.45, flow_bgr, 0.55, 0)
                            cv2.putText(
                                vis,
                                f"Farneback Flow (Frame {frame_idx})",
                                (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA,
                            )
                            sample_frames_b64.append(
                                encode_image_to_base64(vis, quality=85)
                            )
                            output_count += 1

                    prev_gray = gray

                frame_idx += 1

            total_mag = sum(direction_bins) or 1.0
            polar_distribution = [
                round((val / total_mag) * 100, 2) for val in direction_bins
            ]
            motion_metrics = {
                "polar_directions": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                "polar_distribution_pct": polar_distribution,
                "total_motion_energy": round(total_mag, 2),
            }

        elif method == "lucas_kanade":
            # Feature parameters
            feature_params = dict(
                maxCorners=max_corners,
                qualityLevel=0.3,
                minDistance=7,
                blockSize=7,
            )
            # LK optical flow parameters
            lk_params = dict(
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
            )

            # Random distinct colors for point trajectory tracks
            color = np.random.randint(0, 255, (max_corners + 50, 3))

            ret, first_frame = cap.read()
            if ret and first_frame is not None:
                small = resize_for_display(first_frame, max_dim=480)
                old_gray = to_grayscale(small)
                p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
                mask_canvas = np.zeros_like(small)

                frame_idx = 1
                output_count = 0

                while True:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break

                    if frame_idx % frame_step == 0:
                        small = resize_for_display(frame, max_dim=480)
                        frame_gray = to_grayscale(small)

                        if p0 is not None and len(p0) > 0:
                            p1, st, err = cv2.calcOpticalFlowPyrLK(
                                old_gray, frame_gray, p0, None, **lk_params
                            )

                            if p1 is not None and st is not None:
                                good_new = p1[st == 1]
                                good_old = p0[st == 1]

                                for i, (new, old) in enumerate(zip(good_new, good_old)):
                                    a, b = new.ravel()
                                    c, d = old.ravel()
                                    c_val = color[i % len(color)].tolist()
                                    mask_canvas = cv2.line(
                                        mask_canvas,
                                        (int(a), int(b)),
                                        (int(c), int(d)),
                                        c_val,
                                        2,
                                    )
                                    small = cv2.circle(
                                        small, (int(a), int(b)), 4, c_val, -1
                                    )

                                img_vis = cv2.add(small, mask_canvas)

                                if (
                                    output_count < max_output_frames
                                    and (frame_idx // frame_step) % sample_interval == 0
                                ):
                                    cv2.putText(
                                        img_vis,
                                        f"KLT Sparse Flow (Frame {frame_idx})",
                                        (10, 20),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        (6, 182, 212),
                                        1,
                                        cv2.LINE_AA,
                                    )
                                    sample_frames_b64.append(
                                        encode_image_to_base64(img_vis, quality=85)
                                    )
                                    output_count += 1

                                old_gray = frame_gray.copy()
                                p0 = good_new.reshape(-1, 1, 2)
                            else:
                                p0 = cv2.goodFeaturesToTrack(
                                    frame_gray, mask=None, **feature_params
                                )
                                old_gray = frame_gray.copy()
                        else:
                            p0 = cv2.goodFeaturesToTrack(
                                frame_gray, mask=None, **feature_params
                            )
                            old_gray = frame_gray.copy()

                    frame_idx += 1

                motion_metrics = {
                    "method": "Lucas-Kanade Pyramidal Flow (KLT)",
                    "tracked_features_initial": len(p0) if p0 is not None else 0,
                }

        elif method in ["mog2", "knn"]:
            if method == "mog2":
                subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=history, varThreshold=var_threshold, detectShadows=True
                )
            else:
                subtractor = cv2.createBackgroundSubtractorKNN(
                    history=history, dist2Threshold=400.0, detectShadows=True
                )

            temporal_motion_energy = []
            frame_idx = 0
            output_count = 0

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if frame_idx % frame_step == 0:
                    small = resize_for_display(frame, max_dim=480)
                    fg_mask = subtractor.apply(small)

                    # Compute foreground motion pixel count
                    fg_pixels = int(np.count_nonzero(fg_mask == 255))
                    temporal_motion_energy.append({
                        "frame": frame_idx,
                        "motion_pixels": fg_pixels,
                    })

                    if (
                        output_count < max_output_frames
                        and (frame_idx // frame_step) % sample_interval == 0
                    ):
                        # Highlight foreground motion on color image in neon cyan
                        vis = small.copy()
                        vis[fg_mask == 255] = [6, 182, 212]
                        # Shadows in amber
                        vis[fg_mask == 127] = [245, 158, 11]

                        cv2.putText(
                            vis,
                            f"{method.upper()} Foreground (Frame {frame_idx})",
                            (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        sample_frames_b64.append(
                            encode_image_to_base64(vis, quality=85)
                        )
                        output_count += 1

                frame_idx += 1

            motion_metrics = {
                "temporal_energy_curve": temporal_motion_energy[::max(1, len(temporal_motion_energy)//20)],
                "background_subtractor": method.upper(),
            }

        cap.release()

    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    execution_ms = round((time.time() - t0) * 1000, 2)

    return {
        "status": "success",
        "method": method,
        "execution_time_ms": execution_ms,
        "sample_frames": sample_frames_b64,
        "metrics": motion_metrics,
    }
