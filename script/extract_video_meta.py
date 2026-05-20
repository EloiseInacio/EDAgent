import cv2
import numpy as np
import json
from pathlib import Path
import pandas as pd

def snr_across_frames(
    video_path: str,
    roi_size: int = 64,
    corner: str = "top_right",
    max_frames: int | None = None,
) -> float:
    """
    Compute temporal SNR across frames using mean/std in a small ROI in a corner.

    Method:
      1) Convert frames to luma (Y) from RGB (BT.601-ish approximation).
      2) For each frame, compute ROI mean -> m_t
      3) SNR_amp = mean(m_t) / std(m_t), SNR_dB = 20*log10(SNR_amp)

    Returns:
      snr_amp: float
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    # Read first frame to define ROI
    ok, frame_bgr = cap.read()
    if not ok:
        raise ValueError("Could not read first frame.")

    h, w = frame_bgr.shape[:2]
    s = int(np.sqrt(roi_size))
    if s <= 0 or s > min(h, w):
        raise ValueError(f"roi_size must be in [1, {min(h, w)}], got {roi_size}")

    corner = corner.lower()
    if corner == "top_left":
        x0, y0 = 0, 0
    elif corner == "top_right":
        x0, y0 = w - s, 0
    elif corner == "bottom_left":
        x0, y0 = 0, h - s
    elif corner == "bottom_right":
        x0, y0 = w - s, h - s
    else:
        raise ValueError("corner must be one of: top_left, top_right, bottom_left, bottom_right")

    # Helper: BGR -> luma Y (approx). Output float32.
    # Y ≈ 0.114 B + 0.587 G + 0.299 R
    def to_luma_y(bgr: np.ndarray) -> np.ndarray:
        b = bgr[..., 0].astype(np.float32)
        g = bgr[..., 1].astype(np.float32)
        r = bgr[..., 2].astype(np.float32)
        return 0.114 * b + 0.587 * g + 0.299 * r

    means = []

    # Process first frame + rest
    frame_count = 0
    while True:
        if frame_count == 0:
            frame = frame_bgr
        else:
            ok, frame = cap.read()
            if not ok:
                break

        y = to_luma_y(frame)
        patch = y[y0 : y0 + s, x0 : x0 + s]

        # Use the ROI MEAN per frame as the signal sample m_t
        means.append(float(patch.mean()))

        frame_count += 1
        if max_frames is not None and frame_count >= int(max_frames):
            break

    cap.release()

    m = np.array(means, dtype=np.float64)
    if m.size < 2:
        raise ValueError("Need at least 2 frames to compute std/SNR.")

    mu = float(m.mean())
    sigma = float(m.std(ddof=1))  # sample std; use ddof=0 for population std

    # Guard against divide-by-zero / near-zero noise
    eps = 1e-12
    snr_amp = mu / max(sigma, eps)

    return snr_amp



def compute_optical_flow_amplitude(video_path, sampling_rate=5, threshold_ratio=0.5):
    """
    Computes the average motion amplitude for a video using optical flow,
    with temporal downsampling, and estimates the action window
    (start_frame, end_frame) by thresholding the motion amplitude.

    Heuristic:
      - compute one amplitude value per sampled step (here: max flow magnitude)
      - threshold = threshold_ratio * mean(amplitude) (e.g., 0.5 = half the mean)
      - start_frame = first sampled frame where amplitude >= threshold
      - end_frame   = last sampled frame where amplitude >= threshold

    Args:
        video_path: Path to the video file
        sampling_rate: Temporal sampling rate (5 = analyze 1 frame out of 5)
        threshold_ratio: Ratio applied to the mean amplitude to define the threshold

    Returns:
        dict: Computation results (amplitude_mean, amplitude_max, start_frame, end_frame)
    """
    result = {
        "amplitude_mean": 0.0,
        "amplitude_max": 0.0,
        "start_frame": None,
        "end_frame": None,
    }

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            result["error"] = "Cannot open video"
            return result

        ret, prev_frame = cap.read()
        if not ret:
            result["error"] = "Cannot read first frame"
            cap.release()
            return result

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        frame_amplitudes = []
        sampled_frames = []

        frame_idx = 0  # prev_frame index
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            if frame_idx % sampling_rate != 0:
                continue

            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )

            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            amp = float(np.max(magnitude))

            frame_amplitudes.append(amp)
            sampled_frames.append(frame_idx)

            prev_gray = curr_gray

        cap.release()

        if not frame_amplitudes:
            result["error"] = "No frames analyzed"
            return result

        amps = np.asarray(frame_amplitudes, dtype=np.float32)
        mean_amp = float(np.mean(amps))
        result["amplitude_mean"] = mean_amp
        result["amplitude_max"] = float(np.max(amps))

        thr = float(threshold_ratio * mean_amp)
        above = amps >= thr

        if np.any(above):
            first_idx = int(np.argmax(above))  # first True
            last_idx = int(np.where(above)[0][-1])

            # Map back to ORIGINAL frame indices
            start_frame = sampled_frames[first_idx]
            end_frame = sampled_frames[last_idx]

            # Optional: small padding for nicer boundaries
            pad = sampling_rate // 2
            start_frame = max(0, start_frame - pad)
            end_frame = end_frame + pad

            result["start_frame"] = int(start_frame)
            result["end_frame"] = int(end_frame)

        return result

    except Exception as e:
        result["error"] = str(e)
        return result


def get_video_info(video_path: str):
    """
    Returns a dict with:
      - n_frames
      - duration_seconds
      - fps
      - width
      - height
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    if fps <= 0:
        raise ValueError(f"Invalid FPS ({fps}) for video: {video_path}")

    duration_s = n_frames / fps

    return {
        "n_frames": n_frames,
        "duration_seconds": duration_s,
        "fps": fps,
        "width": width,
        "height": height,
    }

def calculate_relative_duration(row):
        if pd.isna(row['start_frame']) or pd.isna(row['end_frame']) or row['n_frames'] == 0:
            return np.nan
        return (row['end_frame'] - row['start_frame']) / row['n_frames']

def compute_relative_activity_duration(df):
    """
    Given a DataFrame with 'start_frame', 'end_frame', and 'n_frames' columns,
    computes the relative activity duration for each video and adds it as a new column.

    Relative Activity Duration = (end_frame - start_frame) / n_frames

    Args:
        df (pd.DataFrame): DataFrame containing video metadata.
    Returns:
        pd.DataFrame: Updated DataFrame with 'relative_activity_duration' column.

    """

    df['relative_activity_duration'] = df.apply(calculate_relative_duration, axis=1)
    return df


def vipy_json_to_csv_category_blurred_filename(
    json_path,
    label,
    output_folder,
    filename_mode="basename",
):
    """
    Writes a CSV with columns ['category', 'blurred_face', 'filename'] from ViPy JSON.

    The output filename is formatted as:
        "{category}/{basename}"
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for item in data:
        if not isinstance(item, dict) or not item:
            continue

        scene = item.get("<class 'vipy.video.Scene'>") or next(iter(item.values()))
        if not isinstance(scene, dict):
            continue

        attrs = scene.get("attributes")
        attrs = attrs if isinstance(attrs, dict) else {}

        blurred_face = attrs.get("blurred_faces")

        filename = scene.get("_filename")
        if not filename:
            continue

        filename_str = str(filename)
        basename = Path(filename_str).name if filename_mode == "basename" else filename_str

        # prepend category with "/" separator
        filename_out = str(Path(str(label)) / basename)

        rows.append(
            {
                "category": label,
                "blurred_face": blurred_face,
                "filename": filename_out,
            }
        )

    df_out = pd.DataFrame(rows, columns=["category", "blurred_face", "filename"])
    output_csv_path = output_folder + f"_{label}.csv"
    return df_out
