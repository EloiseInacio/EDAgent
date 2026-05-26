"""
EDAgent configuration file.

All user-tunable constants are collected here so that you can adapt the
analysis to your dataset and preferences without touching the script internals.
Each constant includes a brief description of what it controls.
"""

# ===========================================================================
# General analysis settings
# ===========================================================================

# Maximum number of rows sampled for expensive per-asset metadata extraction
# (video, audio, skeleton). Raise for richer coverage; lower if extraction is slow.
MAX_ASSET_ROWS: int = 50

# Maximum number of items to show in ranked outputs produced by run_core_eda
# (e.g. top label classes, top co-missing pairs, top outlier rows).
EDA_TOP_N: int = 20

# Same output-size limit, applied to leakage report ranked lists
# (e.g. top suspicious token frequencies, top proxy-column scores).
LEAKAGE_TOP_N: int = 25

# ===========================================================================
# Label and class analysis
# ===========================================================================

# A label class whose relative frequency is below this proportion is flagged
# as rare. Increase if you want to catch more imbalanced classes.
RARE_CLASS_PROPORTION_THRESHOLD: float = 0.01

# A label class with fewer than this many samples is flagged as rare,
# regardless of proportion. Useful for very large datasets where even 1% can
# still be thousands of examples.
RARE_CLASS_COUNT_THRESHOLD: int = 5

# Minimum class size required to run within-class outlier detection (label
# noise heuristic). Classes smaller than this are skipped to avoid false positives.
LABEL_NOISE_MIN_CLASS_SIZE: int = 10

# ===========================================================================
# Numeric feature analysis
# ===========================================================================

# IQR / total-range ratio below which a numeric column is considered
# near-constant and flagged. Raise to catch columns with very small variance.
NEAR_CONSTANT_IQR_RATIO: float = 1e-6

# Span of log10(range) values across all numeric columns above which a
# mixed-scale warning is raised (i.e. columns differ by more than this many
# orders of magnitude). Lower this to flag moderate scale differences.
SCALE_SPREAD_ORDERS_THRESHOLD: float = 2.0

# |skewness| above this triggers a high-skew flag on a numeric column.
SKEWNESS_HIGH: float = 2.0

# Right-skew threshold for log-transform recommendation: a positive-only
# column with skewness above this value gets a "consider log(x)" hint.
SKEWNESS_LOG_CANDIDATE: float = 1.5

# A column whose value range exceeds median_range * this factor is flagged as
# dominating the feature scale, which can bias distance-based models.
DOMINANT_SCALE_RATIO: float = 10.0

# ===========================================================================
# Co-missingness analysis
# ===========================================================================

# Minimum Jaccard similarity between two columns' missing-value masks to
# report the pair as co-missing. 0 = report all pairs; 1 = only identical masks.
CO_MISSING_MIN_JACCARD: float = 0.5

# ===========================================================================
# Leakage detection
# ===========================================================================

# Minimum frequency for a path token to be considered when checking
# label-token correlations. Raise to ignore rare tokens.
LEAKAGE_MIN_TOKEN_COUNT: int = 3

# Proxy-feature purity above this rate is reported as high-risk leakage
# (e.g. a path token that appears almost exclusively in one label class).
LEAKAGE_HIGH_RISK_MATCH_RATE: float = 0.8

# Severity score thresholds for the three-level risk classification.
# Scores are in [0, 1]; anything at or above HIGH is "high risk",
# anything at or above MEDIUM (but below HIGH) is "medium risk".
LEAKAGE_RISK_HIGH_THRESHOLD: float = 0.85
LEAKAGE_RISK_MEDIUM_THRESHOLD: float = 0.55

# ===========================================================================
# Data profiling
# ===========================================================================

# Minimum fraction of rows that must share a common parent stem for a
# hierarchical filename structure to be reported as intentional nesting
# rather than noise. Lower this if your dataset has loosely grouped files.
NESTING_PURITY_THRESHOLD: float = 0.95

# ===========================================================================
# File type detection
# ===========================================================================

# Extensions recognised as text assets when scanning path columns.
# Add entries (e.g. ".xml", ".html") to extend text-asset detection.
TEXT_FILE_SUFFIXES: set = {".txt", ".md", ".log", ".json", ".jsonl", ".csv", ".tsv"}

# Extensions recognised as audio assets when scanning path columns.
AUDIO_FILE_SUFFIXES: set = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

# Column name substrings that hint at a free-text content column (case-insensitive
# after normalisation). Add domain-specific names to improve auto-detection.
TEXT_COLUMN_HINTS: set = {
    "text",
    "transcript",
    "caption",
    "sentence",
    "document",
    "content",
    "utterance",
    "review",
    "comment",
    "description",
    "body",
    "prompt",
    "response",
}

# ===========================================================================
# Audio metadata extraction
# ===========================================================================

# Target sample rate (Hz) for decoding audio files. Librosa resamples on load.
# 22050 Hz is standard for most speech/music feature extraction.
AUDIO_SR_TARGET: int = 22050

# dB threshold below which a signal region is classified as silence
# (passed to librosa.effects.split as top_db). Higher → stricter silence detection.
AUDIO_SILENCE_TOP_DB: float = 60.0

# Amplitude fraction at or above which a sample is considered clipped.
# The default 0.999 treats near-full-scale samples as clipped. Lower to be stricter.
AUDIO_CLIPPING_THRESHOLD: float = 0.999

# Maximum duration (seconds) loaded per audio file. Longer files are truncated
# at this point. Raise for datasets with long recordings; lower to save memory.
AUDIO_MAX_DURATION_S: float = 120.0

# ===========================================================================
# Video metadata extraction
# ===========================================================================

# Side length (pixels) of the square corner region-of-interest used to
# estimate temporal SNR. Must be ≤ min(video height, video width).
VIDEO_ROI_SIZE: int = 64

# Temporal subsampling factor for optical flow: 1 frame is analysed per this
# many frames. 1 = every frame; 5 = every 5th frame (faster, less precise).
VIDEO_FLOW_SAMPLING_RATE: int = 5

# Fraction of the mean optical flow amplitude used as the activity threshold
# when estimating action-window start/end frames. 0.5 = half the mean amplitude.
VIDEO_FLOW_THRESHOLD_RATIO: float = 0.5

# Farneback optical flow parameters (passed directly to cv2.calcOpticalFlowFarneback).
VIDEO_FLOW_PYR_SCALE: float = 0.5   # image pyramid scale factor (<1)
VIDEO_FLOW_LEVELS: int = 3          # number of pyramid levels
VIDEO_FLOW_WINSIZE: int = 15        # averaging window size (pixels)
VIDEO_FLOW_ITERATIONS: int = 3      # iterations per pyramid level
VIDEO_FLOW_POLY_N: int = 5          # pixel neighbourhood size for polynomial expansion
VIDEO_FLOW_POLY_SIGMA: float = 1.2  # Gaussian std for pre-smoothing before polynomial fit

# ===========================================================================
# Skeleton / pose metadata extraction
# ===========================================================================

# Minimum keypoint confidence score to count a joint as visible.
# Applied to COCO 17-joint format. Lower to include uncertain detections.
SKELETON_CONF_THRESHOLD: float = 0.3

# Minimum number of joints that must be visible in a frame for it to be
# counted as a valid (non-occluded) pose frame.
SKELETON_MIN_JOINTS: int = 10

# ===========================================================================
# Text metadata extraction
# ===========================================================================

# Inner boundaries for document length (word count) buckets.
# Four bands are produced: very_short / short / medium / long.
# Adjust these to match the vocabulary size range of your corpus.
#   - very_short : word_count <  TEXT_LENGTH_BUCKET_BINS[0]
#   - short      : word_count <  TEXT_LENGTH_BUCKET_BINS[1]
#   - medium     : word_count <  TEXT_LENGTH_BUCKET_BINS[2]
#   - long       : word_count >= TEXT_LENGTH_BUCKET_BINS[2]
TEXT_LENGTH_BUCKET_BINS: list = [50, 200, 1000]
