# Data Setup

Training and evaluation read frames and annotations from a few dataset roots.
Set these roots in `configs/experiment/train.yaml` / `configs/experiment/eval.yaml`
(or override them on the command line). This document lists the frame layouts,
annotation files, and frame caches each dataset loader expects.

| Dataset | Config key | Used by |
| --- | --- | --- |
| CaptainCook4D | `data_paths.captaincook4d_root` | training (instructions + Ego-CoMist) |
| Ego4D | `data_paths.ego4d_root` | training |
| Ego-Exo4D | `data_paths.egoexo4d_root` | training (Ego-CoMist) |
| Ego-MC-Bench eval frames | `data_paths.ego_mc_bench_frames_root` | evaluation |
| Hugging Face cache | `hf_env.HF_HOME` | annotation downloads |

## Frame Extraction (common conventions)

- Frames are decoded at **2 fps** (`dataset_extracted_fps` / `model_input_fps`).
- Each video gets its own directory of JPEGs. Files sort numerically by the
  trailing digits of the filename (e.g. `frame_000000.jpg`, `frame_000001.jpg`).
- A loader discovers frames from the roots below and, on first run, builds a
  `video_to_frames.pkl` cache at the dataset root. Delete that pickle to force a
  rebuild after changing the frames on disk.
- A video directory may be named `<video_id>`, `<video_id>_360p`, or
  `<video_id>_2fps`; the `_360p` / `_2fps` suffixes are stripped to recover the
  video id.

## CaptainCook4D

Root: `$CAPTAINCOOK4D_PATH` (`data_paths.captaincook4d_root`).

### 1. Download the 360p GoPro videos

Use the official downloader (https://github.com/CaptainCook4D/downloader):

```bash
git clone https://github.com/CaptainCook4D/downloader.git
cd downloader

# Download only the GoPro 360p videos into $CAPTAINCOOK4D_PATH
python download_gopro_data.py --data2d --output_dir $CAPTAINCOOK4D_PATH
```

This writes the videos to:

```text
$CAPTAINCOOK4D_PATH/captain_cook_4d/gopro/resolution_360p/<video_id>_360p.mp4
```

### 2. Extract frames at 360p, 2 fps

Decode each video to JPEGs at 2 fps with ffmpeg, into the directory layout the
loader expects (`<video_id>_360p/` under `resolution_360p_2fps_frames`). The
GoPro videos are already 360p; `scale=-2:360` keeps that height while preserving
aspect ratio:

```bash
SRC=$CAPTAINCOOK4D_PATH/captain_cook_4d/gopro/resolution_360p
DST=$CAPTAINCOOK4D_PATH/captain_cook_4d/gopro/resolution_360p_2fps_frames

for video in "$SRC"/*_360p.mp4; do
    name=$(basename "$video" .mp4)        # e.g. <video_id>_360p
    mkdir -p "$DST/$name"
    ffmpeg -i "$video" -vf "scale=-2:360,fps=2" -q:v 2 "$DST/$name/frame_%06d.jpg"
done
```

Resulting frame layout (searched by the loader in this order):

```text
$CAPTAINCOOK4D_PATH/
  captain_cook_4d/gopro/resolution_360p_2fps_frames/<video_id>_360p/*.jpg
  # or
  frames/<video_id>_360p/*.jpg
  # or
  frames/<video_id>/*.jpg
  video_to_frames.pkl      # built automatically on first run
```

Annotations — the default config uses Hugging Face annotations
(`annotation_source: hf`, repo `qualcomm/qualcomm-interactive-cooking-dataset`,
set `main`), downloaded automatically to `hf_env.HF_HOME`. No local annotation
files are required in that mode.

To use local annotation files instead (`annotation_source: local`):

```text
$CAPTAINCOOK4D_PATH/
  main_train.json
  main_val.json
  main_test.json
```

## Ego4D

Root: `$EGO4D_PATH` (`data_paths.ego4d_root`).

### 1. Get access and download the videos

Ego4D requires accepting the license agreement. Follow the official
instructions at https://ego4d-data.org/ to request access and obtain your AWS
credentials, then install and use the Ego4D CLI:

```bash
pip install ego4d

# Download the GoalStep full-scale videos into $EGO4D_PATH/ego4d_data
python -m ego4d.cli.cli \
    --output_directory $EGO4D_PATH/ego4d_data \
    --datasets full_scale \
    --benchmarks goalstep \
    --version v2
```

This writes the videos to:

```text
$EGO4D_PATH/ego4d_data/v2/full_scale/<video_id>.mp4
```

### 2. Extract frames at 2 fps (short side 512)

Decode each video to JPEGs at 2 fps, scaled so the short side is 512px, into the
layout the loader expects (`frames_fps2_short512/<video_id>/`):

```bash
SRC=$EGO4D_PATH/ego4d_data/v2/full_scale
DST=$EGO4D_PATH/ego4d_data/v2/frames_fps2_short512

for video in "$SRC"/*.mp4; do
    name=$(basename "$video" .mp4)        # <video_id>
    mkdir -p "$DST/$name"
    ffmpeg -i "$video" -vf "scale='if(gt(iw,ih),-2,512)':'if(gt(iw,ih),512,-2)',fps=2" -q:v 2 "$DST/$name/frame_%06d.jpg"
done
```

Resulting frame layout (searched by the loader in this order):

```text
$EGO4D_PATH/
  ego4d_data/v2/frames_fps2_short512/<video_id>/*.jpg
  # or
  frames/<video_id>/*.jpg
```

### 3. Annotations and caches

```text
$EGO4D_PATH/
  video_to_actions_instruct_train.pkl   # per-video action instructions (required)
  videos_to_frames_train.pkl            # frame cache (data_paths.ego4d_videos_to_frames_cache)
```

- `video_to_actions_instruct_<split>.pkl` maps `video_id -> {action_id -> {instruction|narration, start_time, end_time}}`.
- `videos_to_frames_<split>.pkl` is the frame cache; built automatically from the
  frame directories above if it does not exist.

## Ego-Exo4D

Root: `$EGOEXO4D_PATH` (`data_paths.egoexo4d_root`).

### 1. Get access and download the takes

Ego-Exo4D requires accepting the license agreement. Follow the official
instructions at https://ego-exo4d-data.org/ to request access and obtain your
AWS credentials, then install and use the Ego-Exo4D CLI:

```bash
pip install ego4d

# Download the downscaled take videos and metadata into $EGOEXO4D_PATH
egoexo \
    --output_directory $EGOEXO4D_PATH \
    --parts downscaled_takes metadata
```

This writes the egocentric (Aria) take videos under:

```text
$EGOEXO4D_PATH/takes/<take_name>/frame_aligned_videos/downscaled/448/aria01_214-1.mp4
```

(the exact filename of the egocentric stream varies per take).

### 2. Extract frames at 360p, 2 fps

Decode each take's egocentric video to JPEGs at 2 fps, scaled to 360p, into the
layout the loader expects (`data/frames_360p_2fps/<take_name>/`):

```bash
DST=$EGOEXO4D_PATH/data/frames_360p_2fps

for take_dir in "$EGOEXO4D_PATH"/takes/*/; do
    take_name=$(basename "$take_dir")
    # Pick the egocentric (aria) downscaled stream for this take
    video=$(ls "$take_dir"frame_aligned_videos/downscaled/448/aria*.mp4 2>/dev/null | head -n 1)
    [ -z "$video" ] && continue
    mkdir -p "$DST/$take_name"
    ffmpeg -i "$video" -vf "scale=-2:360,fps=2" -q:v 2 "$DST/$take_name/frame_%06d.jpg"
done
```

Resulting frame layout (searched by the loader in this order):

```text
$EGOEXO4D_PATH/
  data/frames_360p_2fps/<take_name>/*.jpg
  # or
  frames/<take_name>/*.jpg
  video_to_frames.pkl      # built automatically on first run
```

### 3. Annotations

```text
$EGOEXO4D_PATH/
  cooking_takes_keysteps_with_actions.json
```

- Keyed by `take_uid`; each take has `take_name`, `split`, and
  `keysteps_with_actions` (each with `interval.start_sec`, `interval.end_sec`,
  and `step_description`). The `take_name` must match the frame directory name.

## Ego-CoMist (counterfactual training data)

`EgoCoMist` provides the counterfactual mistake/intervention annotations, but it
does **not** ship any frames. It reuses the frames you set up above for
**CaptainCook4D**, **Ego4D**, and **Ego-Exo4D** — so complete those three
sections first.

### How the source setups connect

The annotations come from the Hugging Face repo
[`qualcomm/qualcomm-interactive-cooking-dataset-counterfactual-mistakes`](https://huggingface.co/datasets/qualcomm/qualcomm-interactive-cooking-dataset-counterfactual-mistakes)
and are downloaded automatically to `hf_env.HF_HOME` (25,087 rows across 1,110
unique videos). Each row carries a `dataset` field and a `video_id`:

```json
{
  "dataset": "Ego4D",
  "video_id": "39d087b0-afc2-47d8-ba91-b70dd8fab90e",
  "start_time": 1191.26881,
  "end_time": 1204.66377,
  "mistake_type": "preparation_error",
  "texts": ["Microwave the water until boiling.", "For this step, please boil the water in the microwave instead of on the stovetop."],
  "text_timestamps": [1191.26881, 1196.26881],
  "text_types": ["Instruction", "Feedback"]
}
```

The loader resolves frames by mapping `dataset` to the matching root and looking
up the `video_id` in that root's frames:

| `dataset` value | Resolved from | Config key | Rows |
| --- | --- | --- | --- |
| `CaptainCook4D` | CaptainCook4D frames (`<video_id>_360p/`) | `data_paths.captaincook4d_root` | 4,969 |
| `Ego4D` | Ego4D frames (`<video_id>/`) | `data_paths.ego4d_root` | 13,847 |
| `Ego-Exo4D` | Ego-Exo4D frames (`<take_name>/`) | `data_paths.egoexo4d_root` | 6,271 |

A row is skipped if its frames are missing, so make sure all three roots are set
and their frames extracted (per the sections above) before training with
Ego-CoMist. No additional download or extraction step is needed beyond the three
source datasets plus the automatic annotation download.

## Ego-MC-Bench (evaluation)

The evaluation benchmark is the Hugging Face dataset
[`qualcomm/qualcomm-interactive-cooking-dataset-ego-mistake-corrections`](https://huggingface.co/datasets/qualcomm/qualcomm-interactive-cooking-dataset-ego-mistake-corrections)
(40 test videos, 1,597 annotations). Each row has a `video_id` plus aligned
`texts`, `text_types`, and `text_timestamps`:

```json
{
  "video_id": "GX010021",
  "texts": ["Let's first preheat the oven to 350 degrees Fahrenheit.", "Great Job!"],
  "text_types": ["Instruction", "Success"],
  "text_timestamps": [4.418, 15.982]
}
```

The dataset ships annotations at `annotations/annotations.json` and videos at
`videos/<video_id>.MP4`. There are two ways to set it up.

### Option A: Let the loader download and extract (default)

With the default eval config, annotations and videos are downloaded
automatically from Hugging Face to `hf_env.HF_HOME`. If `frames_root` is not
set, the loader extracts frames from the MP4s at 2 fps into
`outputs/cache/qualcomm_interactive_cooking_frames` (requires `opencv-python`).
No manual steps are needed; just run `python scripts/run.py experiment=eval`.

### Option B: Pre-extract frames yourself

Stage the dataset locally and extract frames at 2 fps so evaluation reads them
from `$EVAL_FRAMES_PATH` (`data_paths.ego_mc_bench_frames_root`).

1. Download the dataset (requires accepting the license on the dataset page):

```bash
huggingface-cli download \
    qualcomm/qualcomm-interactive-cooking-dataset-ego-mistake-corrections \
    --repo-type dataset \
    --local-dir $EVAL_DATASET_PATH
```

This gives:

```text
$EVAL_DATASET_PATH/
  annotations/annotations.json
  videos/<video_id>.MP4
```

2. Extract frames at 2 fps into the layout the loader expects
   (`<video_id>/` directories under `$EVAL_FRAMES_PATH`):

```bash
SRC=$EVAL_DATASET_PATH/videos
DST=$EVAL_FRAMES_PATH

for video in "$SRC"/*.MP4; do
    name=$(basename "$video" .MP4)        # <video_id>
    mkdir -p "$DST/$name"
    ffmpeg -i "$video" -vf "fps=2" -q:v 2 "$DST/$name/frame_%06d.jpg"
done
```

Resulting frame layout:

```text
$EVAL_FRAMES_PATH/
  <video_id>/*.jpg
```

To use the locally downloaded annotations/videos instead of the automatic
download, also pass `data.dataset.local_dir=$EVAL_DATASET_PATH` to `python scripts/run.py experiment=eval`.
