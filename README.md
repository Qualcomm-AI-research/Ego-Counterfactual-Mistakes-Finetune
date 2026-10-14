# Streaming Interventions: Can Video Large Language Models Correct Mistakes as They Occur?

Here we provide the code to train and evaluate streaming video large language models for reactive, step-by-step task guidance, introduced in the NeurIPS 2026 (E&D) paper:

[**Streaming Interventions: Can Video Large Language Models Correct Mistakes as They Occur?**](https://arxiv.org/abs/2606.09547)

## Abstract

Learning everyday skills, like cooking a dish, relies increasingly on instructional media such as online videos. This opens the door to the use of video (and multimodal) large language models (LLMs) as task guidance assistants. A crucial capability for the real-world success of a prospective task guidance assistant is its ability to intervene proactively as soon as a mistake is apparent in order to guide the user. To evaluate this crucial capability, we introduce Ego-MC-Bench (Mistake Corrections), a benchmark for evaluating reactive, step-by-step task guidance in realistic cooking scenarios. Extensive experiments show that Ego-MC-Bench is highly challenging for state-of-the-art video LLMs. We argue that a key reason is the limited availability of training data for fine-tuning models on this task. Although there exists a wide range of cooking video datasets, existing datasets lack examples of mistakes along with appropriately timed interventions. To help address this data limitation, we also introduce Ego-CoMist, a counterfactual synthetic dataset created by transforming non-interactive cooking videos into supervised training examples showing proactive interventions. We show that fine-tuning on Ego-CoMist yields performance gains especially for smaller and more efficient video LLMs that are well suited for delivering assistance on edge devices.

This repository fine-tunes `Qwen/Qwen3.5-2B` with LoRA on a weighted mixture of CaptainCook4D, Ego4D GoalStep, and the Ego-CoMist counterfactual dataset (`EgoCoMist`), and evaluates on the Ego-MC-Bench benchmark (`EgoMCBench`).

## Running the Code

### Getting Started

First, clone the repository using the following command:

```
git clone <repo-url> $REPO_PATH
cd $REPO_PATH
```

Here, *$REPO_PATH* is the desired location to download the repository.

Next, build the Docker image with the project requirements (requires Docker with
the NVIDIA container runtime). The image is based on a PyTorch CUDA image and
bundles the project dependencies and `ffmpeg` for frame extraction:

```
make build
```

Then start an interactive GPU container from the built image:

```
make boot-attached
```

All commands below are run inside this container.

The Hugging Face cache (`HF_HOME`) and the dataset roots are configured in `configs/experiment/train.yaml` and `configs/experiment/eval.yaml`. Set each path in those configs, or override it on the command line. The available variables are:

See [DATA.md](DATA.md) for the frame layouts, annotation files, and caches each dataset requires.

| Dataset | Config key |
| --- | --- |
| CaptainCook4D | `data_paths.captaincook4d_root` |
| Ego4D | `data_paths.ego4d_root` |
| Ego4D frame cache | `data_paths.ego4d_videos_to_frames_cache` |
| Ego-Exo4D | `data_paths.egoexo4d_root` |
| Ego-MC-Bench eval frames | `data_paths.ego_mc_bench_frames_root` |
| Hugging Face cache | `hf_env.HF_HOME` |

### Download Checkpoint

The released fine-tuned checkpoint is published as a GitHub release asset.
Download it into `./ckpt` (the default `ckpt_path` in
`configs/experiment/eval.yaml`):

```bash
wget --no-check-certificate -P ./ckpt \
    https://github.com/<org>/<repo>/releases/download/v1.0/pytorch_model.bin
```

This produces `./ckpt/pytorch_model.bin`. Evaluation predictions are written to
`./ckpt/results.json`.

### Training

Run the default cooking fine-tuning job:

```
python scripts/run.py experiment=train
```

Any config value can be overridden on the command line, for example:

```
# Point the datasets and HF cache elsewhere
python scripts/run.py experiment=train \
    hf_env.HF_HOME=$HF_CACHE_PATH \
    data_paths.captaincook4d_root=$CAPTAINCOOK4D_PATH \
    data_paths.ego4d_root=$EGO4D_PATH \
    data_paths.egoexo4d_root=$EGOEXO4D_PATH
```

Training outputs are written under Hydra output directories inside `outputs/logs/train`. Each run saves:

- `config.yaml` and `config_resolved.yaml` for reproducibility.
- `metrics.json` with train, validation, and test losses.
- `checkpoints/epoch-*` DeepSpeed checkpoints plus `checkpoints/latest`.
- `tokenizer/` with the tokenizer files used by the run.

Training requires 8 A100/H100 GPUs with 80GB VRAM.

### Exporting Checkpoints

DeepSpeed saves sharded ZeRO checkpoints under `checkpoints/`. Use
`scripts/export_zero_checkpoint.py` to consolidate a checkpoint into fp32
weights:

```
python scripts/export_zero_checkpoint.py $CHECKPOINT_DIR $FP32_EXPORT_DIR
```

- `$CHECKPOINT_DIR` is either the checkpoint root (e.g. `.../checkpoints`, which
  uses the `latest` tag) or a specific tag directory (e.g. `.../checkpoints/epoch-6`).
- `$FP32_EXPORT_DIR` is created if needed and receives the consolidated weights.
- Pass `--tag epoch-1` to select a tag explicitly when giving the checkpoint root.
- Pass `--safe-serialization` to write safetensors shards instead of a PyTorch
  state dict.

### Evaluation

Run evaluation on the Qualcomm Interactive Cooking benchmark:

```
python scripts/run.py experiment=eval ckpt_path=$CHECKPOINT_PATH
```

This uses the Hugging Face dataset `qualcomm/qualcomm-interactive-cooking-dataset-ego-mistake-corrections`; annotations are downloaded automatically through `huggingface_hub`, and video frames are read from `data_paths.ego_mc_bench_frames_root`. Override paths as needed:

```
python scripts/run.py experiment=eval \
    ckpt_path=$CHECKPOINT_PATH \
    data_paths.ego_mc_bench_frames_root=$EVAL_FRAMES_PATH
```

We also provide a pre-trained checkpoint in "Releases".

Note: This code downloads the [Qualcomm Interactive Cooking Dataset](https://huggingface.co/datasets/qualcomm/qualcomm-interactive-cooking-dataset) automatically.

## Repository Structure

The repository has the following structure,

```text
streaming_vlms
└── docker/ : Docker build files
    └── Dockerfile : PyTorch CUDA base image with project deps
    └── requirements-pip.txt : runtime Python dependencies
└── Makefile : `make build` and `make boot-attached` helpers
└── configs/ : Hydra configuration
    └── experiment/ : train.yaml and eval.yaml (entry configs, data + HF paths)
    └── data/ : default.yaml (the training data mixture)
    └── model/, trainer/, pipeline/, paths/, callbacks/
└── scripts/
    └── run.py : Hydra/DeepSpeed entrypoint
    └── export_zero_checkpoint.py : exports a ZeRO checkpoint to fp32 weights
└── src/
    └── pipeline.py : training and evaluation loop
    └── data.py : builds dataset mixtures from config
    └── collator.py, model.py, eval.py : collation, model setup, eval metrics
    └── datasets/ : dataset loaders
        └── ego4d/ : Goalstep instruction loaders
        └── qualcomm_interactive_cooking/ : EgoCoMist and EgoMCBench loaders
```

## License

The code in this repository is released under the BSD-3-Clause-Clear license. Please refer to LICENSE for details.

## Citation

```text
@inproceedings{streaminginterventions,
   title = {Streaming Interventions: Can Video Large Language Models Correct Mistakes as They Occur?},
   author = {Apratim Bhattacharyya and Shweta Mahajan and Sanjay Haresh and Rajeev Yasarla and Reza Pourreza and Litian Liu and Risheek Garrepalli and Roland Memisevic},
   booktitle = {NeurIPS (E&D)},
   year = {2026},
}
```
