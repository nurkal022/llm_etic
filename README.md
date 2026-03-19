# LLM Ethics — Fine-tuning Qwen3.5-4B for Content Safety

Fine-tuning **Qwen3.5-4B** with LoRA on [NVIDIA Aegis AI Content Safety Dataset 2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) for classifying text as **safe/unsafe** and identifying violated safety categories.

## Quick Start

```bash
# Clone
git clone <repo-url>
cd llm_etic

# Install
pip install -r requirements.txt

# Run training
bash fine_tune/run.sh
```

## Training

```bash
# Full training (1 epoch, ~500 eval samples)
python fine_tune/train.py --epochs 1 --eval_samples 500

# Quick test (30 steps)
python fine_tune/train.py --max_steps 30 --eval_samples 50

# Low VRAM (4-bit quantization)
python fine_tune/train.py --epochs 1 --load_in_4bit --batch_size 4
```

### Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--epochs` | 1 | Training epochs |
| `--max_steps` | None | Override epochs with fixed step count |
| `--batch_size` | 2 | Per-device batch size |
| `--grad_accum` | 4 | Gradient accumulation steps |
| `--lr` | 2e-4 | Learning rate |
| `--lora_r` | 16 | LoRA rank |
| `--max_length` | 2048 | Max sequence length |
| `--eval_samples` | 200 | Test samples for evaluation |
| `--save_steps` | 100 | Checkpoint interval |
| `--load_in_4bit` | off | 4-bit quantization |
| `--output_dir` | outputs | Output directory |

## Outputs

```
outputs/
├── checkpoints/              — model checkpoints
├── logs/
│   ├── dataset_stats.json    — dataset statistics
│   └── training_log.json     — full loss history, GPU, memory
├── plots/
│   ├── 01_training_loss.png
│   ├── 02_lr_schedule.png
│   ├── 03_confusion_matrices.png
│   ├── 04_precision_recall_f1.png
│   ├── 05_category_distribution.png
│   ├── 06_label_distribution.png
│   ├── 07_category_f1_heatmap.png
│   ├── 08_radar_chart.png
│   ├── 09_error_analysis.png
│   └── 10_dashboard.png
├── metrics/
│   ├── eval_metrics.json     — accuracy, F1, classification report
│   └── predictions.json      — raw predictions
├── qwen_aegis_safety_lora/   — final LoRA adapters
├── training_args.json
└── summary.json
```

## Requirements

- Python 3.10+
- CUDA GPU (16GB+ VRAM recommended, 8GB+ with `--load_in_4bit`)
- ~30GB disk for model + dataset cache

## Dataset

**NVIDIA Aegis AI Content Safety Dataset 2.0** — 33,416 annotated human-LLM interactions covering 12 safety categories: Hate/Identity Hate, Sexual, Suicide and Self Harm, Violence, Guns/Illegal Weapons, Threat, PII/Privacy, Sexual (minor), Criminal Planning/Confessions, Harassment, Controlled/Regulated Substances, Profanity.

## Project Structure

```
├── fine_tune/
│   ├── train.py    — main training script
│   └── run.sh      — launch script
├── requirements.txt
└── README.md
```
