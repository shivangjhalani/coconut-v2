import argparse
import os
import sys

# Add repo root to sys.path so that `multimodal` and other project modules resolve
sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))

from datasets import disable_caching
from transformers import AutoTokenizer

from multimodal.dataset_mm import get_dataset_mm


def build_cache(json_path: str, img_root: str, model_id: str, out_dir: str, image_size: int = 448, num_proc: int = 8):
    """Build and save a cached HuggingFace dataset so later training runs skip image tiling."""

    disable_caching()  # avoid writing temp files elsewhere
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    dataset = get_dataset_mm(json_path, tokenizer, img_root, image_size=image_size, num_proc=num_proc)
    os.makedirs(out_dir, exist_ok=True)
    dataset.save_to_disk(out_dir)
    print(f"Saved pre-processed dataset with {len(dataset)} items to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache multimodal dataset for faster later training")
    parser.add_argument("json", help="Path to scienceqa_train.json or similar")
    parser.add_argument("img_root", help="Root directory containing decoded images")
    parser.add_argument("out_dir", help="Target directory to save arrow dataset")
    parser.add_argument("--model", default="OpenGVLab/InternVL3-1B-Pretrained", help="HF model id to obtain tokenizer")
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--num-proc", type=int, default=8, help="Number of CPU workers for preprocessing")
    args = parser.parse_args()

    build_cache(args.json, args.img_root, args.model, args.out_dir, args.image_size, args.num_proc) 