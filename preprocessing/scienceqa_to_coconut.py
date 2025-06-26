import argparse
import base64
import json
import os
import re
from pathlib import Path
from typing import List

from PIL import Image  # type: ignore
import io


def sentence_split(text: str) -> List[str]:
    """Simple sentence splitter based on punctuation."""
    # Remove line breaks for cleaner splitting
    text = text.replace("\n", " ")
    # Split on . ? ! followed by space or end of string
    pieces = re.split(r"(?<=[\.!?])\s+", text)
    return [p.strip() for p in pieces if p.strip()]


def convert_scienceqa_jsonl(
    jsonl_path: str,
    output_json: str,
    image_out_dir: str,
):
    Path(image_out_dir).mkdir(parents=True, exist_ok=True)

    converted: List[dict] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip broken line

            # Basic validity checks
            if not record.get("image") or not record["image"].get("bytes"):
                continue
            if not record.get("choices"):
                continue
            if not record.get("solution"):
                continue

            # Prepare image
            img_b64 = record["image"]["bytes"]
            try:
                img_bytes = base64.b64decode(img_b64)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                continue  # skip if image corrupted

            img_filename = f"scienceqa_{idx:07d}.png"
            img_path = os.path.join(image_out_dir, img_filename)
            try:
                img.save(img_path)
            except Exception:
                continue

            # Merge question + choices
            question_text = record["question"].strip()
            choices = record["choices"]
            choices_str = ", ".join(f"({i}) {c}" for i, c in enumerate(choices))
            merged_question = f"{question_text} The choices are: {choices_str}."

            # Steps from solution sentences
            steps = sentence_split(record["solution"])
            if not steps:
                continue

            answer_idx = record.get("answer")
            if answer_idx is None or not (0 <= answer_idx < len(choices)):
                continue
            answer = str(answer_idx)  # model will output index

            converted.append(
                {
                    "image": os.path.relpath(img_path, start=Path(output_json).parent),
                    "question": merged_question,
                    "steps": steps,
                    "answer": answer,
                }
            )

    # Dump to json
    with open(output_json, "w", encoding="utf-8") as f_out:
        json.dump(converted, f_out, ensure_ascii=False, indent=2)

    print(f"[scienceqa_to_coconut] Converted {len(converted)} examples → {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ScienceQA jsonl (with base64 images) into COCONUT-compatible multimodal JSON."
    )
    parser.add_argument("jsonl", type=str, help="Path to the ScienceQA jsonl file produced earlier")
    parser.add_argument("--output", "-o", type=str, default="data/scienceqa_coconut.json", help="Output JSON file path")
    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/images/scienceqa",
        help="Directory to dump decoded images (will be created if missing)",
    )
    args = parser.parse_args()

    convert_scienceqa_jsonl(args.jsonl, args.output, args.image_dir) 