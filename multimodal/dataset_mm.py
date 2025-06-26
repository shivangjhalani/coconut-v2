import json
import itertools
import random
from dataclasses import dataclass
from typing import Optional
import os

import torch
import torch.distributed as dist
from datasets import Dataset, load_from_disk
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from PIL import ImageFile

from multimodal.transforms import load_image

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_dataset_mm(
    path,
    tokenizer,
    img_root: str,
    max_size: int = 1000000000,
    image_size: int = 448,
    num_proc: int = 2,
 ):
    """Return a multimodal HF Dataset, building it if needed.

    Workflow:
    1. If ``path`` points to a directory containing a previously saved dataset
       (created via ``datasets.save_to_disk``), we load it directly → instant load.
    2. Else we assume ``path`` is a JSON file created by *scienceqa_to_coconut.py*
       and run the expensive image-tiling + tokenisation pipeline, then return the
       in-memory dataset (caller can later save it).
    """

    # Fast path: cached dataset dir
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "dataset_info.json")):
        return load_from_disk(path)

    def tokenize_sample(sample):
        """Tokenize text and load image. If image fails to decode, mark skip."""

        try:
            question_tokenized = tokenizer.encode(sample["question"] + "\n", add_special_tokens=True)
            steps_tokenized = [
                tokenizer.encode(s + "\n", add_special_tokens=False) for s in sample["steps"]
            ]
            answer_tokenized = tokenizer.encode("### " + sample["answer"], add_special_tokens=False) + [
                tokenizer.eos_token_id
            ]

            # Vision processing
            img_path = f"{img_root}/{sample['image']}"
            pixel_values, n_tiles = load_image(img_path, input_size=image_size)

            return {
                "question_tokenized": question_tokenized,
                "steps_tokenized": steps_tokenized,
                "answer_tokenized": answer_tokenized,
                "pixel_values": pixel_values,
                "num_patches": n_tiles,
                "idx": sample["idx"],
                "_skip": False,
            }

        except Exception:
            # Corrupted or missing image → drop later
            return {"_skip": True}

    data = json.load(open(path))[:max_size]
    data = [{**d, "idx": idx} for idx, d in enumerate(data)]

    keys = data[0].keys()
    dataset = Dataset.from_dict({k: [d[k] for d in data] for k in keys})

    # Map with multiprocessing; any record with _skip==True will be removed later
    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            processed_dataset = [
                dataset.map(tokenize_sample, remove_columns=list(dataset.features), num_proc=num_proc)
            ]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]
    else:
        dataset = dataset.map(tokenize_sample, remove_columns=list(dataset.features), num_proc=num_proc)

    # drop bad records
    dataset = dataset.filter(lambda example: not example["_skip"], num_proc=num_proc)

    return dataset


# ---------- Latent-aware helpers (adapted from original dataset.py) ----------

def _convert_instance_to_features(sample, start_id, latent_id, end_id, configs, tokenizer, no_special_marker=False):
    """Utility shared by question/cot dataset builders."""
    # Build input sequence same as original + vision token placeholders already in question text
    input_ids = sample["question_tokenized"]
    if not no_special_marker:
        input_ids += [start_id] + [latent_id] + [end_id]
    input_ids += sample["answer_tokenized"]

    labels = input_ids.copy()

    attention_mask = [1] * len(input_ids)
    position_ids = list(range(len(input_ids)))

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "pixel_values": sample["pixel_values"],
        "num_patches": sample["num_patches"],
        "idx": sample["idx"],
    }


def get_question_latent_dataset_mm(
    scheduled_stage,
    base_dataset_valid,
    configs,
    start_id,
    latent_id,
    end_id,
    *,
    no_special_marker=False,
):
    # identical to original but uses helper
    tokenizer = configs.tokenizer  # injected later

    def process(sample):
        return _convert_instance_to_features(sample, start_id, latent_id, end_id, configs, tokenizer, no_special_marker)

    return base_dataset_valid.map(process, remove_columns=list(base_dataset_valid.features), num_proc=2)


def get_cot_latent_dataset_mm(
    scheduled_stage,
    base_dataset,
    configs,
    start_id,
    latent_id,
    end_id,
    *,
    no_special_marker=False,
    shuffle=False,
):
    tokenizer = configs.tokenizer

    def process(sample):
        return _convert_instance_to_features(sample, start_id, latent_id, end_id, configs, tokenizer, no_special_marker)

    processed = base_dataset.map(process, remove_columns=list(base_dataset.features), num_proc=2)
    if shuffle:
        processed = processed.shuffle(seed=configs.seed)
    return processed 