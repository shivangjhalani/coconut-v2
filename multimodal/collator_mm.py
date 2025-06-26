from typing import Optional, List
import torch
from transformers import PreTrainedTokenizerBase

from dataset import MyCollator  # reuse padding logic for text


class MyCollatorMM(MyCollator):
    """Data collator that additionally merges pixel_values and num_patches_list."""

    def __call__(self, features, return_tensors=None):
        # Separate vision fields
        pixel_stacks: List[torch.Tensor] = [f.pop("pixel_values") for f in features]
        num_patches: List[int] = [f.pop("num_patches") for f in features]

        # Re-use the original text padding logic
        batch = super().__call__(features, return_tensors=return_tensors)

        # Concatenate vision tensors (N_total, 3, H, W)
        pixel_values = torch.cat(pixel_stacks, dim=0)
        batch["pixel_values"] = pixel_values
        batch["num_patches_list"] = torch.tensor(num_patches, dtype=torch.int32)
        return batch 