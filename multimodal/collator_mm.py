from typing import List, Optional
import os

import torch
from transformers import PreTrainedTokenizerBase

from dataset import MyCollator  # reuse padding logic for text

# Local image loader
from multimodal.transforms import load_image


class MyCollatorMM(MyCollator):
    """Data collator that merges text fields (via MyCollator) and dynamically loads
    image tiles if they haven't been pre-computed and attached to the sample."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        image_root: str,
        image_size: int = 448,
        include_num_patches: bool = False,
        latent_id: Optional[int] = None,
        label_pad_token_id: Optional[int] = -100,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            latent_id=latent_id,
            label_pad_token_id=label_pad_token_id,
        )

        self.image_root = image_root
        self.image_size = image_size
        self.include_num_patches = include_num_patches

    def _prepare_vision(self, feature):
        """Return (pixel_values, n_tiles) for a single feature, computing them if necessary."""

        if "pixel_values" in feature and "num_patches" in feature:
            # Already present → just pop & return
            pv = feature.pop("pixel_values")
            npatches = feature.pop("num_patches")
            return pv, npatches

        # Otherwise compute from the stored image filename
        img_name = feature.pop("image", None)
        if img_name is None:
            raise ValueError("Feature is missing both pixel tensors and image path.")

        img_path = img_name if os.path.isabs(img_name) else os.path.join(self.image_root, img_name)
        pixel_values, n_tiles = load_image(img_path, input_size=self.image_size)
        return pixel_values, n_tiles

    def __call__(self, features, return_tensors=None):
        # Prepare vision tensors / num_patches for every feature
        pixel_stacks: List[torch.Tensor] = []
        num_patches: List[int] = []

        for feat in features:
            pv, npatches = self._prepare_vision(feat)
            pixel_stacks.append(pv)
            num_patches.append(npatches)

        # Text padding via parent collator (this will also remove vision keys already popped)
        batch = super().__call__(features, return_tensors=return_tensors)

        # Concatenate and cast vision tensors to bfloat16 (InternVL expects bf16/fp16)
        batch["pixel_values"] = torch.cat(pixel_stacks, dim=0).to(torch.bfloat16)
        # image_flags: 1 per sample – InternVLChat expects this field (B,1)
        batch["image_flags"] = torch.ones((len(num_patches), 1), dtype=torch.int32)
        if self.include_num_patches:
            batch["num_patches_list"] = torch.tensor(num_patches, dtype=torch.int32)
        return batch 