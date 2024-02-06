"""
# Author: Yinghao Li
# Modified: September 30th, 2023
# ---------------------------------------
# Description: collate function for batch processing
"""

import torch
from transformers import DataCollatorForTokenClassification

from .batch import unpack_instances, Batch

class DataCollator(DataCollatorForTokenClassification):
    label2id = {"O": 0, "B-PER": 1,"I-PER": 2 , "B-LOC": 3,"I-LOC": 4,"B-ORG": 5,"I-ORG": 6,"B-MISC": 7,"I-MISC": 8}
    id2label = {v: k for k, v in label2id.items()}

    def __call__(self, instance_list: list[dict]):
        tk_ids, attn_masks, lbs = unpack_instances(instance_list, ["bert_tk_ids", "bert_attn_masks", "bert_lbs"])
        # Convert string labels to integers
        lbs = [[self.label2id.get(label, label) if isinstance(label, str) else label for label in lb] for lb in lbs]

        # Update `tk_ids`, `attn_masks`, and `lbs` to match the maximum length of the batch.
        # The updated type of the three variables should be `torch.int64``.
        # Hint: some functions and variables you may want to use: `self.tokenizer.pad()`, `self.label_pad_token_id`.
        # --- TODO: start of your code ---

        # --- TODO: start of your code ---
        max_length = max([len(ids) for ids in tk_ids])

        # Padding for `tk_ids` and `attn_masks`
        padded_inputs = self.tokenizer.pad(
            {"input_ids": tk_ids, "attention_mask": attn_masks},
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

        tk_ids = padded_inputs["input_ids"].to(torch.int64)
        attn_masks = padded_inputs["attention_mask"].to(torch.int64)

        # Padding for `lbs`
        lbs = [lb + [self.label_pad_token_id] * (max_length - len(lb)) for lb in lbs]
        lbs = torch.tensor(lbs, dtype=torch.int64)

        # --- TODO: end of your code ---

        return Batch(input_ids=tk_ids, attention_mask=attn_masks, labels=lbs)
