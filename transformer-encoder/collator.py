from typing import Any, Dict, List, Tuple
import torch

from transformers import PreTrainedTokenizerBase

class ClassificationCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int, fields: Tuple[str, ...], label_field: str):
        self.tok = tokenizer
        self.max_length = max_length
        self.fields = fields
        self.label_field = label_field

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Prefer pre-tokenized path when "input_ids" are present
        if "input_ids" in batch[0]:
            ids_list = [b["input_ids"] for b in batch]
            ids_list = [ids[: self.max_length] for ids in ids_list]
            enc = self.tok.pad([{"input_ids": ids} for ids in ids_list], padding=True, return_tensors="pt")
        else:
            if len(self.fields) == 1:
                texts = [b[self.fields[0]] for b in batch]
                enc = self.tok(texts, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt")
            else:
                t1 = [b[self.fields[0]] for b in batch]
                t2 = [b[self.fields[1]] for b in batch]
                enc = self.tok(t1, t2, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt")
        labels = torch.tensor([b[self.label_field] for b in batch], dtype=torch.long)
        return {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask, "labels": labels}


class RegressionCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int, fields: Tuple[str, ...]):
        self.tok = tokenizer
        self.max_length = max_length
        self.fields = fields

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Prefer pre-tokenized path when "input_ids" are present
        if "input_ids" in batch[0]:
            ids_list = [b["input_ids"] for b in batch]
            ids_list = [ids[: self.max_length] for ids in ids_list]
            enc = self.tok.pad([{"input_ids": ids} for ids in ids_list], padding=True, return_tensors="pt")
        else:
            # Fallback: tokenize from raw text fields
            texts = [b[self.fields[0]] for b in batch]
            enc = self.tok(texts, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt")
        # labels: list[float] per example -> tensor (B, D)
        labels = torch.tensor([b["labels"] for b in batch], dtype=torch.float)
        return {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask, "labels": labels}
