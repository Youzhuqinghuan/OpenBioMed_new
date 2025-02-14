from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import BatchEncoding, PreTrainedTokenizer

def sub_dict(in_dict: dict[str, Any], keys_to_include: List[str]) -> dict[str, Any]:
    # Get a sub-dictionary based on keys_to_include
    return {k: in_dict[k] for k in keys_to_include}

def sub_batch_by_interval(start: int, end: int, **kwargs: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    # Construct a smaller batch from [start, end) of the original batch
    new_batch = {}
    for key in kwargs:
        new_batch[key] = kwargs[key][start:end]
    return new_batch

def safe_index(l: List[Any], e: Any) -> int:
    # Return index of element e in list l. If e is not present, return the last index
    try:
        return l.index(e)
    except:
        return len(l) - 1

def concatenate_tokens(tokens_to_concat: List[BatchEncoding]) -> BatchEncoding:
    # Concatenate multiple tokenized results by putting the non-padding tokens together
    batch_size = tokens_to_concat[0].input_ids.shape[0]
    concatenated = {
        "input_ids": torch.cat([tokens.input_ids for tokens in tokens_to_concat], dim=-1),
        "attention_mask": torch.cat([tokens.attention_mask for tokens in tokens_to_concat], dim=-1),
    }
    non_padding_length = concatenated["attention_mask"].sum(-1)
    max_length = non_padding_length.max().item()

    new_input_ids = []
    new_attention_mask = []
    for i in range(batch_size):
        perm = torch.cat([
            torch.where(concatenated["attention_mask"][i] == 1)[0],  # non-padding tokens
            torch.where(concatenated["attention_mask"][i] == 0)[0],  # padding tokens
        ])
        new_input_ids.append(concatenated["input_ids"][i][perm[:max_length]])
        new_attention_mask.append(concatenated["attention_mask"][i][perm[:max_length]])

    return BatchEncoding({
        "input_ids": torch.stack(new_input_ids, dim=0),
        "attention_mask": torch.stack(new_attention_mask, dim=0),
    })

def collate_objects_as_list(inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    outputs = {}
    for sample in inputs:
        for k, v in sample.items():
            if k not in outputs:
                outputs[k] = []
            outputs[k].append(v)
    return outputs