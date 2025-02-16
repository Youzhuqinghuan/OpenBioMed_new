from abc import ABC, abstractmethod
from typing import Any, Dict, List
from torch_geometric.data import Data, Batch
import torch

from transformers import AutoTokenizer, DataCollatorWithPadding
class Collator(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, inputs: List[Any]) -> Any:
        raise NotImplementedError
    
    def _collate_single(self, data):
        if isinstance(data[0], Data):
            return Batch.from_data_list(data)
        elif torch.is_tensor(data[0]):
            return torch.stack([x.squeeze() for x in data])
        elif isinstance(data[0], int):
            return torch.tensor(data).view((-1, 1))

class MoleculeCollatorWithPadding(Collator):
    def __init__(self, tokenizer, max_length=512, padding=True, mask=False):
        """
        Args:
            tokenizer (MolEncTokenizer): The tokenizer for molecule sequences.
            padding (bool): Whether or not to apply padding.
            mask (bool): Whether or not to apply masking to the sequences.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.mask = mask

    def __call__(self, batch: List[Dict]):
        """
        Collates a batch of molecule data by applying tokenization, masking, and padding.

        Args:
            batch (List[Dict]): A list of Dict items where each item contains molecule tokens and smiles.

        Returns:
            Dict: Collated batch, including padded and masked tokens.
        """       
        pad_length = max([len(seq) for item in batch for seq in item['original_tokens']])
        for item in batch:
            for i in range(len(item['original_tokens'])):
                n_pad = pad_length - len(item['original_tokens'][i])
                item['original_tokens'][i] += [self.tokenizer.pad_token] * n_pad
                item['masked_pad_masks'][i] += [1] * n_pad
        
        all_token_ids = []
        all_pad_masks = []
        
        for item in batch:
            token_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(item['original_tokens'])).T
            pad_mask = torch.tensor(item['masked_pad_masks']).bool().T
            token_ids = token_ids[:self.max_length]
            pad_mask = pad_mask[:self.max_length]
            all_token_ids.append(token_ids)
            all_pad_masks.append(pad_mask)
        token_ids_tensor = torch.cat(all_token_ids, dim=1)
        pad_mask_tensor = torch.cat(all_pad_masks, dim=1)
        return {
            "encoder_input": token_ids_tensor,
            "encoder_pad_mask": pad_mask_tensor
        }

class PygCollator(Collator):
    def __init__(self, follow_batch: List[str]=[], exclude_keys: List[str]=[]) -> None:
        super().__init__()
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, inputs: List[Data]) -> Batch:
        return Batch.from_data_list(inputs, follow_batch=self.follow_batch, exclude_keys=self.exclude_keys)

class ListCollator(Collator):
    def __call__(self, inputs: List[Any]) -> Any:
        return inputs

class ClassLabelCollator(Collator):
    def __call__(self, inputs: List[Any]) -> Any:
        batch = torch.stack(inputs)
        return batch


class DPCollator(Collator):
    def __init__(self):
        super(DPCollator, self).__init__()

    def __call__(self, mols):
        batch = self._collate_single(mols)
        return batch


class EnsembleCollator(Collator):
    def __init__(self, to_ensemble: Dict[str, Collator]) -> None:
        super().__init__()
        self.collators = {}
        for k, v in to_ensemble.items():
            self.collators[k] = v

    def __call__(self, inputs: List[Dict[str, Any]]) -> Dict[Any, Any]:
        collated = {}
        for k in inputs[0]:
            collated[k] = self.collators[k]([item[k] for item in inputs])
        return collated

    def get_attrs(self) -> List[str]:
        return list(self.collators.keys())