from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, TypeVar

import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from functools import wraps
import torch
from transformers import AutoTokenizer, BatchEncoding

from open_biomed.data import Molecule, Protein, Pocket, Text

T = TypeVar('T', bound=Any)

class Featurized(Generic[T]):
    def __init__(self, value: T):
        self.value = value

class Featurizer(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_attrs(self) -> List[str]:
        raise NotImplementedError

class MoleculeFeaturizer(Featurizer):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, molecule: Molecule) -> Dict[str, Any]:
        raise NotImplementedError

    def get_attrs(self) -> List[str]:
        return ["molecule"]

class MoleculeTransformersFeaturizer(MoleculeFeaturizer):
    def __init__(self,
        tokenizer: str,
        max_length: int=512,
        add_special_tokens: bool=True,
        base: str='SMILES',
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, model_max_length=max_length, truncation=True)
        self.add_special_tokens = add_special_tokens
        self.base = base
        if base not in "SMILES" and "SELFIES":
            raise ValueError("{base} is not a valid 1D representaiton of molecules!")

    def __call__(self, molecule: Molecule) -> BatchEncoding:
        if self.base == "SMILES":
            molecule._add_smiles()
            parse_str = molecule.smiles
        if self.base == "SELFIES":
            molecule._add_selfies()
            parse_str = molecule.selfies
        return self.tokenizer(
            parse_str, 
            truncation=True, 
            add_special_tokens=self.add_special_tokens,
        )
        
# class MoleculeBatch:
#     def __init__(self, tokens=None, encoder_input=None, smiles=None):
#         self.tokens = tokens
#         self.encoder_input = encoder_input
#         self.smiles = smiles
        
class MolMoleculeSTMFeaturizer(MoleculeFeaturizer):
    def __init__(self,
        tokenizer: Any,
        max_length: int=512,
        add_special_tokens: bool=True,
        base: str='SMILES',
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.base = base
        if base not in "SMILES" and "SELFIES":
            raise ValueError("{base} is not a valid 1D representaiton of molecules!")

    def __call__(self, molecule: Molecule) -> BatchEncoding:
        if self.base == "SMILES":
            molecule._add_smiles()
            parse_str = molecule.smiles
        if self.base == "SELFIES":
            molecule._add_selfies()
            parse_str = molecule.selfies
        return self.tokenizer.tokenize(
            [parse_str], # Only one str in each call
            pad=True
        )
        # tokens = self.tokenizer.tokenize(
        #     [parse_str], # Only one str in each call
        #     pad=True
        # )
        # return MoleculeBatch(tokens=tokens, smiles=molecule.smiles)

class TextFeaturizer(Featurizer):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, text: Text) -> Dict[str, Any]:
        raise NotImplementedError

    def get_attrs(self) -> List[str]:
        return ["text"]

class TextTransformersFeaturizer(TextFeaturizer):
    def __init__(self, 
        tokenizer: str,
        max_length: int=512,
        add_special_tokens: bool=True,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, model_max_length=max_length, truncation=True)
        self.add_special_tokens = add_special_tokens

    def __call__(self, text: Text) -> BatchEncoding:
        return self.tokenizer(
            text.str, 
            truncation=True, 
            add_special_tokens=self.add_special_tokens,
        )

class TextMoleculeSTMFeaturizer(TextFeaturizer):
    def __init__(self, 
        tokenizer: Any,
        max_length: int=512,
        add_special_tokens: bool=True,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens

    def __call__(self, text: Text) -> BatchEncoding:
        return self.tokenizer(
            text.str, 
            max_length=self.max_length,
            truncation=True, 
            add_special_tokens=self.add_special_tokens,
        )
class PocketFeaturizer(Featurizer):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, pocket: Pocket) -> Dict[str, Any]:
        raise NotImplementedError

    def get_attrs(self) -> List[str]:
        return ["pocket"]

# For classification tasks, directly convert numbers or arrays into tensors.
class ClassLabelFeaturizer(Featurizer):
    def __init__(self) -> None:
        super().__init__()

    # Input a number or an array, and return a tensor.
    def __call__(self, label: np.array) -> torch.tensor:
        return  torch.tensor(label)

    def get_attrs(self) -> List[str]:
        return ["classlabel"]


class EnsembleFeaturizer(Featurizer):
    def __init__(self, to_ensemble: Dict[str, Featurizer]) -> None:
        super().__init__()
        self.featurizers = {}
        for k, v in to_ensemble.items():
            self.featurizers[k] = v

    def __call__(self, **kwargs: Any) -> Dict[Any, Any]:
        featurized = {}
        for k, v in kwargs.items():
            featurized[k] = self.featurizers[k](v)
        return featurized

    def get_attrs(self) -> List[str]:
        return list(self.featurizers.keys())

if __name__ == "__main__":
    from transformers import DataCollatorWithPadding
    featurizer = TextTransformersFeaturizer("./checkpoints/molt5/base")
    a = featurizer(text=Text.from_str("Hello"))
    b = featurizer(text=Text.from_str("Hello World"))
    collator = DataCollatorWithPadding(featurizer.tokenizer, padding=True)
    print(collator([a, b]))