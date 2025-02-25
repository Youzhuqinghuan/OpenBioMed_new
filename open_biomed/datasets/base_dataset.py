from abc import ABC, abstractmethod
from functools import wraps
from typing import List, Optional, Tuple
from typing_extensions import Self, Any

import copy
from torch.utils.data import Dataset
from tqdm import tqdm

from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import Featurizer, Featurized

# def assign_split(func):
#     @wraps(func)
#     def wrapper(self, **kwargs):
#         train, valid, test = func(self, **kwargs)
#         if train is not None:
#             train.split = "train"
#         if valid is not None:
#             valid.split = "valid"
#         test.split = "test"
#         return train, valid, test
#     return wrapper

def assign_split(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):  # 修改为接收*args和**kwargs
        # Perform split assignment after splitting the dataset
        train, valid, test = func(self, *args, **kwargs)  # 传递所有参数到原函数
        if train is not None:
            train.split = "train"
        if valid is not None:
            valid.split = "valid"
        test.split = "test"
        return train, valid, test
    return wrapper

# def featurize_with_label(retain_label=True):
#     def decorator(func):
#         @wraps(func)
#         def wrapper(self, index: int) -> Featurized[Any]:
#             kwargs = func(self, index)
#             if not getattr(self, "split") == "train" and not retain_label:
#                 kwargs.pop("label")
        
#             return self.featurizer(**kwargs)
        
#         return wrapper
#     return decorator

def featurize_with_label(func):
    @wraps(func)
    def wrapper(self, index: int) -> Featurized[Any]:
        # Perform featurization after the __getitem__() function of a dataset
        kwargs = func(self, index)
        # We skip featurizing labels for validation and test sets
        # if not getattr(self, "split") == "train":
        #     kwargs.pop("label")
        return self.featurizer(**kwargs)
    return wrapper

def featurize(func):
    @wraps(func)
    def wrapper(self, index: int) -> Featurized[Any]:
        # Perform featurization after the __getitem__() function of a dataset
        kwargs = func(self, index)
        # We skip featurizing labels for validation and test sets
        if not getattr(self, "split") == "train":
            kwargs.pop("label")
        return self.featurizer(**kwargs)
    return wrapper

class BaseDataset(Dataset, ABC):
    def __init__(self, cfg: Config, featurizer: Featurizer) -> None:
        super(BaseDataset, self).__init__()
        self.cfg = cfg
        self.featurizer = featurizer
        self._load_data()

    def get_subset(self, indexes: List[int], attrs: List[str]) -> Self:
        new_dataset = copy.deepcopy(self)
        for attr in attrs:
            values = getattr(new_dataset, attr)
            new_dataset.__setattr__(attr, [values[i] for i in indexes])
        return new_dataset

    @abstractmethod
    def _load_data(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def split(self, split_cfg: Optional[Config]=None) -> Tuple[Any, Any, Any]:
        raise NotImplementedError

    def save(self, file: str, format: str='lmdb') -> None:
        raise NotImplementedError

    @classmethod
    def from_file(cls, file: str, format: str='lmdb') -> Self:
        raise NotImplementedError

    @featurize
    def __getitem__(self, index) -> Any:
        raise NotImplementedError