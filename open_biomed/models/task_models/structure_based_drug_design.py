from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import torch

from open_biomed.data import Molecule, Pocket
from open_biomed.models.base_model import BaseModel
from open_biomed.utils.collator import EnsembleCollator
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import EnsembleFeaturizer
from open_biomed.utils.misc import sub_dict

class SBDDModel(BaseModel, ABC):
    def __init__(self, model_cfg: Config) -> None:
        super().__init__(model_cfg)

    def _add_task(self) -> None:
        self.supported_tasks["sbdd"] = {
            "forward_fn": self.forward_sbdd,
            "predict_fn": self.predict_sbdd,
            "featurizer": EnsembleFeaturizer({
                **sub_dict(self.featurizers, ["pocket"]),
                "label": self.featurizers["molecule"]
            }),
            "collator": EnsembleCollator({
                **sub_dict(self.collators, ["pocket"]),
                "label": self.collators["molecule"]
            })
        }
    
    @abstractmethod
    def forward_sbdd(self, 
        pocket: List[Pocket], 
        label: List[Molecule],
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def predict_sbdd(self,
        pocket: List[Pocket], 
    ) -> List[Molecule]:
        raise NotImplementedError