from typing import Dict, List
from typing_extensions import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorWithPadding, BatchEncoding
from transformers.modeling_outputs import BaseModelOutput

from open_biomed.data import Molecule, Text
from open_biomed.models.task_models import TextBasedMoleculeEditingModel, MoleculeCaptioningModel, TextGuidedMoleculeGenerationModel, MoleculeQAModel, ProteinQAModel
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import MoleculeTransformersFeaturizer, TextTransformersFeaturizer, MolMoleculeSTMFeaturizer, Featurized
from open_biomed.utils.mega_molbart.mega_mol_bart import MegaMolBART
from open_biomed.utils.collator import MoleculeCollatorWithPadding
from open_biomed.utils.misc import concatenate_tokens
from open_biomed.utils.fgprompt import FGPromptGenerator

class UnName(TextBasedMoleculeEditingModel):
    def __init__(self, model_cfg: Config) -> None:
        super(UnName, self).__init__(model_cfg)
        self.device = model_cfg.device
        self.main_model = T5ForConditionalGeneration.from_pretrained(model_cfg.hf_model_name_or_path).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_cfg.hf_model_name_or_path)
        self.fg_model = FGPromptGenerator.load_from_checkpoint(
            model_cfg.fg_path,
            model_name=model_cfg.hf_model_name_or_path
        ).to(self.device)
        for param in self.fg_model.parameters():
            param.requires_grad = False
        for param in self.fg_model.main_model.encoder.parameters():
            param.requires_grad = True
        self.alpha = nn.Parameter(torch.tensor(0.5)).to(self.device)
        self.featurizers = {
            "molecule": MoleculeTransformersFeaturizer(
                tokenizer=model_cfg.hf_model_name_or_path, 
                max_length=model_cfg.smiles_max_length,
                base='SMILES',
            ),
            "text": TextTransformersFeaturizer(
                tokenizer=model_cfg.hf_model_name_or_path,
                max_length=model_cfg.text_max_length,
            )
        }
        self.collators = {
            "molecule": DataCollatorWithPadding(self.tokenizer, padding=True),
            "text": DataCollatorWithPadding(self.tokenizer, padding=True),
        }
        for parent in reversed(type(self).__mro__[1:-1]):
            if hasattr(parent, '_add_task'):
                parent._add_task(self)

    def forward_text_based_molecule_editing(self, 
        molecule: Featurized[Molecule], 
        text: Featurized[Text], 
        label: Featurized[Molecule],
    ) -> Dict[str, torch.Tensor]:
        device = self.device
        for key in text:
            text[key] = text[key].to(device)
        for key in molecule:
            molecule[key] = molecule[key].to(device)
        for key in label:
            label[key] = label[key].to(device)
        concatenated = concatenate_tokens([molecule, text])
        main_enc = self.main_model.encoder(**concatenated).last_hidden_state
        fg_enc = self.fg_model.main_model.encoder(**concatenated).last_hidden_state
        fused = main_enc + (self.alpha) * fg_enc
        encoder_outputs = BaseModelOutput(last_hidden_state=fused)
        # encoder_outputs = BaseModelOutput(last_hidden_state=self.main_model.encoder(**concatenated).last_hidden_state)
        return {"loss": self.main_model(
            encoder_outputs=encoder_outputs,
            attention_mask=concatenated.attention_mask,
            decoder_attention_mask=label.attention_mask,
            labels=label.input_ids
        ).loss}

    @torch.no_grad()
    def predict_text_based_molecule_editing(self, 
        molecule: Featurized[Molecule], 
        text: Featurized[Text],
    ) -> List[Molecule]:
        device = self.device
        for key in text:
            text[key] = text[key].to(device)
        for key in molecule:
            molecule[key] = molecule[key].to(device)
        concatenated = concatenate_tokens([molecule, text])
        main_enc = self.main_model.encoder(**concatenated).last_hidden_state
        fg_enc = self.fg_model.main_model.encoder(**concatenated).last_hidden_state
        fused = main_enc + (self.alpha) * fg_enc
        encoder_outputs = BaseModelOutput(last_hidden_state=fused)
        # encoder_outputs = BaseModelOutput(last_hidden_state=self.main_model.encoder(**concatenated).last_hidden_state)
        decoder_outputs = self.main_model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=concatenated.attention_mask,
            **self.config.predict.todict(),
        )
        preds = self.tokenizer.batch_decode(decoder_outputs, skip_special_tokens=True)
        return [Molecule.from_smiles(smi) for smi in preds]