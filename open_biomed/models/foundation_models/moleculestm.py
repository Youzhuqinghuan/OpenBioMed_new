from typing import Dict, List
from typing_extensions import Any
import os

import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorWithPadding
from transformers.modeling_outputs import BaseModelOutput
from transformers import AutoModel, AutoTokenizer

from open_biomed.data import Molecule, Text
from open_biomed.models.task_models import TextBasedMoleculeEditingModel, MoleculeCaptioningModel, TextGuidedMoleculeGenerationModel, MoleculeQAModel, ProteinQAModel
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import MolMoleculeSTMFeaturizer, TextMoleculeSTMFeaturizer, Featurized
from open_biomed.utils.misc import concatenate_tokens
from open_biomed.utils.mega_molbart.mega_mol_bart import MegaMolBART
from open_biomed.utils.mlp import MLP
from open_biomed.utils.collator import MoleculeCollatorWithPadding

def load_language_molecule_and_edit_models(model_cfg: Config):
    text_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=model_cfg.scibert_path)
    text_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=model_cfg.scibert_path)
    text_dim = 768

    input_model_path = os.path.join(model_cfg.MoleculeSTM_model_dir, "text_model.pth")
    print("Loading from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    text_model.load_state_dict(state_dict, strict=False) # Use 'strict=False' to ignore embeddings.position_ids

    # This is loading from the pretarined_MegaMolBART
    MegaMolBART_wrapper = MegaMolBART(vocab_path=model_cfg.vocab_path, input_dir=model_cfg.MegaMolBART_generation_model_dir, output_dir=None)   
    molecule_model = MegaMolBART_wrapper.model
    
    print("Loading from pretrained MegaMolBART ({}).".format(model_cfg.MegaMolBART_generation_model_dir))
    molecule_dim_generation = 256
    if model_cfg.MoleculeSTM_molecule_type == "SMILES":  # For MegaMolBART
        molecule_dim_MoleculeSTM = 256
    else:  # For GIN
        molecule_dim_MoleculeSTM = 300

    text2latent = nn.Linear(text_dim, model_cfg.SSL_emb_dim)
    input_model_path = os.path.join(model_cfg.MoleculeSTM_model_dir, "text2latent_model.pth")
    print("Loading from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path)
    text2latent.load_state_dict(state_dict)
    
    mol2latent = nn.Linear(molecule_dim_MoleculeSTM, model_cfg.SSL_emb_dim)
    input_model_path = os.path.join(model_cfg.MoleculeSTM_model_dir, "mol2latent_model.pth")
    print("Loading from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path)
    mol2latent.load_state_dict(state_dict)

    # generation2MoleculeSTM = nn.Linear(molecule_dim_generation, model_cfg.SSL_emb_dim)
    generation2MoleculeSTM = MLP(molecule_dim_generation, [model_cfg.SSL_emb_dim, model_cfg.SSL_emb_dim])
    input_model_path = os.path.join(model_cfg.language_edit_model_dir, "generation2foundation_model.pth")
    print("Loading from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path)
    generation2MoleculeSTM.load_state_dict(state_dict)

    # MoleculeSTM2generation = nn.Linear(model_cfg.SSL_emb_dim, molecule_dim_generation)
    MoleculeSTM2generation = MLP(model_cfg.SSL_emb_dim, [molecule_dim_generation, molecule_dim_generation])
    input_model_path = os.path.join(model_cfg.language_edit_model_dir, "foundation2generation_model.pth")
    print("Loading from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path)
    MoleculeSTM2generation.load_state_dict(state_dict)

    return text_model, text_tokenizer, text_dim, molecule_model, MegaMolBART_wrapper, molecule_dim_generation, text2latent, mol2latent, generation2MoleculeSTM, MoleculeSTM2generation

def mean_pooling(token_embeddings, attention_mask):
    attention_mask = ~attention_mask
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float() # [seq_len, batch_size, emb_size(256)]
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 0) # [batch_size, emb_size(256)]
    sum_mask = torch.clamp(input_mask_expanded.sum(0), min=1e-9) # [batch_size, emb_size(256)]
    return sum_embeddings / sum_mask

class MoleculeSTM(TextBasedMoleculeEditingModel):
    def __init__(self, model_cfg: Config) -> None:
        super(MoleculeSTM, self).__init__(model_cfg)
        self.cfg = model_cfg
        self.text_model, self.text_tokenizer, self.text_dim, self.molecule_model, self.MegaMolBART_wrapper, self.molecule_dim, \
            self.text2latent, self.mol2latent, self.generation2foundation, self.foundation2generation = load_language_molecule_and_edit_models(model_cfg)
        
        for param in self.text_model.parameters():
            param.requires_grad = False
        for param in self.molecule_model.parameters():
            param.requires_grad = False
        for param in self.text2latent.parameters():
            param.requires_grad = False
        for param in self.mol2latent.parameters():
            param.requires_grad = False

        # Just for debug
        device = 'cuda'
        self.text_model = self.text_model.to(device)
        self.molecule_model = self.molecule_model.to(device)
        self.text2latent = self.text2latent.to(device)
        self.mol2latent = self.mol2latent.to(device)
        self.generation2foundation = self.generation2foundation.to(device)
        self.foundation2generation = self.foundation2generation.to(device)
        
        self.featurizers = {
            "molecule": MolMoleculeSTMFeaturizer(
                tokenizer=self.MegaMolBART_wrapper.tokenizer, 
                base='SMILES',
            ),
            "text": TextMoleculeSTMFeaturizer(
                tokenizer=self.text_tokenizer,
            )
        }
        self.collators = {
            "molecule": MoleculeCollatorWithPadding(self.MegaMolBART_wrapper.tokenizer, padding=True),
            "text": DataCollatorWithPadding(self.text_tokenizer, padding=True),
        }
        for parent in reversed(type(self).__mro__[1:-1]):
            if hasattr(parent, '_add_task'):
                parent._add_task(self)

    def forward_text_based_molecule_editing(self, 
        molecule: Featurized[Molecule], 
        text: Featurized[Text], 
        label: Featurized[Molecule],
    ) -> Dict[str, torch.Tensor]:
        # TODO: foward
        print('forward_text_based_molecule_editing: ')
        
        device = 'cuda'
        for key in text:
            text[key] = text[key].to(device)
        text_output = self.text_model(input_ids=text['input_ids'], attention_mask=text['attention_mask'])
        text_repr = text_output["pooler_output"] # [batch_size, emb_size(768)] for [CLS]
        text_repr = self.text2latent(text_repr) # [batch_size, emb_size(256)]

        for key in molecule:
            molecule[key] = molecule[key].to(device)

        molecule_output = self.molecule_model.encode(molecule) # [seq_len, batch_size, emb_size(256)]
        molecule_pad_mask = molecule['encoder_pad_mask'].detach().clone()
        molecule_repr = mean_pooling(molecule_output, molecule_pad_mask) # [batch_size, emb_size(256)]
        molecule_repr = self.generation2foundation(molecule_repr)

        for key in label:
            label[key] = label[key].to(device)
            
        label_output = self.molecule_model.encode(label) # [seq_len, batch_size, emb_size(256)]
        label_pad_mask = label['encoder_pad_mask'].detach().clone()
        label_repr = mean_pooling(label_output, label_pad_mask) # [batch_size, emb_size(256)]
        label_repr = self.generation2foundation(label_repr)
        
        return
        # TODO: Just for debug
        device = 'cuda'
        
        for key in text:
            text[key] = text[key].to(device)
        text_output = self.text_model(input_ids=text['input_ids'].squeeze(), attention_mask=text['attention_mask'].squeeze())
        text_repr = text_output["pooler_output"]
        text_repr = self.text2latent(text_repr)
        text_repr = text_repr.to(device)

        molecule, molecule_smiles = molecule['encoder_input'], molecule['smiles']
        print(molecule_smiles)
        for key in molecule:
            molecule[key] = molecule[key].to(device)
        molecule_output = self.molecule_model.encode(molecule)

        regenerate_mols = self.MegaMolBART_wrapper.inverse_transform(
            [molecule_output.to(device)], 
            molecule['encoder_pad_mask'].bool(),
            self.cfg.batch_size_train, k=1, sanitize=True
            )
        print(regenerate_mols)
        # molecule_repr = molecule_output[0, :, :].to(device)
        molecule_repr = molecule_output.to(device)
        molecule_repr = self.mol2latent(molecule_repr)
        molecule_repr = molecule_repr.to(device)
        # molecule_output = self.foundation2generation(molecule_repr)
        # regenerate_mols = self.MegaMolBART_wrapper.inverse_transform(
        #     [molecule_output.to(device)], 
        #     molecule['encoder_pad_mask'].bool(),
        #     self.cfg.batch_size_train, k=1, sanitize=True
        #     )
        # print(regenerate_mols)
        
        # incorrect completion
        # fused_repr = molecule_repr + text_repr
        # fused_repr = self.foundation2generation(fused_repr)
        # regenerate_mols = self.MegaMolBART_wrapper.inverse_transform(
        #     [fused_repr.to(device)], 
        #     molecule['encoder_pad_mask'].bool(),
        #     self.cfg.batch_size_train, k=1, sanitize=True
        #     )
        # print(regenerate_mols)
        return

    def predict_text_based_molecule_editing(self, 
        molecule: Featurized[Molecule], 
        text: Featurized[Text],
    ) -> List[Molecule]:
        # TODO: predict
        # concatenated = concatenate_tokens([molecule, text])
        # encoder_outputs = BaseModelOutput(last_hidden_state=self.main_model.encoder(**concatenated).last_hidden_state)
        # decoder_outputs = self.main_model.generate(
        #     encoder_outputs=encoder_outputs,
        #     attention_mask=concatenated.attention_mask,
        #     **self.config.predict.todict(),
        # )
        # preds = self.tokenizer.batch_decode(decoder_outputs, skip_special_tokens=True)
        # return [Molecule.from_smiles(smi) for smi in preds]
        return
    
