from typing import Dict, List
from typing_extensions import Any
import os
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_float32_matmul_precision('medium')
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
    state_dict = torch.load(input_model_path)
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

        self.device = model_cfg.device
        self.text_model = self.text_model.to(self.device)
        self.molecule_model = self.molecule_model.to(self.device)
        self.text2latent = self.text2latent.to(self.device)
        self.mol2latent = self.mol2latent.to(self.device)
        self.generation2foundation = self.generation2foundation.to(self.device)
        self.foundation2generation = self.foundation2generation.to(self.device)
        self.fusion = FeatureFusionModule(
            emb_dim=model_cfg.SSL_emb_dim,
            n_layers=model_cfg.fusion_n_layer,
            n_heads=model_cfg.fusion_n_head,
            max_seq_len=model_cfg.smiles_max_length
        ).to(self.device)
        
        for param in self.text_model.parameters():
            param.requires_grad = False
        # for param in self.molecule_model.parameters():
        #     param.requires_grad = False
        for param in self.text2latent.parameters():
            param.requires_grad = False
        for param in self.mol2latent.parameters():
            param.requires_grad = False
        for param in self.generation2foundation.parameters():
            param.requires_grad = False
        for param in self.foundation2generation.parameters():
            param.requires_grad = False
        
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
        device = self.device
        for key in text:
            text[key] = text[key].to(device)
        text_output = self.text_model(input_ids=text['input_ids'], attention_mask=text['attention_mask'])
        text_repr = text_output["pooler_output"] # [batch_size, emb_size(768)] for [CLS]
        text_repr = self.foundation2generation(self.text2latent(text_repr)) # [batch_size, emb_size(256)]

        for key in molecule:
            molecule[key] = molecule[key].to(device)

        molecule_output = self.molecule_model.encode(molecule) # [seq_len, batch_size, emb_size(256)]
        molecule_pad_mask = molecule['encoder_pad_mask']

        for key in label:
            label[key] = label[key].to(device)
            
        label_output = self.molecule_model.encode(label) # [seq_len, batch_size, emb_size(256)]
        label_pad_mask = label['encoder_pad_mask']
        
        fused_repr = self.fusion(text_repr, molecule_output, molecule_pad_mask) # [seq_len, batch_size, emb_size(256)]
        
        fused_repr = mean_pooling(fused_repr, molecule_pad_mask) # [batch_size, emb_size(256)]
        fused_repr = F.normalize(fused_repr, dim=-1)
        label_repr = mean_pooling(label_output, label_pad_mask) # [batch_size, emb_size(256)]
        label_repr = F.normalize(label_repr, dim=-1)
        cos_loss = 1 - F.cosine_similarity(fused_repr, label_repr, dim=-1)
        cos_loss = cos_loss.mean()

        return {
            "loss": cos_loss
        }

    def predict_text_based_molecule_editing(self, 
        molecule: Featurized[Molecule], 
        text: Featurized[Text],
    ) -> List[Molecule]:
        device = self.device
        for key in text:
            text[key] = text[key].to(device)
        text_output = self.text_model(input_ids=text['input_ids'], attention_mask=text['attention_mask'])
        text_repr = text_output["pooler_output"] # [batch_size, emb_size(768)] for [CLS]
        text_repr = self.foundation2generation(self.text2latent(text_repr)) # [batch_size, emb_size(256)]

        for key in molecule:
            molecule[key] = molecule[key].to(device)
        molecule_output = self.molecule_model.encode(molecule) # [seq_len, batch_size, emb_size(256)]
        molecule_pad_mask = molecule['encoder_pad_mask']
        _, batch_size = molecule['encoder_input'].shape

        fused_repr = self.fusion(text_repr, molecule_output, molecule_pad_mask) # [seq_len, batch_size, emb_size(256)]

        regenerate_mols = self.MegaMolBART_wrapper.inverse_transform(
            fused_repr.to(device), 
            molecule['encoder_pad_mask'].bool(),
            batch_size, k=self.cfg.predict.num_beams, sanitize=True, device=self.device
            )

        return [Molecule.from_smiles(smi) for smi in regenerate_mols]
    
class FeatureFusionModule(nn.Module):
    def __init__(self, 
                 emb_dim=256, 
                 n_layers=2,
                 n_heads=4,
                 max_seq_len=512):
        super().__init__()
        
        self.register_buffer(
            "position_emb",
            self._create_sinusoidal_emb(max_seq_len*2, emb_dim)
        )
        
        self.self_attn = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=emb_dim,
                nhead=n_heads,
                dim_feedforward=emb_dim*4,
                dropout=0.1,
                batch_first=False
            ),
            num_layers=n_layers
        )
        
    def _create_sinusoidal_emb(self, max_len, dim):
        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(1)  # [max_len, 1, dim]

    def forward(self, text_repr, mol_repr, molecule_pad_mask):
        """
        text_repr: [batch_size, 256] 文本全局特征
        mol_repr: [seq_len, batch_size, 256] 分子序列特征
        molecule_pad_mask: [seq_len, batch_size]
        return: [seq_len, batch_size, 256]
        """
        seq_len = mol_repr.size(0)
        batch_size = mol_repr.size(1)
        text_mask = torch.zeros(seq_len, batch_size, 
                              dtype=torch.bool, device=mol_repr.device)
        full_mask = torch.cat([text_mask, molecule_pad_mask], dim=0).transpose(0, 1) # [bs, 2seq]

        text_seq = text_repr.unsqueeze(0).expand(seq_len, -1, -1)  # [seq, bs, 256]
        
        concat_seq = torch.cat([text_seq, mol_repr], dim=0)  # [2seq, bs, 256]
        
        positions = self.position_emb[:concat_seq.size(0)]  # [2seq, 1, dim]
        concat_seq = concat_seq + positions

        attn_output = self.self_attn(
            concat_seq,
            src_key_padding_mask=full_mask
            )  # [2seq, bs, 256]
        
        fused_mol = attn_output[seq_len:]  # [seq, bs, 256]
        
        return fused_mol
    
# bash scripts/train.sh text_based_molecule_editing moleculestm fs_mol_edit 0 > /data/hucp/output.log 2>&1
