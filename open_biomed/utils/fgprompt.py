from typing import Dict, List
from typing_extensions import Any

import pytorch_lightning as pl
from transformers import T5Tokenizer, T5EncoderModel, DataCollatorWithPadding, T5ForConditionalGeneration, BatchEncoding
from transformers.modeling_outputs import BaseModelOutput
from rdkit import Chem
from rdkit.Chem import BRICS
import json
import torch
from transformers import T5Tokenizer
from open_biomed.utils.misc import concatenate_tokens

class FGPromptGenerator(pl.LightningModule):
    def __init__(self, model_name='seyonec/ChemT5-small', lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.main_model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # self.lr = lr

    def forward(self, molecule, text, fgprompt):
        concatenated = concatenate_tokens([molecule, text])
        encoder_outputs = BaseModelOutput(last_hidden_state=self.main_model.encoder(**concatenated).last_hidden_state)
        loss = self.main_model(
            encoder_outputs=encoder_outputs,
            attention_mask=concatenated.attention_mask,
            decoder_attention_mask=fgprompt.attention_mask,
            labels=fgprompt.input_ids
        ).loss
        return loss
    
    def predict(self, molecule, text, max_length=256):
        concatenated = concatenate_tokens([molecule, text])
        return self.main_model.encoder(**concatenated).last_hidden_state

    def generate(self, molecule, text, max_length=256):
        concatenated = concatenate_tokens([molecule, text])
        outputs = self.main_model.generate(
            inputs_embeds=self.main_model.encoder(**concatenated).last_hidden_state,
            attention_mask=concatenated.attention_mask,
            max_length=max_length,
            num_beams=5,
            early_stopping=True
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def training_step(self, batch, batch_idx):
        molecule = batch["molecule"]
        text = batch["text"]
        fgprompt = batch["fgprompt"]

        loss = self(molecule, text, fgprompt)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        molecule = batch["molecule"]
        text = batch["text"]
        fgprompt = batch["fgprompt"]

        loss = self(molecule, text, fgprompt)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        molecule = batch["molecule"]
        text = batch["text"]
        fgprompt = batch["fgprompt"]

        loss = self(molecule, text, fgprompt)
        preds = self.generate(molecule, text)
        # 保存预测结果
        self._save_predictions(preds, batch)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def _save_predictions(self, preds, batch):
        with open("./predictions.jsonl", "a") as f: 
            for i in range(len(preds)):
                original = self.tokenizer.decode(batch["molecule"].input_ids[i]).replace("<pad>", "").strip()
                text = self.tokenizer.decode(batch["text"].input_ids[i]).replace("<pad>", "").strip()
                true_fgprompt = self.tokenizer.decode(batch["fgprompt"].input_ids[i]).replace("<pad>", "").strip()
                pred_fgprompt = preds[i].replace("<pad>", "").strip() if isinstance(preds[i], str) else preds[i]
                result = {
                    "original": original,
                    "text": text,
                    "pred_fgprompt": pred_fgprompt,
                    "true_fgprompt": true_fgprompt
                }
                f.write(json.dumps(result) + "\n")


    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0)