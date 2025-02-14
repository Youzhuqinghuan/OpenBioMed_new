from typing import Optional, Union
from typing_extensions import Any

from absl import logging
import json
import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from transformers import BertTokenizerFast

class Queue:
    def __init__(self, max_len=50):
        self.items = [1]
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)   

class GradientClip(pl.Callback):
    def __init__(self, max_grad_norm: Union[float, str]='Q', Q=Queue(3000)) -> None:
        super().__init__()
        # self.max_norm = max_norm
        self.gradnorm_queue = Q
        if max_grad_norm == 'Q':
            self.max_grad_norm = max_grad_norm
        else:
            self.max_grad_norm = float(max_grad_norm)

    def on_before_optimizer_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer: torch.optim.Optimizer) -> None:
        # zero graidents if they are not finite
        if self.max_grad_norm == 'Q':
            max_grad_norm = 1.5 * self.gradnorm_queue.mean() + 2 * self.gradnorm_queue.std()
            max_grad_norm = max_grad_norm.item()
        else:
            max_grad_norm = self.max_grad_norm
        grad_norm = torch.nn.utils.clip_grad_norm_(
            pl_module.parameters(), max_norm=max_grad_norm, norm_type=2.0
        )

        if self.max_grad_norm == 'Q':
            if float(grad_norm) > max_grad_norm:
                self.gradnorm_queue.add(float(max_grad_norm))
            else:
                self.gradnorm_queue.add(float(grad_norm))

        if float(grad_norm) > max_grad_norm:
            logging.info(
                f"Clipped gradient with value {grad_norm:.1f} "
                f"while allowed {max_grad_norm:.1f}",
            )
        pl_module.log_dict(
            {
                "grad_norm": grad_norm.item(),
                'max_grad_norm': max_grad_norm,
            },
            on_step=True,
            prog_bar=False,
            logger=True,
            batch_size=pl_module.train_cfg.batch_size,
        )

class RecoverCallback(pl.Callback):
    def __init__(self, latest_ckpt: str, recover_trigger_loss: float=1e3, resume: bool=False) -> None:
        super().__init__()
        self.latest_ckpt = latest_ckpt
        self.recover_trigger_loss = recover_trigger_loss
        self.resume = resume

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        super().setup(trainer, pl_module, stage)
        if os.path.exists(self.latest_ckpt) and self.resume:
            print(f"recover from checkpoint: {self.latest_ckpt}")
            checkpoint = torch.load(self.latest_ckpt)
            pl_module.load_state_dict(checkpoint["state_dict"])
            # pl_module.load_from_checkpoint(self.latest_ckpt)
        elif not os.path.exists(self.latest_ckpt) and self.resume:
            print(
                f"checkpoint {self.latest_ckpt} not found, training from scratch"
            )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if "loss" not in outputs:
            return None

        if outputs["loss"] > self.recover_trigger_loss:
            logging.warning(
                f"loss too large: {outputs}\n recovering from checkpoint: {self.latest_ckpt}"
            )
            if os.path.exists(self.latest_ckpt):
                checkpoint = torch.load(self.latest_ckpt)
                pl_module.load_state_dict(checkpoint["state_dict"])
            else:
                for layer in pl_module.children():
                    if hasattr(layer, "reset_parameters"):
                        layer.reset_parameters()
                logging.warning(
                    f"checkpoint {self.latest_ckpt} not found, training from scratch"
                )

        else:
            pass

class TextOverlapEvalCallback(pl.Callback):
    def __init__(self, tokenizer_type: Optional[str]="BERT") -> None:
        super().__init__()
        self.outputs = []
        self.eval_dataset = None
        self.tokenizer_type = tokenizer_type
        if tokenizer_type == "BERT":
            abs_path = os.path.abspath("./open_biomed/checkpoints/tokenizers/bert-base-uncased/")
            self.tokenizer = BertTokenizerFast.from_pretrained(abs_path)
            self.filter_tokens = ["[PAD]", "[CLS]", "[SEP]"]
        if tokenizer_type is None:
            self.tokenizer = None
            self.filter_tokens = []

    def on_validation_batch_end(self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        outputs: Optional[STEP_OUTPUT], 
        batch: Any, 
        batch_idx: int, 
        dataloader_idx: int = 0
    ) -> None:
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self.outputs.extend(outputs)

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        super().on_validation_epoch_start(trainer, pl_module)
        self.status = "val"
        self.outputs = []
        self.eval_dataset = trainer.val_dataloaders.dataset

    def on_validation_epoch_end(self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule
    ) -> None:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

        gts = self.eval_dataset.labels
        output_tokens = []
        gt_tokens = []
        meteor_scores = []
        rouge_scores_1 = []
        rouge_scores_2 = []
        rouge_scores_L = []

        for i in range(len(self.outputs)):
            orig_mol = self.eval_dataset.molecules[i]
            rouge = scorer.score(str(self.outputs[i]), str(gts[i]))
            rouge_scores_1.append(rouge['rouge1'].fmeasure)
            rouge_scores_2.append(rouge['rouge2'].fmeasure)
            rouge_scores_L.append(rouge['rougeL'].fmeasure)

            if self.tokenizer_type == "BERT":
                output_token = self.tokenizer.tokenize(str(self.outputs[i]), truncation=True, max_length=512, padding='max_length')
                gt_token = self.tokenizer.tokenize(str(gts[i]), truncation=True, max_length=512, padding='max_length')
            elif self.tokenizer_type is None:
                output_token = output_token.split(" ")
                gt_token = gt_token.split(" ")
            for token in self.filter_tokens:
                output_token = list(filter((token).__ne__), output_token)
                gt_token = list(filter((token).__ne__), gt_token)
            output_tokens.append(output_token)
            gt_tokens.append(gt_token)
            meteor_scores.append(meteor_score(gt_tokens[i], output_tokens[i]))

        bleu2 = corpus_bleu(gt_tokens, output_tokens, weights=(0.5, 0.5))
        bleu4 = corpus_bleu(gt_tokens, output_tokens, weights=(0.25, 0.25, 0.25, 0.25))
             
        output_str = "\t".join([str(orig_mol), str(self.outputs[i]), 
                                    f"{np.mean(bleu2):.4f}",  f"{np.mean(bleu4):.4f}",
                                    f"{np.mean(rouge_scores_1):.4f}", f"{np.mean(rouge_scores_2):.4f}",
                                    f"{np.mean(rouge_scores_L):.4f}"]) + "\n"

        output_path = os.path.join(trainer.default_root_dir, f"{self.status}_outputs", f"epoch{pl_module.current_epoch}")
        out_metrics = {
            f"{self.status}/BLEU-2": np.mean(bleu2),
            f"{self.status}/BLUE-4": np.mean(bleu4),
            f"{self.status}/Meteor": np.mean(meteor_scores),
            f"{self.status}/ROUGE-1": np.mean(rouge_scores_1),
            f"{self.status}/ROUGE-2": np.mean(rouge_scores_2),
            f"{self.status}/ROUGE-L": np.mean(rouge_scores_L)
        }
        pl_module.log_dict(out_metrics)
        print(json.dumps(out_metrics, indent=4))
        json.dump(out_metrics, open(output_path + "_metrics.json", "w"))
        
        with open(output_path + "_outputs.txt", "w") as f:
            f.write(output_str)

    def on_test_batch_end(self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        super().on_test_epoch_start(trainer, pl_module)
        self.status = "test"
        self.outputs = []
        self.eval_dataset = trainer.test_dataloaders.dataset

    def on_test_epoch_end(self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule
    ) -> None:
        self.on_validation_epoch_end(trainer, pl_module)