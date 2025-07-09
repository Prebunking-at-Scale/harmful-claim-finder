# Load a model and use it to tag data
from typing import Any, Dict, List

import torch
from pydantic import BaseModel
from transformers import BertTokenizer

from harmful_claim_finder.claim_type_detector import utils


class ClaimTypeResult(BaseModel):
    types_detected: List[str]
    type_scores: dict[str, Any]

    def to_json(self) -> Dict[str, Any]:
        return {
            "detected_claims": self.types_detected,
            "claims_type_scores": self.type_scores,
        }


class ClaimTypeResults(BaseModel):
    results: List[ClaimTypeResult]

    def to_json(self) -> List[Dict[str, Any]]:
        return [result.to_json() for result in self.results]


class ClaimTyper:
    def __init__(self, target_list: list[str]):
        self.max_len = 256

        self.target_list = target_list
        num_classes = len(target_list)
        self.model = utils.BERTClass(num_classes)
        self.model.to(utils.device)
        self.tokenizer = BertTokenizer.from_pretrained(
            utils._BERT_TO_DOWNLOAD_LOC, local_files_only=True
        )
        self.optimizer = torch.optim.Adam(params=self.model.parameters())
        (
            self.model,
            self.optimizer,
            self.thresholds,
            checkpoint_epoch,
            valid_loss_min,
        ) = utils.load_checkpoint(utils.live_model_path, self.model, self.optimizer)

    def label_one(self, text: str) -> ClaimTypeResult:
        """Make claim-type predictions for a single piece text."""
        # Convert text into model-input format; split into tokens, convert them to token-ids, pad/truncate as needed etc.
        encodings = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        self.model.eval()  # make sure model is in eval model, so no gradient updates
        with torch.no_grad():  # again, no gradient updates. Not sure we need both?
            input_ids = encodings["input_ids"].to(utils.device, dtype=torch.long)
            attention_mask = encodings["attention_mask"].to(
                utils.device, dtype=torch.long
            )
            token_type_ids = encodings["token_type_ids"].to(
                utils.device, dtype=torch.long
            )
            output = self.model(input_ids, attention_mask, token_type_ids)
            # Raw output is from the BERT model, so we need to pass through a sigmoid layer before interpreting as a probability
            final_output = torch.sigmoid(output).cpu().detach().numpy().tolist()[0]
            # use a threshold for each class to convert probs into a list of 0/1's
            pred_scores = {
                label: prob for label, prob in zip(self.target_list, final_output)
            }
            pred_classes = [
                label
                for label, prob in zip(self.target_list, final_output)
                if prob > self.thresholds[label]
            ]
        return ClaimTypeResult(types_detected=pred_classes, type_scores=pred_scores)

    def label_batch(self, texts: list[str]) -> list[ClaimTypeResult]:
        """Add claim types to each of a list of texts."""

        encodings = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        self.model.eval()  # make sure model is in eval model, so no gradient updates
        with torch.no_grad():
            input_ids = encodings["input_ids"].to(utils.device, dtype=torch.long)
            attention_mask = encodings["attention_mask"].to(
                utils.device, dtype=torch.long
            )
            token_type_ids = encodings["token_type_ids"].to(
                utils.device, dtype=torch.long
            )
            outputs = self.model(input_ids, attention_mask, token_type_ids)
            # Raw output is from the BERT model, so we need to pass through a sigmoid layer before interpreting as a probability
            final_outputs = torch.sigmoid(outputs).cpu().detach().numpy().tolist()
            batch_results = []
            for i in range(len(texts)):
                pred_classes = [
                    label
                    for label, prob in zip(self.target_list, final_outputs[i])
                    if prob > self.thresholds[label]
                ]
                pred_scores = {
                    label: prob
                    for label, prob in zip(self.target_list, final_outputs[i])
                }
                batch_results.append(
                    ClaimTypeResult(
                        types_detected=pred_classes, type_scores=pred_scores
                    )
                )

        return batch_results
