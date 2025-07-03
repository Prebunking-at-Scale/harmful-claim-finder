import logging
import os
from google.cloud import storage
from pathlib import Path
import torch
from transformers import BertModel
from typing import Dict, Optional
from typing import Any, Tuple

_logger = logging.getLogger(__name__)
live_model_path = (Path(__file__).parent.absolute() / "prod_model.pt").resolve()

SERVICE_ACCOUNT_FILE_ENV = "CLAIM_TYPE_MODEL_SERVICE_ACCOUNT_FILE"
_CLAIM_TYPE_MODEL_FORCE_DOWNLOAD = "CLAIM_TYPE_MODEL_FORCE_DOWNLOAD"
_MODEL_FILE_TO_DOWNLOAD_LOC = live_model_path
_MODEL_BUCKET = "pas-ai-models"
_MODEL_BLOB = "claim-type-detector/v2023-10-03/prod_model.pt"
MODEL_NAME = "bert-base-multilingual-cased"

_BERT_MODEL_FORCE_DOWNLOAD = bool(os.environ.get("BERT_MODEL_FORCE_DOWNLOAD"))
_BERT_PRETRAINED_PREFIX = "bert-models/bert-base-multilingual-cased"
_BERT_TO_DOWNLOAD_LOC = (
    Path(__file__).parent.absolute() / _BERT_PRETRAINED_PREFIX
).resolve()

# Do we have a GPU (=CUDA) available?
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class BERTClass(torch.nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(BERTClass, self).__init__()
        self.bert_model = BertModel.from_pretrained(
            _BERT_TO_DOWNLOAD_LOC, local_files_only=True, return_dict=True
        )
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, num_classes)

    def forward(self, input_ids: Any, attn_mask: Any, token_type_ids: Any) -> Any:
        # define a forward pass through the model
        output = self.bert_model(
            input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output


def load_checkpoint(
    checkpoint_fpath: Path, model: BERTClass, optimizer: torch.optim.Adam
) -> Tuple[
    BERTClass,
    torch.optim.Adam,
    Dict[str, float],
    int,
    float,
]:
    """
    Load checkpoint (=model) from local filesystem.
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath, map_location=device, weights_only=False)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint["state_dict"])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint["optimizer"])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    # so we can carry on training and know if we improve further
    valid_loss_min = checkpoint["valid_loss_min"]
    # Thresholds are used to convert probabalitistic predictions to binary class labels
    thresholds = checkpoint["thresholds"]
    return model, optimizer, thresholds, checkpoint["epoch"], valid_loss_min


def save_checkpoint(state: Any, checkpoint_path: Any, suffix: Any = None) -> None:
    """
    This creates a single file containing everything needed to load & use a model.

    Save checkpoint (=model) to local filesystem.
    state: checkpoint we want to save
    checkpoint_path: path to save checkpoint
    """
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    if suffix:
        ext_pos = checkpoint_path.rfind(".")
        checkpoint_path = (
            checkpoint_path[0:ext_pos] + suffix + checkpoint_path[ext_pos:]
        )
    torch.save(state, checkpoint_path)


def upload_checkpoint(checkpoint_path: Path) -> None:
    """Copy model from local filesystem to cloud"""
    if SERVICE_ACCOUNT_FILE_ENV in os.environ:
        client = storage.Client.from_service_account_json(
            os.environ[SERVICE_ACCOUNT_FILE_ENV]
        )
    else:
        client = storage.Client()

    bucket = client.get_bucket(_MODEL_BUCKET)
    blob = bucket.blob(_MODEL_BLOB)
    blob.upload_from_filename(checkpoint_path)
    _logger.info(f"Uploaded claim type model from {checkpoint_path} to {bucket}/{blob}")


def _is_download_needed() -> bool:
    need_to_download = False
    if _CLAIM_TYPE_MODEL_FORCE_DOWNLOAD in os.environ:
        need_to_download = True
    if _MODEL_FILE_TO_DOWNLOAD_LOC.exists():
        file_info = os.stat(str(_MODEL_FILE_TO_DOWNLOAD_LOC))
        # A file with size zero might as well not exist:
        if file_info.st_size < 1:
            need_to_download = True
    else:
        need_to_download = True
    return need_to_download


def download_checkpoint_if_needed(
    model_blob_path: Optional[str] = None, local_path: Optional[Path] = None
) -> None:
    """Download model from cloud to local filesystem, unless it's already there."""
    if not _is_download_needed():
        return

    if SERVICE_ACCOUNT_FILE_ENV in os.environ:
        client = storage.Client.from_service_account_json(
            os.environ[SERVICE_ACCOUNT_FILE_ENV]
        )
    else:
        client = storage.Client()

    if model_blob_path is None:
        model_blob_path = _MODEL_BLOB
    if local_path is None:
        local_path = _MODEL_FILE_TO_DOWNLOAD_LOC

    model_blob = client.bucket(_MODEL_BUCKET).blob(model_blob_path)

    with open(local_path, mode="wb") as f:
        model_blob.download_to_file(f)
    _logger.info(f"Downloaded claim type model from {model_blob_path} to {local_path}")


def download_bert_pretrained_if_needed(
    bert_pretrained_prefix: Optional[str] = None,
    local_path: Optional[Path] = None,
) -> None:
    if SERVICE_ACCOUNT_FILE_ENV in os.environ:
        client = storage.Client.from_service_account_json(
            os.environ[SERVICE_ACCOUNT_FILE_ENV]
        )
    else:
        client = storage.Client()

    if not bert_pretrained_prefix:
        bert_pretrained_prefix = _BERT_PRETRAINED_PREFIX
    if not local_path:
        local_path = _BERT_TO_DOWNLOAD_LOC

    blob: storage.Blob
    for blob in client.bucket(_MODEL_BUCKET).list_blobs(prefix=_BERT_PRETRAINED_PREFIX):
        *_, filename = str(blob.name).split("/")
        if not (local_path / filename).exists() or _BERT_MODEL_FORCE_DOWNLOAD:
            try:
                os.makedirs(local_path)
            except FileExistsError:
                pass

            blob.download_to_filename(local_path / filename)

    _logger.info(
        f"Downloaded pretrained BERT model and tokenizer from {bert_pretrained_prefix} to {local_path}"
    )


if __name__ == "__main__":
    upload_checkpoint(live_model_path)
