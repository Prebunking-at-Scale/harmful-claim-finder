from harmful_claim_finder.claim_type_detector.claim_typer import ClaimTyper
from harmful_claim_finder.claim_type_detector import utils

claim_types = [
    "personal",
    "quantity",
    "correlation",
    "rules",
    "predictions",
    "voting",
    "opinion",
    "support",
    "other",
    "not_claim",
]


def create_claim_typer() -> ClaimTyper:
    utils.download_checkpoint_if_needed()
    utils.download_bert_pretrained_if_needed()
    return ClaimTyper(claim_types)


if __name__ == "__main__":
    typer = create_claim_typer()
    print(typer.label_one("Foo"))
