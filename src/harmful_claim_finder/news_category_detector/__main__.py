import logging
import os

import click

_logger = logging.getLogger(__name__)
_logger.setLevel(os.getenv("LOG_LEVEL", logging.INFO))


@click.group()
def base_command() -> None:
    pass  # empty for now


@base_command.command()
@click.option("--project-name", required=True, type=str)
@click.option(
    "--input-queue-name", required=True, type=str, help="Logical input data queue name"
)
@click.option(
    "--output-queue-name",
    required=True,
    type=str,
    help="Logical output data queue name",
)
@click.option(
    "--output-prefix",
    required=True,
    type=str,
    help="A prefix to put before the generated path for output files.",
)
def pipeline(
    project_name: str,
    input_queue_name: str,
    output_queue_name: str,
    output_prefix: str,
) -> None:
    from article_enrichment_processor import processing

    processing.run_article_enrichment_processor(
        project_name, input_queue_name, output_queue_name, output_prefix
    )


if __name__ == "__main__":
    base_command()
