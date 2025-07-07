import os

from cachetools import cached, TTLCache
import requests
from ff_streaming.file_processing.models import JSON


class OrganisationChecker:
    def __init__(self) -> None:
        self.base_url = os.environ["FULLFACT_API_URL"]
        self.api_token = os.environ["INTERNAL_API_TOKEN"]

    @cached(cache=TTLCache(maxsize=1024, ttl=600))
    def interested_organisations(self, publication: str) -> JSON:
        try:
            with requests.get(
                f"{self.base_url}/api/publications_v2/media_feeds/{publication}/organisations",
                headers={"FULLFACT-INTERNAL-TOKEN": self.api_token},
            ) as resp:
                resp.raise_for_status()
                return resp.json()  # type: ignore

        except requests.HTTPError:
            # So we'll at least get checked by default model
            return {}
