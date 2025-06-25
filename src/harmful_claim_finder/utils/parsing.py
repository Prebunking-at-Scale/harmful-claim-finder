import json_repair
from json_repair.json_parser import JSONReturnType

ParsedType = JSONReturnType | tuple[JSONReturnType, list[dict[str, str]]]


def parse_model_json_output(model_output: str) -> ParsedType:
    parsed = json_repair.loads(model_output)
    if not parsed and parsed != []:
        raise ValueError("Could not parse the string: ", model_output)
    return parsed
