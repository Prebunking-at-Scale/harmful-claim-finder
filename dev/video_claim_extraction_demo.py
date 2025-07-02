import json
from pathlib import Path

from harmful_claim_finder.claim_extraction import extract_claims_from_video
from harmful_claim_finder.pastel.inference import CheckworthyClaimDetector

videos = [
    "gs://pas-prototyping-storage/ds-test-videos/7304218971153124651.mp4",
    "gs://pas-prototyping-storage/ds-test-videos/7149378297489558830.mp4",
    "gs://pas-prototyping-storage/ds-test-videos/7172332152292576558.mp4",
    "gs://pas-prototyping-storage/ds-test-videos/7234318005587447083.mp4",
    "gs://pas-prototyping-storage/ds-test-videos/7236655201992576282.mp4",
    "gs://pas-prototyping-storage/ds-test-videos/7282908798186999083.mp4",
    "gs://pas-prototyping-storage/ds-test-videos/7300977537717407022.mp4",
    "gs://pas-prototyping-storage/ds-test-videos/7328225789827190059.mp4",
    "gs://pas-prototyping-storage/ds-test-videos/7331223830645509419.mp4",
    "gs://pas-prototyping-storage/ds-test-videos/7337234721497271598.mp4",
    "gs://pas-prototyping-storage/ds-test-videos/7357367790744931630.mp4",
]


def find_checkworthy_claims():
    output = {}
    for video_uri in videos:
        try:
            claims = extract_claims_from_video(video_uri)
            pastel = CheckworthyClaimDetector(countries=["GBR"])
            claims_text = [claim.claim for claim in claims]
            scores = pastel.score_sentences(claims_text)
            jsonable = [claim.model_dump(mode="json") for claim in claims]
            for claim_dict, score in zip(jsonable, scores):
                claim_dict["score"] = float(score)
            output[video_uri] = jsonable
        except Exception as exc:
            print(f"Something went wrong with {video_uri}: {repr(exc)}")
            continue

    Path("output_claim_extraction_pastel.json").write_text(
        json.dumps(output, indent=4, ensure_ascii=False)
    )


if __name__ == "__main__":
    find_checkworthy_claims()
