import sys
import multiprocessing as mp
from pathlib import Path

import requests
import pandas as pd
from rich.progress import track

prompt_question = "Does the recording description '{}' describe a one-shot sound similar to a snare drum sound?"

ollama_api = "http://localhost:11434/api/generate"
metadata_path = "/home/jp/mgr2024/freesound_meta.csv"
# model = "qwen2.5:1.5b"
model = "qwen2.5:14b"
prompt_template = prompt_question + " Answer only with a literal 'yes' or 'no', no other symbols."


def _classify_description(s: str) -> bool:
    res = requests.post(
        ollama_api,
        json={
            "model": model,
            "prompt": prompt_template.format(s),
            "think": False,
            "stream": False,
        },
    )
    match res.json()["response"]:
        case "yes" | "Yes" | "Yes.":
            result = True
        case "no" | "No" | "No.":
            result = False
        case _:
            print(f"got unexpected response: {res.json()['response']}")
            result = False

    return result


def main(out_name: str) -> None:
    laion_metadata = pd.read_csv(metadata_path)
    laion_metadata["llm_ready_desc"] = laion_metadata["caption1"] + " - " + laion_metadata["caption2"]

    results = []
    all_descriptions = laion_metadata["llm_ready_desc"].values
    with mp.Pool(8) as pool:
        results = list(track(pool.imap(_classify_description, all_descriptions, chunksize=64), total=len(all_descriptions)))

    pd.Series(results, name=prompt_question).to_csv(Path(metadata_path).with_stem(out_name), index=False)


main(sys.argv[1])