"""
utils/model_selector.py
=======================
Contribution 2 - Model Selector Agent
"""

import json
import os
from pathlib import Path

MODEL_ZOO_PATH = Path("./model_zoo.json")

def _build_lookup(zoo_path: Path) -> dict:
    zoo = json.loads(zoo_path.read_text())
    lookup = {}
    for category, models in zoo.items():
        if not isinstance(models, list):
            continue
        for m in models:
            if not isinstance(m, dict):
                continue
            key = (m["crop"], m["task"], m["modality"])
            lookup[key] = m["id"]
    return lookup

LOOKUP = _build_lookup(MODEL_ZOO_PATH)

def _ask(client, model_name: str, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0,
        max_tokens=10,
    )
    return resp.choices[0].message.content.strip().lower().split()[0].rstrip(".,")

def select_model(user_query: str, client=None, model_name: str = None):
    from openai import AzureOpenAI

    if client is None:
        client = AzureOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_API_URL"],
            api_version=os.environ["AZURE_API_VERSION"],
        )
    if model_name is None:
        model_name = os.environ["MODEL_NAME"]

    system = (
        "You are a plant phenotyping assistant. "
        "Answer with EXACTLY ONE word from the options given. "
        "No explanation. No punctuation. Just the word."
    )

    crop = _ask(
        client, model_name, system,
        f"Task: '{user_query}'\n\n"
        "What crop species is this task about?\n"
        "Options: rice, wheat, maize, banana, coffee, arabidopsis, potato, multi_crop\n"
        "Choose the single best match."
    )
    print(f"[ModelSelector] Q1 crop     -> {crop}")

    task = _ask(
        client, model_name, system,
        f"Task: '{user_query}'\n\n"
        "What is the primary computer vision task?\n"
        "Options: nutrient_deficiency, detection_counting, segmentation, temporal_phenotyping\n"
        "Choose the single best match."
    )
    print(f"[ModelSelector] Q2 task     -> {task}")

    modality = _ask(
        client, model_name, system,
        f"Task: '{user_query}'\n\n"
        "What imaging modality or sensor is used?\n"
        "Options: close_range_rgb, uav_aerial, satellite\n"
        "Choose the single best match."
    )
    print(f"[ModelSelector] Q3 modality -> {modality}")

    key = (crop, task, modality)
    model_id = LOOKUP.get(key)

    if model_id:
        print(f"[ModelSelector] Match: {key} -> {model_id}")
    else:
        print(f"[ModelSelector] No match for {key}")
        print(f"[ModelSelector] Available keys: {list(LOOKUP.keys())}")

    return model_id
