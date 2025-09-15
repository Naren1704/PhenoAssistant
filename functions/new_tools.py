# Example of defining a new tool, which will be automatically added to PhenoAssistant with the @pheno_tool decorator.
from utils.registry import pheno_tool
from typing import Annotated, Literal, List, Dict, Optional, Tuple, Any

# @pheno_tool(
#     name="leaf_counting", # this is the name of the tool that will be registered and recognised by PhenoAssistant; do not name it the same as an existing tool
#     description="Count leaves from an image" # description of the tool, which will instruct PhenoAssistant how to use it
# )
# def count_leaves(
#     img_path: Annotated[str, "Path to the image to process"], # define the input parameter using Annotated
#     ) -> str:
#     # ... your implementation ...
#     return str(6)

# # Comment the following out if you want to use Huggingface Inference API
# import os, requests
# from pathlib import Path
# def infer_hf_model_single(filename, repo_id, img_type='png'):
#     API_URL = "https://router.huggingface.co/hf-inference/models/" + repo_id
#     headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}",}
#     with open(filename, "rb") as f:
#         data = f.read()
#     response = requests.post(API_URL, headers={"Content-Type": f"image/{img_type}", **headers}, data=data)
#     return response.json()

# @pheno_tool(
#     name="infer_huggingface_model_api", # this is the name of the tool that will be registered and recognised by PhenoAssistant; do not name it the same as an existing tool
#     description="Only use this tool when users specifically specifies that they want to it. Infer a huggingface model via API on one or multiple images. The user must supply the list of image paths and the model repo."
# )
# def infer_hf_model(image_urls: Annotated[List[str], "A list of image paths."], 
#                    repo_id: Annotated[str, "Model identifiers on HuggingFace Inference API."],):
#     results = [infer_hf_model_single(f, repo_id, Path(f).suffix.lower().lstrip(".")) for f in image_urls]
#     return results