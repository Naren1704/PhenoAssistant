# PhenoAssistant

This is the official code repository for our paper [*"PhenoAssistant: A Conversational Multi-Agent AI System for Automated Plant Phenotyping"*](https://arxiv.org/abs/2504.19818).

---

## Chat logs

- **Case studies' chat logs**  
  Chat logs for the case studies presented in our manuscript are available in:  
  - [case1.ipynb](https://github.com/fengchen025/PhenoAssistant/blob/v2/case1.ipynb)  
  - [case2.ipynb](https://github.com/fengchen025/PhenoAssistant/blob/v2/case2.ipynb)  
  - [case3.ipynb](https://github.com/fengchen025/PhenoAssistant/blob/v2/case3.ipynb)  

- **Evaluation's chat logs**  
  Chat logs and results for the evaluations presented in our manuscript are available in:  
  - [eval_tool_selection.ipynb](https://github.com/fengchen025/PhenoAssistant/blob/v2/eval_tool_selection.ipynb)
  - [eval_vision_model_selection.ipynb](https://github.com/fengchen025/PhenoAssistant/blob/v2/eval_vision_model_selection.ipynb)  
  - [eval_data_analysis.ipynb](https://github.com/fengchen025/PhenoAssistant/blob/v2/eval_data_analysis.ipynb)
  - **(New)** [eval_tool_selection_new.ipynb](https://github.com/fengchen025/PhenoAssistant/blob/v2/eval_tool_selection_new.ipynb): using an LLM-based evaluator to evaluate tool selection on 20 tasks
  - **(New)** [eval_tool_selection_critic.ipynb](https://github.com/fengchen025/PhenoAssistant/blob/v2/eval_tool_selection_critic.ipynb): using an LLM-based evaluator to evaluate tool selection with a Critic agent on 20 tasks
  - **(New)** [eval_vision_model_selection_new.ipynb](https://github.com/fengchen025/PhenoAssistant/blob/v2/eval_vision_model_selection_new.ipynb): evaluation on refined vision model selection tasks
  - **(New)** [eval_data_analysis_new.ipynb](https://github.com/fengchen025/PhenoAssistant/blob/v2/eval_data_analysis_new.ipynb): evaluation on 10 more data analysis tasks 

- **(New)** Pipeline reproduction experiments: [case1_pipe.ipynb](https://github.com/fengchen025/PhenoAssistant/blob/v2/case1_pipe.ipynb) 
---

## Key components of PhenoAssistant

- Implementation of agents is available at [agents.py](https://github.com/fengchen025/PhenoAssistant/blob/v2/agents.py)
- Implementation of tools is available at [functions](https://github.com/fengchen025/PhenoAssistant/tree/v2/functions)

---

## Environment setup and demo

To play with a demo, make sure you have **GPU** (to infer deep learning models) and **Azure OpenAI (or OpenAI) API key** available, then follow these steps:

1. Clone the repository:  
   `git clone https://github.com/fengchen025/PhenoAssistant.git`
2. Navigate into the project directory:  
   `cd PhenoAssistant`
3. Create the conda environment (this may take ~15 minutes):  
   `conda env create -f environment.yml`
4. Activate the environment:  
   `conda activate phenoassistant`
5. Install requirements for Leaf-only-sam:
   - `mkdir -p ./models`
   - `pip install git+https://github.com/facebookresearch/segment-anything.git`
   - `wget -O ./models/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth`
6. Set up [.env.yaml](https://github.com/fengchen025/PhenoAssistant/blob/v2/.env.yaml) with your API key. See comments inside the file for guidance.
7. Run the demo at [demo.ipynb](https://github.com/fengchen025/PhenoAssistant/blob/v2/demo.ipynb). Depending on your machine, it may take ~15 minutes to complete. Example outputs are shown in the notebook and saved at [./results/demo](https://github.com/fengchen025/PhenoAssistant/tree/v2/results/demo).

---

Note: All case studies, evaluations, and demo results were generated using GPT-4o (version: 2024-08-06) via Azure OpenAI. Using a different model or provider may lead to different results.

---
## (New) Adding external tools and agents
To add customised tools and agents, please refer to xx.

**(New)** Adding customised tools and agents
**(New)** Connect with HuggingFace Inference API
