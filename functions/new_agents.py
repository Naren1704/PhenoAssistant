# Example of defining a new agent, which will be automatically added to PhenoAssistant with the @pheno_tool decorator.
from utils.registry import pheno_tool
from typing import Annotated, Literal, List, Dict, Optional, Tuple, Any
import autogen
from agents import gpt_config, user_proxy

# # define a new agent
# SYSTEM_PROMPT = '''
# You are a great assistant.
# '''
# plant_assistant = autogen.AssistantAgent(
#     name="plant_assistant",
#     system_message=SYSTEM_PROMPT,
#     llm_config=gpt_config,
# )

# # wrap the agent into a (or multiple) pheno_tool to be called by the manager agent 
# @pheno_tool(
#     name="plant_assistance", # this is the name of the tool that will be registered and recognised by PhenoAssistant; do not name it the same as an existing tool
#     description="Answer any question about plants" # description of the tool, which will instruct PhenoAssistant how to use it
# )
# def plant_assistance(
#     message: Annotated[str, "Describe the task to be solved."], # define the input parameter using Annotated (typically you need a 'message' to allow the manager agent pass a task here)
#     ) -> str:
#     # ... your implementation ...
#     res = user_proxy.initiate_chat(
#         plant_assistant,
#         clear_history=True,
#         silent=False,
#         message=message,
#         summary_method="reflection_with_llm", # or "last_msg"
#         # summary_args={"summary_prompt":""},
#         max_turns=1,
#     )
#     return res.summary