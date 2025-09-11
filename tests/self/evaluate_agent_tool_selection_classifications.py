# source: https://github.com/Arize-ai/phoenix/blob/main/tutorials/evals/evaluate_agent_tool_selection_classifications.ipynb
import nest_asyncio
import json
from tqdm import tqdm
import os
from getpass import getpass
import pandas as pd
import requests
import json
import random
import re

import matplotlib.pyplot as plt
import openai
import pandas as pd
from pycm import ConfusionMatrix

import phoenix.evals.default_templates as templates
from phoenix.evals import (
    AnthropicModel,
    OpenAIModel,
    llm_classify,
)

# Parse tool definitions into a dict: tool_name -> list of required parameters
def extract_tool_param_templates(tool_definitions):
    tools = {}
    pattern = r"^(\w+):.*?\| Parameters: (.*)$"
    for line in tool_definitions.strip().split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            name, param_json = match.groups()
            try:
                param_schema = json.loads(param_json)
                required = param_schema.get("required", [])
                tools[name] = required
            except json.JSONDecodeError:
                continue
    return tools

if __name__ == "__main__":
    nest_asyncio.apply()
    pd.set_option("display.max_colwidth", None)

    # Load dataset
    url = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/BFCL_v3_exec_multiple.json"
    response = requests.get(url)
    dataset = [json.loads(line) for line in response.text.strip().splitlines()]

    # Collect all unique tool definitions (these are all the tools the agent can choose from)
    unique_tools = {}
    for entry in dataset:
        for tool in entry.get("function", []):
            if tool["name"] not in unique_tools:
                unique_tools[tool["name"]] = tool

    tool_definitions_text = "\n".join(
        f"{tool['name']}: {tool['description']} | Parameters: {json.dumps(tool['parameters'])}"
        for tool in unique_tools.values()
    )

    # Prepare data for evaluation
    eval_data = []
    for entry in dataset:
        question = entry["question"][0][0]["content"]
        ground_truths = entry.get("ground_truth", [])
        for gt in ground_truths:
            eval_data.append(
                {"question": question, "tool_call": gt, "tool_definitions": tool_definitions_text}
            )

    df_eval = pd.DataFrame(eval_data)


    tool_param_templates = extract_tool_param_templates(tool_definitions_text)
    tool_names = list(tool_param_templates.keys())

    # Sample 20 wrong examples
    wrong_examples = []
    for i in range(20):
        row = df_eval.iloc[i]
        question = row["question"]
        correct_tool_call_str = row["tool_call"]

        # Get correct tool name (assume it's before the first '(')
        try:
            correct_tool_name = correct_tool_call_str.split("(")[0]
        except Exception as e:
            print(f"Skipping row {i} due to error: {e}")
            continue

        # Get a different tool
        incorrect_tools = [tool for tool in tool_names if tool != correct_tool_name]
        if not incorrect_tools:
            continue

        wrong_tool = random.choice(incorrect_tools)
        required_params = tool_param_templates.get(wrong_tool, [])

        # Build dummy argument string
        dummy_args = []
        for param in required_params:
            dummy_value = random.choice(
                [42, 3.14, '"example"', "[1, 2, 3]", "True"]
            )  # random but plausible
            dummy_args.append(f"{param}={dummy_value}")

        wrong_tool_call = f"{wrong_tool}({', '.join(dummy_args)})"

        wrong_examples.append(
            {
                "question": question,
                "tool_call": wrong_tool_call,
                "tool_definitions": tool_definitions_text,
            }
        )

    # Create and label the correct examples
    df_eval["true_label"] = "correct"

    # Create and label the incorrect examples
    df_wrong = pd.DataFrame(wrong_examples)
    df_wrong["true_label"] = "incorrect"

    # Combine both datasets
    df_combined = pd.concat([df_eval, df_wrong], ignore_index=True)

    # OPTIONAL: Shuffle AFTER labeling
    df_eval_final = df_combined.sample(frac=1).reset_index(drop=True)

    print(templates.TOOL_SELECTION_PROMPT_TEMPLATE)

    model_id = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    base_url = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-4-mvk-17b-128e-fp8/v1"

    model = OpenAIModel(
        model=model_id,
        temperature=0,
        api_key='/',
        base_url=base_url,
        default_headers={'RITS_API_KEY': os.environ["RITS_API_KEY"]},
    )

    models = [model_id]
    all_results = []
    for model_name in tqdm(models):
        print(f"\n🧪 Evaluating model: {model_name}")
        results = llm_classify(
            data=df_eval,
            template=templates.TOOL_SELECTION_PROMPT_TEMPLATE,
            model=model,
            rails=["correct", "incorrect"],
            provide_explanation=False,
        )

        df_result = df_eval.copy()
        df_result["label"] = results["label"]
        df_result["model"] = model_name
        all_results.append(df_result)
        break

    pass