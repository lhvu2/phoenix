import os
from phoenix.evals import (
    RAG_RELEVANCY_PROMPT_TEMPLATE,
    RAG_RELEVANCY_PROMPT_RAILS_MAP,
    OpenAIModel,
    llm_classify,
)

from langchain_openai import ChatOpenAI, OpenAI
# chat_llm = ChatOpenAI(
#     model="ibm-granite/granite-3.1-8b-instruct",
#     temperature=0,
#     max_retries=2,
#     api_key='/',
#     base_url='https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-1-8b-instruct/v1',
#     default_headers={'RITS_API_KEY': os.environ["RITS_API_KEY"]},
# )

# model = OpenAI(
#     model="ibm-granite/granite-3.1-8b-instruct",
#     temperature=0,
#     max_retries=2,
#     api_key='/',
#     base_url='https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-1-8b-instruct/v1',
#     default_headers={'RITS_API_KEY': os.environ["RITS_API_KEY"]},
# )

# # Set your API key
# os.environ["OPENAI_API_KEY"] = "your-openai-key"

# # Create a model
#model = OpenAIModel(model="gpt-4", temperature=0.0)
model = OpenAIModel(
    model="ibm-granite/granite-3.1-8b-instruct",
    temperature=0,
    api_key='/',
    base_url='https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-1-8b-instruct/v1',
    default_headers={'RITS_API_KEY': os.environ["RITS_API_KEY"]},
)

# Prepare your dataset
import pandas as pd

df = pd.DataFrame([
    {
        "reference": "The Eiffel Tower is located in Paris, France. It was constructed in 1889 as the entrance arch to the 1889 World's Fair.",
        "query": "Where is the Eiffel Tower located?",
        "response": "The Eiffel Tower is located in Paris, France.",
    },
    {
        "reference": "The Great Wall of China is over 13,000 miles long. It was built over many centuries by various Chinese dynasties to protect against nomadic invasions.",
        "query": "How long is the Great Wall of China?",
        "response": "The Great Wall of China is approximately 13,171 miles (21,196 kilometers) long.",
    },
    {
        "reference": "The Amazon rainforest is the largest tropical rainforest in the world. It covers much of northwestern Brazil and extends into Colombia, Peru and other South American countries.",
        "query": "What is the largest tropical rainforest?",
        "response": "The Amazon rainforest is the largest tropical rainforest in the world. It is home to the largest number of plant and animal species in the world.",
    },
])

# Evaluate your data
rails = list(RAG_RELEVANCY_PROMPT_RAILS_MAP.values())
results = llm_classify(df, model, RAG_RELEVANCY_PROMPT_TEMPLATE, rails)
pass
