# source: https://arize.com/docs/ax/cookbooks/evaluation/trace-level-evaluations-for-a-recommendation-agent

from agents import Agent, Runner, function_tool
from typing import List, Union
from openai import OpenAI
import ast
import os


# model_id = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
# base_url = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-4-mvk-17b-128e-fp8/v1"

model_id = "meta-llama/llama-3-3-70b-instruct"
base_url = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-3-70b-instruct/v1"

gen_params = {}
gen_params["extra_headers"] = {"RITS_API_KEY": os.environ.get("RITS_API_KEY")}

client = OpenAI(api_key=os.environ.get("RITS_API_KEY"), base_url=base_url)

@function_tool
def movie_selector_llm(genre: str) -> List[str]:
    prompt = (
        f"List up to 5 recent popular streaming movies in the {genre} genre. "
        "Provide only movie titles as a Python list of strings."
    )
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=150,
        **gen_params
    )
    content = response.choices[0].message.content
    try:
        movie_list = ast.literal_eval(content)
        if isinstance(movie_list, list):
            return movie_list[:5]
    except Exception:
        return content.split('\n')

@function_tool
def reviewer_llm(movies: Union[str, List[str]]) -> str:
    if isinstance(movies, list):
        movies_str = ", ".join(movies)
        prompt = f"Sort the following movies by rating from highest to lowest and provide a short review for each:\n{movies_str}"
    else:
        prompt = f"Provide a short review and rating for the movie: {movies}"
    response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300,
            **gen_params
    )
    return response.choices[0].message.content.strip()

@function_tool
def preview_summarizer_llm(movie: str) -> str:
    prompt = f"Write a 1-2 sentence summary describing the movie '{movie}'."
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=100,
        **gen_params
    )
    return response.choices[0].message.content.strip()

agent = Agent(
    name="MovieRecommendationAgentLLM",
    tools=[movie_selector_llm, reviewer_llm, preview_summarizer_llm],
    instructions=(
        "You are a helpful movie recommendation assistant with access to three tools:\n"
        "1. MovieSelector: Given a genre, returns up to 5 recent streaming movies.\n"
        "2. Reviewer: Given one or more movie titles, returns reviews and sorts them by rating.\n"
        "3. PreviewSummarizer: Given a movie title, returns a 1-2 sentence summary.\n\n"
        "Your goal is to provide a helpful, user-friendly response combining relevant information."
    ),
)

user_input = "Which comedy movie should I watch?"
result = Runner.run(agent, user_input)
print(result.final_output)

pass
