import pandas as pd

#source: https://arize.com/docs/ax/cookbooks/agents/tracing-a-routing-agent
API_KEY = "ak-3a5bb238-2bc7-4ba9-8a56-3bc292107f3d-XdqrMnBQnoUw0nybtl8cOsHPYihx8W77"
SPACE_ID = "U3BhY2U6Mjc4NjQ6bUZxRw=="

# Import open-telemetry dependencies
from arize.otel import register

# Setup OTEL via our convenience function
tracer_provider = register(
    space_id=SPACE_ID,
    api_key=API_KEY,
    project_name="agents-tracing-example",  # name this to whatever you would like
)
# Import the automatic instrumentor from OpenInference
from openinference.instrumentation.openai import OpenAIInstrumentor

# Finish automatic instrumentation
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

import nest_asyncio
import pandas as pd
from phoenix.evals import OpenAIModel

nest_asyncio.apply()

GEN_TEMPLATE = """
You are an assistant that generates complex customer service questions.
The questions should often involve:

Multiple Categories: Questions that could logically fall into more than one category (e.g., combining product details with a discount code).
Vague Details: Questions with limited or vague information that require clarification to categorize correctly.
Mixed Intentions: Queries where the customer’s goal or need is unclear or seems to conflict within the question itself.
Indirect Language: Use of indirect or polite phrasing that obscures the direct need or request (e.g., using "I was wondering if..." or "Perhaps you could help me with...").
For specific categories:

Track Package: Include vague timing references (e.g., "recently" or "a while ago") instead of specific dates.
Product Comparison and Product Search: Include generic descriptors without specific product names or IDs (e.g., "high-end smartphones" or "energy-efficient appliances").
Apply Discount Code: Include questions about discounts that might apply to hypothetical or past situations, or without mentioning if they have made a purchase.
Product Details: Ask for comparisons or details that involve multiple products or categories ambiguously (e.g., "Tell me about your range of electronics that are good for home office setups").

Examples of More Challenging Questions
"There's an issue with one of the items I think I bought last month—what should I do?"
"I need help with something I ordered, or maybe I'm just looking for something new. Can you help?"

Some questions should be straightforward uses of the provided functions

Respond with a list, one question per line. Do not include any numbering at the beginning of each line. Do not include any category headings.
Generate 25 questions. Be sure there are no duplicate questions.
"""

functions = [
    {
        "type": "function",
        "name": "product_comparison",
        "description": "Compare features of two products.",
        "parameters": {
            "type": "object",
            "properties": {
                "product_a_id": {
                    "type": "string",
                    "description": "The unique identifier of Product A.",
                },
                "product_b_id": {
                    "type": "string",
                    "description": "The unique identifier of Product B.",
                },
            },
            "required": ["product_a_id", "product_b_id"],
        },
    },
    {
        "type": "function",
        "name": "product_search",
        "description": "Search for products based on criteria.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string.",
                },
                "category": {
                    "type": "string",
                    "description": "The category to filter the search.",
                },
                "min_price": {
                    "type": "number",
                    "description": "The minimum price of the products to search.",
                    "default": 0,
                },
                "max_price": {
                    "type": "number",
                    "description": "The maximum price of the products to search.",
                },
                "page": {
                    "type": "integer",
                    "description": "The page number for pagination.",
                    "default": 1,
                },
                "page_size": {
                    "type": "integer",
                    "description": "The number of results per page.",
                    "default": 20,
                },
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "customer_support",
        "description": "Get contact information for customer support regarding an issue.",
        "parameters": {
            "type": "object",
            "properties": {
                "issue_type": {
                    "type": "string",
                    "description": "The type of issue (e.g., billing, technical support).",
                }
            },
            "required": ["issue_type"],
        },
    },
    {
        "type": "function",
        "name": "track_package",
        "description": "Track the status of a package based on the tracking number.",
        "parameters": {
            "type": "object",
            "properties": {
                "tracking_number": {
                    "type": "integer",
                    "description": "The tracking number of the package.",
                }
            },
            "required": ["tracking_number"],
        },
    },
    {
        "type": "function",
        "name": "product_details",
        "description": "Returns details for a given product id",
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "string",
                    "description": "The id of a product to look up.",
                }
            },
            "required": ["product_id"],
        },
    },
    {
        "type": "function",
        "name": "apply_discount_code",
        "description": "Applies the discount code to a given order.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "integer",
                    "description": "The id of the order to apply the discount code to.",
                },
                "discount_code": {
                    "type": "string",
                    "description": "The discount code to apply",
                },
            },
            "required": ["order_id", "discount_code"],
        },
    },
]

import os

model_id = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
base_url = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-4-mvk-17b-128e-fp8/v1"

model = OpenAIModel(
    model=model_id,
    temperature=0,
    api_key='/',
    base_url=base_url,
    default_headers={'RITS_API_KEY': os.environ["RITS_API_KEY"]},
)

resp = model(GEN_TEMPLATE)

split_response = resp.strip().split("\n")

questions_df = pd.DataFrame(split_response, columns=["questions"])
questions_df["generated_function"] = ""
questions_df["response"] = ""
print(questions_df["questions"])

ROUTER_TEMPLATE = """ You are comparing a response to a question, and verifying whether that response should have made a function call instead of responding directly. Here is the data:
    [BEGIN DATA]
    ************
    [Question]: {question}
    ************
    [Response]: {generated_function}
    [END DATA]

Compare the Question above to the response. You must determine whether the reponse
decided to call the correct function.
Your response must be single word, either "correct" or "incorrect",
and should not contain any text or characters aside from that word.
"incorrect" means that the agent should have made function call instead of responding directly and did not, or the function call chosen was the incorrect one.
"correct" means the selected function would correctly and fully answer the user's question.

Here is more information on each function:
{function_info}

"""

from opentelemetry import trace

tracer = trace.get_tracer(__name__)

from openinference.semconv.trace import (
    SpanAttributes,
    ToolCallAttributes,
    OpenInferenceSpanKindValues,
)


import os
from langchain_openai import ChatOpenAI, OpenAI

model_id = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
base_url = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-4-mvk-17b-128e-fp8/v1"

rits_client = OpenAI(api_key=os.environ.get("RITS_API_KEY"), base_url=base_url)

TASK_MODEL = model_id
#TASK_MODEL = "gpt-4o"
gen_params = {}
gen_params["extra_headers"] = {"RITS_API_KEY": os.environ.get("RITS_API_KEY")}


def agent_router(input):
    # Obtain a tracer instance
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span(
        "AgentOperation",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value
        },
    ) as span:
        response = rits_client.chat.completions.create(
            model=TASK_MODEL,
            temperature=0,
            functions=functions,
            messages=[
                {
                    "role": "system",
                    "content": " ",
                },
                {
                    "role": "user",
                    "content": input["questions"],
                },
            ],
            **gen_params
        )

        if hasattr(response.choices[0].message.function_call, "name"):
            function_call_name = response.choices[0].message.function_call.name
            arguments = response.choices[0].message.function_call.arguments
            # Call handle_function_call if a function call is detected
            generated_response = handle_function_call(
                function_call_name, arguments
            )
        else:
            function_call_name = "no function called"
            arguments = "no function called"
            generated_response = response.choices[0].message.content
        span.set_attribute(SpanAttributes.INPUT_VALUE, input["questions"])
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, generated_response)
        ret = {
            "question": input,
            "function_call_name": function_call_name,
            "arguments": arguments,
            "output": generated_response,
        }
    return ret


def handle_function_call(function_call_name, arguments):
    tracer = trace.get_tracer(__name__)

    # Start a new span for the tool function handling
    with tracer.start_as_current_span(
        "HandleFunctionCall",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL.value,
            ToolCallAttributes.TOOL_CALL_FUNCTION_NAME: function_call_name,
            ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON: str(
                arguments
            ),
            SpanAttributes.INPUT_VALUE: function_call_name,
        },
    ):
        # Here, we simulate the LLM call to generate a response based on function_call_name and arguments
        prompt = f"Function '{function_call_name}' was called with the following arguments: {arguments}. Generate a simulated looking response for this function call. Don't mention it's simulated in your response."

        # Simulate calling the LLM with the constructed prompt
        response = rits_client.chat.completions.create(
            model=TASK_MODEL,
            temperature=0.7,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            **gen_params
        )

        # Extract the generated response from the LLM
        generated_response = response.choices[0].message.content

        return generated_response

def process_questions(df):
    results = []
    for _, row in df.iterrows():
        # Apply the run_prompt function to each question in the dataframe
        result = agent_router({"questions": row["questions"]})
        results.append(result)

    # Convert the results into a DataFrame
    results_df = pd.DataFrame(results)
    return results_df


returned_df = process_questions(questions_df)

pass