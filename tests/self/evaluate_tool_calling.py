# source: https://github.com/Arize-ai/phoenix/blob/b107d9bc848efd38f030a8c72954e89616c43723/tutorials/evals/evaluate_tool_calling.ipynb

import os

import pandas as pd
from langchain.agents import AgentType, initialize_agent
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_openai import ChatOpenAI, OpenAI
from openinference.instrumentation.langchain import LangChainInstrumentor

import phoenix as px
import phoenix.evals.default_templates as templates
from phoenix.evals import (
    # TOOL_CALLING_PROMPT_RAILS_MAP,
    # TOOL_CALLING_PROMPT_TEMPLATE,
    OpenAIModel,
    llm_classify,
)

# from phoenix.trace.langchain import LangChainInstrumentor
from phoenix.trace.dsl import SpanQuery


## function definitions using pydantic decorator
@tool
def product_comparison(product_a_id: str, product_b_id: str) -> dict:
    """
    Compare features of two products.

    Parameters:
    product_a_id (str): The unique identifier of Product A.
    product_b_id (str): The unique identifier of Product B.

    Returns:
    dict: A dictionary containing the comparison of the two products.
    """

    if product_a_id == "" or product_b_id == "":
        return {"error": "missing product id"}

    # Implement the function logic here
    return {"comparison": "Similar"}


@tool
def product_details(product_id: str) -> dict:
    """
    Get detailed features on one product.

    Parameters:
    product_id (str): The unique identifier of the Product.

    Returns:
    dict: A dictionary containing product details.
    """

    if product_id == "":
        return {"error": "missing product id"}

    # Implement the function logic here
    return {"name": "Product Name", "price": "$12.50", "Availability": "In Stock"}


@tool
def apply_discount_code(order_id: int, discount_code: str) -> dict:
    """
    Applies a discount code to an order.

    Parameters:
    order_id (str): The unique identifier of the order.
    discount_code (str): The discount code to apply.

    Returns:
    dict: A dictionary containing the updated order details.
    """

    if order_id == "" or discount_code == "":
        return {"error": "missing order id or discount code"}

    # Implement the function logic here
    return {"applied": "True"}


@tool
def product_search(
    query: str,
    category: str = None,
    min_price: float = 0.0,
    max_price: float = None,
    page: int = 1,
    page_size: int = 20,
) -> dict:
    """
    Search for products based on criteria.

    Parameters:
    query (str): The search query string.
    category (str, optional): The category to filter the search. Default is None.
    min_price (float, optional): The minimum price of the products to search. Default is 0.
    max_price (float, optional): The maximum price of the products to search. Default is None.
    page (int, optional): The page number for pagination. Default is 1.
    page_size (int, optional): The number of results per page. Default is 20.

    Returns:
    dict: A dictionary containing the search results and pagination info.
    """

    if query == "":
        return {"error": "missing query"}

    # Implement the function logic here
    return {"results": [], "pagination": {"total": 0, "page": 1, "page_size": 20}}


@tool
def customer_support(issue_type: str) -> dict:
    """
    Get contact information for customer support regarding an issue.

    Parameters:
    issue_type (str): The type of issue (e.g., billing, technical support).

    Returns:
    dict: A dictionary containing the contact information for customer support.
    """

    if issue_type == "":
        return {"error": "missing issue type"}

    # Implement the function logic here
    return {"contact": issue_type}


@tool
def track_package(tracking_number: int) -> dict:
    """
    Track the status of a package based on the tracking number.

    Parameters:
    tracking_number (str): The tracking number of the package.

    Returns:
    dict: A dictionary containing the tracking status of the package.
    """
    if tracking_number == "":
        return {"error": "missing tracking number"}

    # Implement the function logic here
    return {"status": "Delivered"}


GEN_TEMPLATE = """
You are an assistant that generates complex customer service questions. You will try to answer the question with the tool if possible,
do your best to answer, ask for more information only if needed.
The questions should often involve:

Please reference the product names, the product details, product IDS and product information.

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
Multiple Categories

"I recently bought a samsung 106i smart phone, and I was wondering if there's a way to check what deals I might have missed or if my order is on its way?"
"Could you tell me if the samsung 15H adapater in my last order are covered under warranty and if they have shipped yet?"
Vague Details

"There's an issue with one of the Vizio 14Y TV I think I bought last month—what should I do?"
"I need help with a iPhone 16H I ordered, or maybe I'm just looking for something new. Can you help?"
Mixed Intentions

"I'm not sure if I should ask for a refund or just find out when it will arrive. What do you suggest?"
"Could you help me decide whether to upgrade my product or just track the current one?"
Indirect Language

"I was wondering if you might assist me in figuring out a problem I have with an order, or maybe it's more of a query?"
"Perhaps you could help me understand the benefits of your premium products compared to the regular ones?"

Some questions should be straightforward uses of the provided functions

Respond with a list, one question per line. Do not include any numbering at the beginning of each line. Do not include any category headings.
Generate 20 questions.
"""

# @title JSON Function / Tool
json_tools = """
tools = [
    {
        "name": "product_comparison",
        "description": "Compare features of two products.",
        "parameters": {
            "type": "object",
            "properties": {
                "product_a_id": {
                    "type": "string",
                    "description": "The unique identifier of Product A."
                },
                "product_b_id": {
                    "type": "string",
                    "description": "The unique identifier of Product B."
                }
            },
            "required": ["product_a_id", "product_b_id"]
        }
    },
    {
        "name": "product_search",
        "description": "Search for products based on criteria.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string."
                },
                "category": {
                    "type": "string",
                    "description": "The category to filter the search.",
                    "default": None
                },
                "min_price": {
                    "type": "number",
                    "description": "The minimum price of the products to search.",
                    "default": 0
                },
                "max_price": {
                    "type": "number",
                    "description": "The maximum price of the products to search.",
                    "default": None
                },
                "page": {
                    "type": "integer",
                    "description": "The page number for pagination.",
                    "default": 1
                },
                "page_size": {
                    "type": "integer",
                    "description": "The number of results per page.",
                    "default": 20
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "customer_support",
        "description": "Get contact information for customer support regarding an issue.",
        "parameters": {
            "type": "object",
            "properties": {
                "issue_type": {
                    "type": "string",
                    "description": "The type of issue (e.g., billing, technical support)."
                }
            },
            "required": ["issue_type"]
        }
    },
    {
        "name": "track_package",
        "description": "Track the status of a package based on the tracking number.",
        "parameters": {
            "type": "object",
            "properties": {
                "tracking_number": {
                    "type": "integer",
                    "description": "The tracking number of the package."
                }
            },
            "required": ["tracking_number"]
        }
    },
    {
        "name": "product_details",
        "description": "Returns details for a given product id",
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "string",
                    "description": "The id of a product to look up."
                }
            },
            "required": ["product_id"]
        }
    },
    {
        "name": "apply_discount_code",
        "description": "Applies the discount code to a given order.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "integer",
                    "description": "The id of the order to apply the discount code to."
                },
                "discount_code": {
                    "type": "string",
                    "description": "The discount code to apply"
                }
            },
            "required": ["order_id, discount_code"]
        }
    }
]

"""

if __name__ == "__main__":
    session = px.launch_app()
    LangChainInstrumentor().instrument()

    tools = [
        product_comparison,
        product_search,
        customer_support,
        track_package,
        apply_discount_code,
        product_details,
    ]

    # model_id="ibm-granite/granite-3.1-8b-instruct"
    # base_url='https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-1-8b-instruct/v1'

    model_id = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    base_url = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-4-mvk-17b-128e-fp8/v1"

    model = OpenAI(
        model=model_id,
        temperature=0,
        max_retries=2,
        api_key="/",
        base_url=base_url,
        default_headers={"RITS_API_KEY": os.environ["RITS_API_KEY"]},
    )

    resp = model(GEN_TEMPLATE)
    split_response = resp.strip().split("\n")
    questions_df = pd.DataFrame(split_response, columns=["questions"])

    llm = ChatOpenAI(
        model=model_id,
        temperature=0,
        max_retries=2,
        api_key="/",
        base_url=base_url,
        default_headers={"RITS_API_KEY": os.environ["RITS_API_KEY"]},
    )

    # llm = ChatOpenAI(model="gpt-4o")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent_executor = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS)
    questions_df["response"] = questions_df["questions"].apply(agent_executor.invoke)

    query = (
        SpanQuery()
        .where(
            # Filter for the `LLM` span kind.
            # The filter condition is a string of valid Python boolean expression.
            "span_kind == 'LLM'",
        )
        .select(
            # Extract and rename the following span attributes
            question="llm.input_messages",
            response="llm.output_messages",
            tool_call="llm.function_call",
        )
    )
    pc = px.Client()
    trace_df = pc.query_spans(query)
    trace_df = questions_df
    trace_df["tool_call"] = trace_df["tool_call"].fillna("No tool used")

    eval_model = OpenAIModel(
        model=model_id,
        temperature=0,
        api_key="/",
        base_url=base_url,
        default_headers={"RITS_API_KEY": os.environ["RITS_API_KEY"]},
    )

    rails = list(templates.TOOL_CALLING_PROMPT_RAILS_MAP.values())

    response_classifications = llm_classify(
        dataframe=trace_df,
        template=templates.TOOL_CALLING_PROMPT_TEMPLATE.template.replace(
            "{tool_definitions}", json_tools
        ),
        model=eval_model,
        rails=rails,
        provide_explanation=True,
    )
    pass
