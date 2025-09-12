import os
from openai import OpenAI

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

model_id = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
endpoint_url = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-4-mvk-17b-128e-fp8/v1"
rits_client = OpenAI(api_key=os.environ.get("RITS_API_KEY"), base_url=endpoint_url)

gen_params = {}
gen_params["extra_headers"] = {"RITS_API_KEY": os.environ.get("RITS_API_KEY")}

message = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": f"Please describe this image",
        },
        {
            "type": "image_url",
            "image_url": {"url": "https://huggingface.co/datasets/patrickvonplaten/random_img/resolve/main/yosemite.png"},
        },
    ],
}

completion = rits_client.chat.completions.create(
    messages=[message], 
    model=model_id,
    functions=functions,
    **gen_params
)

question = "I'm looking for a new laptop, can you recommend some high-end options that are good for gaming and video editing?"
response = rits_client.chat.completions.create(
            model=model_id,
            temperature=0,
            functions=functions,
            messages=[
                {
                    "role": "system",
                    "content": " ",
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
            **gen_params
        )

print(response)
pass
