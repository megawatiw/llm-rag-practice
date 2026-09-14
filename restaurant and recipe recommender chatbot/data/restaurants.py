import shutil
import json
import os
from datetime import datetime

from langchain_core.documents import Document
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional

from utils.database import *
from utils.utils import *


class Restaurant(BaseModel):
    name: str
    location: str
    type: str
    food_style: str
    rating: Optional[float] = None
    price_range: Optional[int] = None
    signatures: List[str] = Field(default_factory=list)
    vibe: Optional[str] = None
    environment: str
    shortcomings: List[str] = Field(default_factory=list)


def _load_data():
    ### 1.1: Define the file_path to the text file
    file_path = "./California-Culinary-Map.txt"

    ### 1.2: Open the text file
    with open(file_path, 'r') as file:
        data = file.read()

    ### 1.3: Print the first 100 characters of the restaurant data
    print(data)

    ### 2.1: Split the restaurant paragraphs into list (hint: use .split)
    restaurant_list = data.split('\n\n')

    ### 2.2: Since the first item is the dataset name, we remove it
    restaurant_list = restaurant_list[1:]

    ### 2.3: Print out the number of restaurants we have
    print(len(restaurant_list))

    ### 2.4: Print out the first item to have a closer look at the content
    print(restaurant_list[0])

    return restaurant_list


def _restaurant_data_structure_prompt_generation(example, restaurant_paragraph):
    EXAMPLE_RESTAURANT_PARAGRAPH =  example  # use the second restaurant paragraph as the example
    EXAMPLE_OUTPUT = """
        {
        "name": "Mar de Cortez",
        "location": "Santa Monica",
        "type": "casual taqueria",
        "food_style": "Baja-style seafood",
        "rating": 4.2,
        "price_range": 1,
        "signatures": [
            "beer-battered snapper tacos",
            "zesty octopus ceviche"
        ],
        "vibe": "salt-air energy",
        "environment": "a premier sun-drenched spot for open-air dining near the pier."
        "shortcomings": []
        }
    """

    base_system_msg = f"""
    You are an AI assistant that helps identifying and extracting information from a paragraph and convert it into structured format.
    """

    base_user_prompt = f"""
    Task:
    Identify main features of a restaurant described in the paragraph. For the price range, convert the '$' symbol into integer representing the number of the '$' symbols.
    Convert the information into a structured key-value pair format. In the final result, do not include restaurant description from the input and only include the structured format.

    Restaurant description:
    {restaurant_paragraph}

    Example:
    Input Restaurant Description: {EXAMPLE_RESTAURANT_PARAGRAPH}
    Output:
    {EXAMPLE_OUTPUT}

    """
    return base_system_msg, base_user_prompt


def _JSON_auto_repair_prompts(candidate_json_output, error_message):
    auto_repair_system_msg = """
    You are a helpful assistant who helps repairing and correcting texts to conform with the required JSON format and structure.
    """
    auto_repair_prompt = f"""
    Task:
    You are given a text, supposed to be in JSON format, but there are still errors on the formatting.
    Given the error message, identify the error type, find the part in text that causes the error, and repair the text so that it conforms with the correct JSON format.
    Do not include inputted original text and error message in the final result. Include only the corrected output formatted in JSON.

    Original text: {candidate_json_output}
    Error message: {error_message}

    Example:
    Input original text:
    {{
        "name":"Iron & Embers",
        "location":"Arts District, DTLA",
        "type":"industrial American steakhouse",
        "vibe":"moody, masculine sophistication",
        "rating":4.8,"price_range":4,
        "signature":"45-day dry-aged ribeye served with bone marrow chimichurri",
        "environment":"smells deeply of white oak smoke"
    }}
    Input error message:
    {{
        Validation failed: [
            {{
                "type":"missing",
                "loc":["food_style"],
                "msg":"Field required",
                "url":"https://errors.pydantic.dev/2.10/v/missing"
            }}
        ]
    }}
    Corrected output:
    {{
        "name":"Iron & Embers",
        "location":"Arts District, DTLA",
        "food_style":"industrial American steakhouse",
        "type": "fine dining",
        "vibe":"moody, masculine sophistication",
        "rating":4.8,
        "price_range":4,
        "signature":"45-day dry-aged ribeye served with bone marrow chimichurri",
        "environment":"smells deeply of white oak smoke"
    }}    
    """
    return auto_repair_system_msg, auto_repair_prompt


def _save_restaurant(db_conn, table_name, data):
    structured_restaurant_lists_json = [json.loads(response) for response in data]

    try:
        if not db_conn.table_exists(table_name):
            db_conn.create_table( f"""
                CREATE TABLE {table_name} (
                    itemId INTEGER PRIMARY KEY,
                    name VARCHAR(255),
                    location VARCHAR(255),
                    type VARCHAR(255),
                    food_style VARCHAR(255),
                    rating DECIMAL(3,2),
                    price_range INTEGER,
                    signatures TEXT[],
                    vibe TEXT,
                    environment TEXT,
                    shortcomings TEXT[]
                )
            """)
            db_conn.conn.commit()
    except Exception as e:
        print(f"Error creating table {table_name}: {e}")
        return False

    # For each item in the restaurant list, assign it with an itemId to be consistent with the one in the user review data:
    for i, restaurant in enumerate(structured_restaurant_lists_json):
        restaurant['itemId'] = 1000001 + i
        try:
            db_conn.insert_item(f"""
                INSERT INTO {table_name} (itemId, name, location, type, food_style, rating, price_range, signatures, vibe, environment, shortcomings)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (itemId) DO UPDATE SET
                name=EXCLUDED.name, location=EXCLUDED.location, type=EXCLUDED.type
            """, (
                restaurant['itemId'], restaurant['name'], restaurant['location'],
                restaurant['type'], restaurant['food_style'], restaurant.get('rating'),
                restaurant.get('price_range'), restaurant.get('signatures', []),
                restaurant.get('vibe'), restaurant['environment'], restaurant.get('shortcomings', [])
            ))
        except Exception as e:
            print(f"Error inserting restaurant: {e}")
            continue

    return None

def _save_restaurant_text_embeddings(db_conn, table_name, restaurant_list):
    article_docs = []

    for i, r in enumerate(restaurant_list):
        name = str(r.get("name", "")).strip()
        if not name:
            continue

        text = (
            f"Restaurant: {name}\n"
            f"Cuisine: {r.get('food_style', '')}\n"
            f"Location: {r.get('location', '')}\n"
            f"Vibe: {r.get('vibe', '')}\n"
            f"Environment: {r.get('environment', '')}\n"
            f"Ratings: {r.get('ratings', '')}\n"
            f"Price Range: {r.get('price_range', '')}\n"
        )

        # GUARANTEED UNIQUE
        doc_id = f"rest_{i}"

        article_docs.append(
            Document(
                page_content=text.strip(),
                metadata={
                    "doc_id": doc_id,
                    "cuisine": r.get("food_style"),
                    "location": r.get("location"),
                    "price_range": r.get("price_range"),
                    "ratings": r.get("ratings"),
                    "source": "restaurant",
                },
            )
        )

    print("✅ article docs:", len(article_docs))

    A = embed_texts([d.page_content for d in article_docs])

    for i in range(len(A)):
        itemId = 1000001 + i
        try:
            db_conn.insert_item(f"""
                INSERT INTO {table_name} (itemId, embedding, text, model_name, created_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (itemId) DO UPDATE SET
                text=EXCLUDED.text, model_name=EXCLUDED.model_name
            """, (
                itemId, A[i][1], article_docs[i].page_content, "meta-llama/llama-3-3-70b-instruct", datetime.now()
            ))
        except Exception as e:
            print(f"Error inserting restaurant: {e}")
            continue

    return None

def process_restaurant_data():
    restaurant_list = _load_data()
    llm = init_llm("meta-llama/llama-3-3-70b-instruct")
    structured_restaurant_lists = []

    for i, restaurant_paragraph in enumerate(restaurant_list):
        ### 2.1: Produce your initial output
        base_system_msg, base_user_prompt = _restaurant_data_structure_prompt_generation(example=restaurant_list[1], restaurant_paragraph=restaurant_paragraph)
        message = [
            {"role": "system", "content": base_system_msg},
            {"role": "user", "content": base_user_prompt}
        ]

        ### 1.3: Get the final response output and return it
        structured_output = llm.chat(message)['choices'][0]['message']['content']

        ### 2.2: Validation and Auto Correction loop on the output (Hint: while loop)
        while True:
            try:
                restaurant_data = Restaurant.model_validate_json(structured_output)
                # print(f"Success! Validated: {restaurant_data.name}")
                break
            except ValidationError as e:
                auto_repair_system_msg, auto_repair_prompt = _JSON_auto_repair_prompts(structured_output, e.json())
                structured_output = llm.chat([
                    {"role": "system", "content": auto_repair_system_msg},
                    {"role": "user", "content": auto_repair_prompt}
                ])['choices'][0]['message']['content']
                # print(f"Validation failed: {e.json()}")

        ### 2.3: Append your finalized response to the structured_restaurant_lists
        structured_restaurant_lists.append(structured_output)

        # A manual progress bar
        if (i+1)%20 == 0:
            print(f'{i+1} out of {len(restaurant_list)} is done')

    # A final message to notify the completion
    print('ALL DONE!!')

    db_conn = DatabaseConnection()
    db_conn.connect()
    _save_restaurant(db_conn, "restaurants", structured_restaurant_lists)
    _save_restaurant_text_embeddings(db_conn, "restaurants_text_embeddings", structured_restaurant_lists)
    db_conn.disconnect()

    # _save_to_chromadb(structured_restaurant_lists)