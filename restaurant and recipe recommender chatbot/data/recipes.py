import shutil
import glob
import json
import os
import base64

from langchain_core.documents import Document
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional

from utils.database import *
from utils.utils import *


def __load_data():
    with open('./data/Recipes.json', 'r') as file:
        recipe_data = json.load(file)
    return recipe_data


def __image_caption_prompt_template(food_name):
    # food_name: the food name of the recipe

    ### Step 3.1: Design the prompts
    image_caption_system_msg = "You are a helpful AI assistant who helps generating textual description of a food."
    image_caption_prompt_txt = "Given the food name, generate a textual description that explains about the food. \
    You should focus on the ingredients, cooking style, and presentation. Avoid unnecessary details or speculative information."

    return image_caption_system_msg, image_caption_prompt_txt


def __save_to_db(db_conn, table_name, data):
    structured_recipes_lists_json = [json.loads(response) for response in data]

    try:
        if not db_conn.table_exists(table_name):
            db_conn.create_table(f"""
                CREATE TABLE {table_name} (
                    id INTEGER PRIMARY KEY,
                    name VARCHAR(255),
                    cuisine VARCHAR(255),
                    servings INTEGER,
                    prep_time VARCHAR(255),
                    cook_time VARCHAR(255),
                    total_time VARCHAR(255),
                    ingredients TEXT[],
                    instructions TEXT[],
                    image_description TEXT
                )
            """)
            db_conn.conn.commit()
    except Exception as e:
        print(f"Error creating table {table_name}: {e}")
        return

    # For each item in the restaurant list, assign it with an itemId to be consistent with the one in the user review data:
    for i, recipe in enumerate(structured_recipes_lists_json):
        try:
            db_conn.insert_item(f"""
                INSERT INTO {table_name} (id, name, cuisine, servings, prep_time, cook_time, total_time, ingredients, instructions, image_description)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                name=EXCLUDED.name, cuisine=EXCLUDED.cuisine, servings=EXCLUDED.servings, prep_time=EXCLUDED.prep_time, cook_time=EXCLUDED.cook_time, total_time=EXCLUDED.total_time, ingredients=EXCLUDED.ingredients, instructions=EXCLUDED.instructions, image_description=EXCLUDED.image_description
            """, (
                recipe['id'], recipe['name'], recipe['cuisine'], recipe['servings'],
                recipe['prep_time'], recipe['cook_time'], recipe['total_time'],
                recipe['ingredients'], recipe['instructions'], recipe['image_description']
            ))
        except Exception as e:
            print(f"Error inserting recipe: {e}")
            continue

    return

def __save_to_chromadb(recipe_list):
    db_dir = get_chromadb_dir()

    if os.path.isdir(db_dir):
        shutil.rmtree(db_dir)  # Reset vector DB (important for reruns)

    IMG_DIR = "data"
    image_paths = sorted(glob.glob(f"{IMG_DIR}/**/*.png", recursive=True))

    image_docs = []

    for i, (p, rec) in enumerate(zip(image_paths, recipe_list)):
        doc_id = f"img_{i}"

        image_docs.append(
            Document(
                # keeps retrieval results readable
                page_content=rec.get("name", f"recipe image {i}"),
                metadata={
                    "doc_id": doc_id,
                    "image_path": p,
                    "source": "recipe_image",
                    "recipe_id": rec.get("id"),
                    "cuisine": rec.get("cuisine"),
                },
            )
        )

    print("✅ image docs:", len(image_docs))

    V = embed_images([d.metadata["image_path"] for d in image_docs])

    image_db = Chroma(
        collection_name="food_images",
        persist_directory=db_dir,
    )

    image_db._collection.upsert(
        ids=[d.metadata["doc_id"] for d in image_docs],
        embeddings=V.tolist(),
        documents=[d.page_content for d in image_docs],
        metadatas=[d.metadata for d in image_docs],
    )

    print("✅ Image DB ready")


def process_recipes_data():
    recipe_list = __load_data()
    vision_llm = init_llm("meta-llama/llama-4-maverick-17b-128e-instruct-fp8")

    for i in range(len(recipe_list)):
        if (i + 1) % 20 == 0:
            print(f'{i + 1} out of {len(recipe_list)} is done')

        ### Step 4.1: Get the caption prompts
        food_name = recipe_list[i]['name']
        image_caption_system_msg, image_caption_prompt_txt = __image_caption_prompt_template(food_name)

        ### Step 4.2: Encode the input image to a base64 string
        food_id = recipe_list[i]['id']
        image_path = f"./data/synthetic_recipe_images/recipe{str(food_id)}.png"
        with open(image_path, 'rb') as img_file:
            img_bytes = img_file.read()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        ### Step 4.3: Define the messages for the model
        messages = [
            {"role": "system", "content": image_caption_system_msg},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": image_caption_prompt_txt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                    }
                ]
            }
        ]

        ### Step 4.4: Get the response with the prompts
        response = vision_llm.chat(messages)

        ### Save the response as another item in the recipe data
        recipe_list[i]['image_description'] = response['choices'][0]['message']['content']

    print('ALL DONE!')

    db_conn = DatabaseConnection()
    db_conn.connect()
    __save_to_db(db_conn, "recipes", recipe_list)
    db_conn.disconnect()

    __save_to_chromadb(recipe_list)