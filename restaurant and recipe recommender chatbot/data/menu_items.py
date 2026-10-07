import json
import base64
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
from utils.utils import init_llm
from utils.database import DatabaseConnection

# TOD: Add json auto repair for the menu items data, e.g., if the price is negative, set it to 0.0, if the discounted price is greater than the price, set it to the price.


class MenuItem(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    discounted_price: float
    picture_url: Optional[str] = None
    ordered_count: int = 0
    is_available: bool = True


def _load_data():
    with open('./data/Recipes.json', 'r') as file:
        menu_list_data = json.load(file)
    return menu_list_data


def _image_caption_prompt_template(food_name):
    # food_name: the food name of the recipe

    image_caption_system_msg = f"You are a helpful AI assistant who helps generating textual description for {food_name}."
    image_caption_prompt_txt = "Given the food name and image, generate a textual description that explains about the food. \
    You should focus on the ingredients, cooking style, and taste profile. Avoid unnecessary details or speculative information."

    return image_caption_system_msg, image_caption_prompt_txt


def _save_to_db(db_conn, table_name, data):
    structured_menu_items_json = [json.loads(response) for response in data]

    try:
        if not db_conn.table_exists(table_name):
            db_conn.create_table(f"""
                CREATE TABLE {table_name} (
                    menu_id INTEGER PRIMARY KEY,
                    restaurant_id INTEGER NOT NULL REFERENCES restaurants(restaurant_id) ON DELETE CASCADE,
                    name VARCHAR(255),
                    description VARCHAR(255),
                    price INTEGER,
                    discounted_price INTEGER,
                    picture_url VARCHAR(255),
                    ordered_count INTEGER,
                    is_available BOOLEAN,
                )
            """)

            db_conn.conn.commit()
    except Exception as e:
        print(f"Error creating table {table_name}: {e}")
        return

    # For each item in the restaurant list, assign it with an itemId to be consistent with the one in the user review data:
    for i, menu in enumerate(structured_menu_items_json):
        try:
            db_conn.insert_item(f"""
                INSERT INTO {table_name} (menu_id, restaurant_id, name, description, price, discounted_price, picture_url, ordered_count, is_available)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (menu_id) DO UPDATE SET
                title=EXCLUDED.title, description=EXCLUDED.description, price=EXCLUDED.price, discounted_price=EXCLUDED.discounted_price, picture_url=EXCLUDED.picture_url, ordered_count=EXCLUDED.ordered_count, is_available=EXCLUDED.is_available
            """, (
                menu['id'], menu['restaurant_id'], menu['name'], menu['description'],
                menu['price'], menu['discounted_price'], menu['picture_url'],
                menu['ordered_count'], menu['is_available']
            ))
        except Exception as e:
            print(f"Error inserting menu item: {e}")
            continue

    return


def process_menu_item_data():
    menu_list = _load_data()
    vision_llm = init_llm("meta-llama/llama-4-maverick-17b-128e-instruct-fp8")

    for i in range(len(menu_list)):
        if (i + 1) % 20 == 0:
            print(f'{i + 1} out of {len(menu_list)} is done')

        if menu_list[i]['description'] is None:
            if menu_list[i]['picture_url'] is not None:
                ### Step 1: Get the caption prompts
                food_name = menu_list[i]['title']
                image_caption_system_msg, image_caption_prompt_txt = _image_caption_prompt_template(food_name)

                ### Step 2: Encode the input image to a base64 string
                food_id = menu_list[i]['id']
                image_path = f"./data/synthetic_menu_images/menu{str(food_id)}.png"
                with open(image_path, 'rb') as img_file:
                    img_bytes = img_file.read()
                img_base64 = base64.b64encode(img_bytes).decode("utf-8")

                ### Step 3: Define the messages for the model
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

                ### Step 4: Get the response with the prompts
                response = vision_llm.chat(messages)

                ### Save the response as another item in the recipe data
                menu_list[i]['description'] = response['choices'][0]['message']['content']

            else:
                # If no image is available, use a default description or leave it as is
                menu_list[i]['description'] = "No description available."

    print('ALL DONE!')

    db_conn = DatabaseConnection()
    db_conn.connect()
    _save_to_db(db_conn, "menu_items", menu_list)
    db_conn.disconnect()