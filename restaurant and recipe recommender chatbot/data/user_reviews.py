import numpy as np
import matplotlib.pyplot as plt
import json
import ast
import base64
import requests

from tenacity import retry, stop_after_attempt, wait_exponential
from typing import List, Optional

from utils.database import DatabaseConnection
from utils.utils import *


### URL Request function with Retry
# Retries up to 10 times, starting at 1s and doubling (1s, 2s, 4s...)
@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=1, max=10))
def __get_data_with_retry(url):
    response = requests.get(url, timeout=5)
    response.raise_for_status() # Must raise error for retry to trigger
    return response

def __load_data():
    with open('./data/Synthetic_User_Reviews.json', 'r') as file:
        user_review_data = json.load(file)
    return user_review_data

def __review_context_image_caption_prompt_template(reviews):
    # reviews: the written review content

    ### Step 2.1: Design your prompts
    review_context_image_caption_system_msg = "You are a culinary expert who write descriptions about restaurants and its food based on user reviews"
    review_context_image_caption_prompt_txt = "Generate a concise description about a restaurant given its pictures that aligns with the sentiment and details expressed in the user reviews. \
       Take the user reviews as context and add it to the description. Focus on the details of what the writer think about the restaurant and its food. \
       Avoid unnecessary details or speculative information."

    return review_context_image_caption_system_msg, review_context_image_caption_prompt_txt


def __save_to_db(db_conn, table_name, data):
    structured_reviews_lists_json = [json.loads(response) for response in data]

    try:
        if not db_conn.table_exists(table_name):
            db_conn.create_table(f"""
                CREATE TABLE {table_name} (
                    reviewId INTEGER PRIMARY KEY,
                    userId VARCHAR(255),
                    itemId INTEGER FOREIGNKEY,
                    title VARCHAR(255),
                    text VARCHAR(255),
                    date DATETIME,
                    rating DECIMAL(3,2),
                    language VARCHAR(255),
                    images TEXT[],
                    image_captions TEXT
                )
            """)
            db_conn.conn.commit()
    except Exception as e:
        print(f"Error creating table {table_name}: {e}")
        return

    # For each item in the restaurant list, assign it with an itemId to be consistent with the one in the user review data:
    for i, review in enumerate(structured_reviews_lists_json):
        try:
            db_conn.insert_item(f"""
                INSERT INTO {table_name} (reviewId, userId, itemId, title, text, date, rating, language, images, image_captions)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (reviewId) DO UPDATE SET
                userId=EXCLUDED.userId, itemId=EXCLUDED.itemId, title=EXCLUDED.title, text=EXCLUDED.text, date=EXCLUDED.date, rating=EXCLUDED.rating, language=EXCLUDED.language, images=EXCLUDED.images, image_captions=EXCLUDED.image_captions
            """, (
                review['reviewId'], review['userId'], review['itemId'], review['title'],
                review['text'], review['date'], review['rating'], review['language'],
                review['images'], review['image_captions']
            ))
        except Exception as e:
            print(f"Error inserting review: {e}")
            continue

    return


def process_user_reviews_data():
    user_review_data = __load_data()
    vision_llm = init_llm("meta-llama/llama-4-maverick-17b-128e-instruct-fp8")

    for i in range(len(user_review_data)):
        ### Step 3.1: Convert the string to the Python list of image urls
        review_images = ast.literal_eval(user_review_data[i]['images'])

        review_image_captions = []
        if len(review_images) > 0:
            for img_url in review_images:
                try:
                    ### Step 3.2: Use get_data_with_retry to get the image_data
                    image_data = __get_data_with_retry(img_url)
                    print("Success!")
                except Exception as e:
                    print(f"All retries failed at url {img_url}:", e)
                    continue

                ### Step 3.3: Encode the input image to a base64 string
                image = image_data.content
                with open('review_image_placeholder.jpg', 'wb') as img_file:
                    img_file.write(image)

                with open('review_image_placeholder.jpg', 'rb') as img_file:
                    img_bytes = img_file.read()
                img_base64 = base64.b64encode(img_bytes).decode("utf-8")

                ### Step 3.4: Get the prompts, get the response, and finally append the response to review_image_captions
                review_context_image_caption_system_msg, review_context_image_caption_prompt_txt = __review_context_image_caption_prompt_template(
                    user_review_data[i]['text'])

                ### Step 3.5: Define the messages for the model
                messages = [
                    {"role": "system", "content": review_context_image_caption_system_msg},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": review_context_image_caption_prompt_txt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                            }
                        ]
                    }
                ]

                response = vision_llm.chat(messages)
                review_image_captions.append(response['choices'][0]['message']['content'])

        ### Append the review_image_captions to the review data
        user_review_data[i]['image_captions'] = review_image_captions

    print('ALL DONE!')

    db_conn = DatabaseConnection()
    db_conn.connect()
    __save_to_db(db_conn, "user_reviews", user_review_data)
    db_conn.disconnect()


