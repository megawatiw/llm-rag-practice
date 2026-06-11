import requests
import re
import base64
import os
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import Model, ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)

credentials = Credentials(
    url="https://us-south.ml.cloud.ibm.com"
)
client = APIClient(credentials)
model_id = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
project_id = "skills-network"
params = TextChatParameters()

model = ModelInference(
    model_id=model_id,
    credentials=credentials,
    project_id=project_id,
    params=params
)


def input_image_setup(uploaded_file):
    if not uploaded_file:
        raise FileNotFoundError("No file found")
    
    bytes_data = uploaded_file.read()
    encoded_img = base64.b64encode(bytes_data).decode("utf-8")
    
    return encoded_img


def format_response(response_text):
    # Replace section headers that are bolded with '**' to HTML paragraph tags with bold text
    response_text = re.sub(r"\*\*(.*?)\*\*", r"<p><strong>\1</strong></p>", response_text)

    # Convert bullet points denoted by "*" to HTML list items
    response_text = re.sub(r"(?m)^\s*\*\s(.*)", r"<li>\1</li>", response_text)

    # Wrap list items within <ul> tags for proper HTML structure and indentation
    response_text = re.sub(r"(<li>.*?</li>)+", lambda match: f"<ul>{match.group(0)}</ul>", response_text, flags=re.DOTALL)

    # Ensure that all paragraphs have a line break after them for better separation
    response_text = re.sub(r"</p>(?=<p>)", r"</p><br>", response_text)

    # Ensure the disclaimer and other distinct paragraphs have proper line breaks
    response_text = re.sub(r"(\n|\\n)+", r"<br>", response_text)

    return response_text


def generate_model_response(encoded_image, user_query, assistant_prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": f"{assistant_prompt}\n\n{user_query}"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + encoded_image
                    }
                }
            ]
        }
    ]

    try:
        response = model.chat(messages=messages)
        raw_response = response['choices'][0]['message']['content']
        formatted_response = format_response(raw_response)
        return formatted_response
    except Exception as e:
        print(f"Error in generating response: {e}")
        return "<p>An error occurred while generating the response.</p>"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        user_query = request.form.get("user_query")
        uploaded_file = request.files.get("file")

        if uploaded_file:
            encoded_image = input_image_setup(uploaded_file)
            if not encoded_image:
                flash("Error processing the image. Please try again.", "danger")
                return redirect(url_for("index"))
            
            assistant_prompt = """
                You are an expert nutritionist. Your task is to analyze the food items displayed in the image and provide a detailed nutritional assessment using the following format:

                1. **Identification**: List each identified food item clearly, one per line.
                2. **Portion Size & Calorie Estimation**: For each identified food item, specify the portion size and provide an estimated number of calories. Use bullet points with the following structure:
                - **[Food Item]**: [Portion Size], [Number of Calories] calories

                Example:
                *   **Salmon**: 6 ounces, 210 calories
                *   **Asparagus**: 3 spears, 25 calories

                3. **Total Calories**: Provide the total number of calories for all food items.

                Example:
                Total Calories: [Number of Calories]

                4. **Nutrient Breakdown**: Include a breakdown of key nutrients such as **Protein**, **Carbohydrates**, **Fats**, **Vitamins**, and **Minerals**. Use bullet points, and for each nutrient provide details about the contribution of each food item.

                Example:
                *   **Protein**: Salmon (35g), Asparagus (3g), Tomatoes (1g) = [Total Protein]

                5. **Health Evaluation**: Evaluate the healthiness of the meal in one paragraph.

                6. **Disclaimer**: Include the following exact text as a disclaimer:

                The nutritional information and calorie estimates provided are approximate and are based on general food data. 
                Actual values may vary depending on factors such as portion size, specific ingredients, preparation methods, and individual variations. 
                For precise dietary advice or medical guidance, consult a qualified nutritionist or healthcare provider.

                Format your response exactly like the template above to ensure consistency.
            """
            response = generate_model_response(encoded_image, user_query, assistant_prompt)

            return render_template("index.html", user_query=user_query, response=response)

        else:
            flash("Please upload an image file.", "danger")
            return redirect(url_for("index"))
        
    return render_template("index.html")


'''@app.route('/generate', methods=['POST'])
def index():
    pass  '''      

if __name__ == '__main__':
    app.run(debug=True)