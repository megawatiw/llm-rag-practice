import numpy as np
import os
from typing import List, Tuple, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from utils.utils import retrieve_articles, retrieve_images_by_text
from utils.database import get_chromadb_dir


gemini_api_key = "AIzaSyBk6M77Wp9I9Ch7edPtmgGVe02ZsLqebn0"
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=gemini_api_key, temperature=0.7)


def _minmax(x):
    """Min-max normalize to [0, 1] with safe handling for constant arrays."""
    x = np.array(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if abs(hi - lo) < 1e-8:
        return np.ones_like(x)  # all equal -> treat as same confidence
    return (x - lo) / (hi - lo)

def _classify_intent(user_message: str, llm: ChatGoogleGenerativeAI) -> str:
    """Classify user intent as restaurant, recipe, both, or clarification."""

    system_prompt = """You are an intent classifier for a food recommendation system.

        Analyze the user's message and classify it as ONE of:
        - "restaurant" - User wants restaurant recommendations
        - "recipe" - User wants recipe recommendations
        - "both" - User wants both restaurant and recipe recommendations
        - "clarification" - User needs help or is asking a question
        - "database" - User wants to add/edit/delete database entries

        Examples:
        "Where should I eat tonight?" → restaurant
        "How do I make lasagna?" → recipe
        "I want dinner ideas" → both
        "What can you help me with?" → clarification
        "I want to add a new restaurant" → database

        Respond with ONLY the classification label."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]

    response = llm.invoke(messages)
    intent = response.content[0]['text'].strip().lower()

    # Validate intent
    valid_intents = ["restaurant", "recipe", "both", "clarification", "database"]
    if intent not in valid_intents:
        intent = "clarification"

    return intent

def _format_recommendations(recommendations: List[Dict[str, Any]]) -> str:
    """Format the recommendations into a user-friendly string."""
    if not recommendations:
        return "No recommendations found."

    output_lines = []
    for rec in recommendations:
        line = (
            f"**{rec['modality'].capitalize()}**: {rec['id']}\n"
            f"- Cuisine: {rec.get('cuisine', 'N/A')}\n"
            f"- Location: {rec.get('location', 'N/A')}\n"
            f"- Source: {rec.get('source', 'N/A')}\n"
            f"- Text Score: {rec.get('text_score', 0):.4f}\n"
            f"- Image Score: {rec.get('img_score', 0):.4f}\n"
            f"- Fused Score: {rec.get('fused', 0):.4f}\n"
            f"- Snippet: {rec.get('snippet', '')[:200]}...\n"
        )
        output_lines.append(line)

    return "\n".join(output_lines)

# ================================
# Multimodal fusion
# ================================

def _fuse_rank(
        query: str,
        article_db,
        image_db,
        k_text: int = 5,
        k_img: int = 5,
        w_text: float = 0.6,
        w_img: float = 0.4,
        where_text: dict | None = None,
        where_img: dict | None = None,
        top_n: int = 5
):
    # Retrieve per modality
    t_ids, t_docs, t_metas, t_sims = retrieve_articles(article_db, query, k=k_text, where=where_text)
    i_ids, i_docs, i_metas, i_sims = retrieve_images_by_text(image_db, query, k=k_img, where=where_img)

    # Normalize within modality
    t_norm = _minmax(t_sims)
    i_norm = _minmax(i_sims)

    # Build one mixed candidate list with fused scores
    rows = []
    for j in range(len(t_ids)):
        rows.append({
            "modality": "article",
            "id": t_metas[j].get("doc_id", t_ids[j]) if isinstance(t_metas[j], dict) else t_ids[j],
            "cuisine": t_metas[j].get("cuisine", "N/A") if isinstance(t_metas[j], dict) else "N/A",
            "location": t_metas[j].get("location", "N/A") if isinstance(t_metas[j], dict) else "N/A",
            "source": t_metas[j].get("source", "N/A") if isinstance(t_metas[j], dict) else "N/A",
            "text_score": float(t_norm[j]),
            "img_score": 0.0,
            "fused": float(w_text * t_norm[j]),
            "snippet": (t_docs[j] or "").replace("\n", " ").strip(),
        })

    for j in range(len(i_ids)):
        rows.append({
            "modality": "image",
            "id": i_metas[j].get("doc_id", i_ids[j]) if isinstance(i_metas[j], dict) else i_ids[j],
            "cuisine": i_metas[j].get("cuisine", "N/A") if isinstance(i_metas[j], dict) else "N/A",
            "location": i_metas[j].get("location", "N/A") if isinstance(i_metas[j], dict) else "N/A",
            "source": i_metas[j].get("source", "N/A") if isinstance(i_metas[j], dict) else "N/A",
            "text_score": 0.0,
            "img_score": float(i_norm[j]),
            "fused": float(w_img * i_norm[j]),
            "snippet": (i_docs[j] or "").replace("\n", " ").strip(),
        })

    # Sort by fused score (desc rerank)
    rows.sort(key=lambda r: r["fused"], reverse=True)

    # if top_n not specified, return full pool (k_text + k_img)
    if top_n is None:
        return rows

    top_n = max(0, min(int(top_n), len(rows)))
    return rows[:top_n]


def generate_recommendations(message: str, history: List[Tuple[str, str]]) -> str:
    """Main chatbot function that handles user requests."""

    try:
        # Step 1: Init vector db
        DB_DIR = get_chromadb_dir()
        if not os.path.isdir(DB_DIR):
            raise RuntimeError(
                f"Vector database directory not found: '{DB_DIR}'. "
                "Please run Lesson 1 (Multimodal Vector Index Construction) first."
            )
        article_db = Chroma(collection_name="restaurant_articles", persist_directory=DB_DIR)
        image_db = Chroma(collection_name="food_images", persist_directory=DB_DIR)

        # Step 2: Classify intent
        intent = _classify_intent(message, llm)
        print(f"Classified intent: {intent}")

        # Step 3: Handle different intents
        if intent == "clarification":
            return """I'm your food recommendation assistant! I can help you with:

                🍽️ **Restaurant recommendations** - Tell me your cuisine preferences, dietary restrictions, and occasion
                👨‍🍳 **Recipe recommendations** - Let me know what you'd like to cook
                📝 **Database management** - List or add restaurants and recipes
                
                Just describe what you're looking for, and I'll provide personalized recommendations!"""

        elif intent == "database":
            return """To manage the database, please use the tabs above:

                - **Add Restaurant**: Submit a new restaurant
                - **Add Recipe**: Submit a new recipe
                - **List Restaurants/Recipes**: View existing entries
                
                Is there anything else I can help you with?"""

        elif intent in ["restaurant", "recipe", "both"]:
            '''# Step 3: Extract preferences
            preferences = extract_preferences(message, llm)
            print(f"Extracted preferences: {preferences}")

            # Step 4: Run workflow
            recommendations = run_recommendation_workflow(preferences, intent)

            # Step 5: Format output
            formatted_output = format_recommendations(recommendations)

            return formatted_output'''

            if intent == "restaurant":
                text_w = 0.7
                img_w = 0.3
            elif intent == "recipe":
                text_w = 0.3
                img_w = 0.7
            else:
                text_w = 0.5
                img_w = 0.5

            rows = _fuse_rank(
                message,
                article_db,
                image_db,
                k_text=5,
                k_img=5,
                w_text=text_w,
                w_img=img_w,
                where_text=None,
                where_img=None,
                top_n=5
            )

            formatted_output = _format_recommendations(rows)
            return formatted_output
        else:
            return "I'm not sure how to help with that. Can you rephrase your request?"

    except Exception as e:
        return f"I encountered an error: {str(e)}. Please make sure you have set your Gemini API key."