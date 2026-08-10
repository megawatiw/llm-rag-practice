import numpy as np
import torch
from PIL import Image
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

# IBM WatsonX imports
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import (
    ModelTypes,
    DecodingMethods,
)

# ---- Text embedding model (384-d) ----
text_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---- Image embedding model (512-d) ----
device = "cpu"
clip_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_name, use_fast=True)
clip_model.eval()


def init_llm(model_id):
    project_id = "5177c3e8-f1b6-4221-bfb0-f6ea5ce551cb"
    api_key = "bLh7dxSZkViRM9dGoOGEsEocTf8ke8tikR3uD15CSr2s"

    credentials = Credentials(
        url="https://jp-tok.ml.cloud.ibm.com",
        api_key=api_key
    )

    ### 1.1: Define the model by ModelInference
    model = ModelInference(
        model_id=model_id,
        project_id=project_id,
        credentials=credentials
    )

    return model

def embed_texts(texts, batch_size=64):
    return text_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,  # cosine-ready
    ).astype(np.float32)


@torch.no_grad()
def embed_images(paths, batch_size=16):
    vecs = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i+batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch]
        inputs = clip_processor(images=imgs, return_tensors="pt").to(device)
        feats = clip_model.get_image_features(**inputs)          # (B,512)
        feats = feats / feats.norm(dim=-1, keepdim=True)         # cosine-ready
        vecs.append(feats.cpu().numpy().astype(np.float32))
    return np.vstack(vecs)

