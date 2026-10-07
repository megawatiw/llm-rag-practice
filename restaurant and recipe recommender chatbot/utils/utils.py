import numpy as np
import torch
import os

from langchain_chroma import Chroma

from dotenv import load_dotenv

# IBM WatsonX imports
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import (
    ModelTypes,
    DecodingMethods,
)

load_dotenv()

def init_llm(model_id):
    project_id = os.getenv("IBM_PROJECT_ID")
    project_url = os.getenv("IBM_PROJECT_URL")
    api_key = os.getenv("IBM_API_KEY")

    if not project_id or not project_url or not api_key:
        raise RuntimeError("Project credentials not found.")

    credentials = Credentials(
        url=project_url,
        api_key=api_key
    )

    ### 1.1: Define the model by ModelInference
    model = ModelInference(
        model_id=model_id,
        project_id=project_id,
        credentials=credentials
    )

    return model
