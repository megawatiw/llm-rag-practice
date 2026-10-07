import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

class EmbeddingService:
    def __init__(self):
        self.text_embedding_model = None
        self.clip_model = None
        self.clip_processor = None

    def _unwrap(self, res: dict):
        """Chroma returns lists-of-lists; unwrap the first query."""
        ids   = res.get("ids", [[]])[0]
        docs  = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        return ids, docs, metas, dists

    def _to_similarity(self, dists):
        """Convert 'smaller is better' distance to 'larger is better' similarity."""
        d = np.array(dists, dtype=np.float32)
        return 1.0 - d

    def init_text_embedding_model(self, model_name):
        self.text_embedding_model = SentenceTransformer(model_name)

    def init_image_embedding_model(self, model_name):
        device = "cpu"
        self.clip_model = CLIPModel.from_pretrained(model_name).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
        self.clip_model.eval()

    def embed_texts(self, texts, batch_size=64):
        return self.text_embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # cosine-ready
        ).astype(np.float32)

    @torch.no_grad()
    def embed_images(self, paths, batch_size=16):
        vecs = []
        for i in range(0, len(paths), batch_size):
            batch = paths[i:i + batch_size]
            imgs = [Image.open(p).convert("RGB") for p in batch]
            inputs = self.clip_processor(images=imgs, return_tensors="pt").to("cpu")
            feats = self.clip_model.get_image_features(**inputs)  # (B,512)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # cosine-ready
            vecs.append(feats.cpu().numpy().astype(np.float32))
        return np.vstack(vecs)

    @torch.no_grad()
    def embed_query_clip_text(self, query: str):
        inputs = self.clip_processor(text=[query], return_tensors="pt", padding=True).to("cpu")
        feats = self.clip_model.get_text_features(**inputs)  # (1,512)
        feats = feats / feats.norm(dim=-1, keepdim=True)  # cosine-ready
        return feats[0].cpu().numpy().astype(np.float32)

    def retrieve_articles(self, article_db, query: str, k: int = 5, where: dict | None = None):
        q_vec = self.embed_texts([query])[0]  # 384-d
        res = article_db._collection.query(
            query_embeddings=[q_vec.tolist()],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids, docs, metas, dists = self._unwrap(res)
        sims = self._to_similarity(dists)
        return ids, docs, metas, sims

    def retrieve_images_by_text(self, image_db, query: str, k: int = 5, where: dict | None = None):
        q_vec = self.embed_query_clip_text(query)  # 512-d
        res = image_db._collection.query(
            query_embeddings=[q_vec.tolist()],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids, docs, metas, dists = self._unwrap(res)
        sims = self._to_similarity(dists)
        return ids, docs, metas, sims

