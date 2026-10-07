def _save_to_chromadb(restaurant_list):
    db_dir = get_chromadb_dir()

    if os.path.isdir(db_dir):
        shutil.rmtree(db_dir)  # Reset vector DB (important for reruns)

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

    article_db = Chroma(
        collection_name="restaurant_articles",
        persist_directory=db_dir,
    )

    article_db._collection.upsert(
        ids=[d.metadata["doc_id"] for d in article_docs],
        embeddings=A.tolist(),
        documents=[d.page_content for d in article_docs],
        metadatas=[d.metadata for d in article_docs],
    )

def _save_recipe_to_chromadb(recipe_list):
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