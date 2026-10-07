import json

class EmbeddingRepo:
    def __init__(self, conn):
        self.conn = conn

    def upsert_restaurant_embeddings(self, restaurant_id: int, embeddings: list) -> str:
        self.conn.connect()
        try:
            self.conn.cursor.execute("""
                UPDATE restaurants_text_embeddings
                SET embeddings = %s
                WHERE id = %s
            """, (embeddings, restaurant_id))
            self.conn.conn.commit()
            status = f"Embeddings for restaurant ID {restaurant_id} updated successfully!"
        except Exception as e:
            status = f"Error updating embeddings: {e}"
        finally:
            self.conn.disconnect()
        return status

    def get_restaurant_embeddings(self, restaurant_id: int) -> list:
        self.conn.connect()
        self.conn.cursor.execute("SELECT embeddings FROM restaurants_text_embeddings WHERE id = %s", (restaurant_id,))
        row = self.conn.cursor.fetchone()
        self.conn.disconnect()

        if row:
            return row[0]
        else:
            raise ValueError("Restaurant embeddings not found")

    def enqueue_embedding_job(self, entity_type, entity_id, modality, payload):
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO embedding_jobs (entity_type, entity_id, modality, payload, status)
                VALUES (%s, %s, %s, %s, 'pending')
                RETURNING id
            """, (entity_type, entity_id, modality, json.dumps(payload)))
            return cur.fetchone()[0]