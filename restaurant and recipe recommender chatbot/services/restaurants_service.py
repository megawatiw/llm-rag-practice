from data.restaurants import Restaurant


class RestaurantService:
    def __init__(self, conn):
        self.conn = conn

    def upsert_restaurant(self, name, cuisine, price_range, location, description):
        self.conn.connect()

        # Insert the new restaurant into the database
        try:
            self.conn.insert_item(f"""
                INSERT INTO restaurants (name, type, price_range, location, environment)
                VALUES (%s, %s, %s, %s, %s)
            """, (name, cuisine, price_range, location, description))
            status = f"Restaurant '{name}' added successfully!"
        except Exception as e:
            status = f"Error adding restaurant: {e}"

        self.conn.disconnect()
        return status

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

    def get_restaurant(self, restaurant_id: int) -> Restaurant:
        self.conn.connect()
        self.conn.cursor.execute("SELECT * FROM restaurants WHERE id = %s", (restaurant_id,))
        row = self.conn.cursor.fetchone()
        self.conn.disconnect()

        if row:
            return Restaurant(
                id=row[0],
                name=row[1],
                type=row[2],
                price_range=row[3],
                location=row[4],
                environment=row[5]
            )
        else:
            raise ValueError("Restaurant not found")

    def update_restaurant(self, restaurant_id: int, name: str, cuisine: str, price_range: str, location: str, description: str) -> str:
        self.conn.connect()
        try:
            self.conn.cursor.execute("""
                UPDATE restaurants
                SET name = %s, type = %s, price_range = %s, location = %s, environment = %s
                WHERE id = %s
            """, (name, cuisine, price_range, location, description, restaurant_id))
            self.conn.conn.commit()
            status = f"Restaurant '{name}' updated successfully!"
        except Exception as e:
            status = f"Error updating restaurant: {e}"
        finally:
            self.conn.disconnect()
        return status

    def delete_restaurant(self, restaurant_id: int) -> str:
        self.conn.connect()
        try:
            self.conn.cursor.execute("DELETE FROM restaurants WHERE id = %s", (restaurant_id,))
            self.conn.conn.commit()
            status = f"Restaurant with ID {restaurant_id} deleted successfully!"
        except Exception as e:
            status = f"Error deleting restaurant: {e}"
        finally:
            self.conn.disconnect()
        return status

    def list_restaurants(self) -> list:
        self.conn.connect()
        self.conn.cursor.execute("SELECT * FROM restaurants")
        rows = self.conn.cursor.fetchall()
        self.conn.disconnect()

        restaurants = []
        for row in rows:
            restaurants.append(Restaurant(
                id=row[0],
                name=row[1],
                type=row[2],
                price_range=row[3],
                location=row[4],
                environment=row[5]
            ))
        return restaurants

    def get_restaurant_embeddings(self, restaurant_id: int) -> list:
        self.conn.connect()
        self.conn.cursor.execute("SELECT embeddings FROM restaurants_text_embeddings WHERE id = %s", (restaurant_id,))
        row = self.conn.cursor.fetchone()
        self.conn.disconnect()

        if row:
            return row[0]
        else:
            raise ValueError("Restaurant embeddings not found")

