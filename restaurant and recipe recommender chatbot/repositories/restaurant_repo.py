from data.restaurants import Restaurant

class RestaurantRepo:
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

    def get_restaurant(self, restaurant_id: int) -> dict:
        self.conn.connect()
        self.conn.cursor.execute("SELECT * FROM restaurants WHERE id = %s", (restaurant_id,))
        row = self.conn.cursor.fetchone()
        self.conn.disconnect()

        if row:
            return {
                "id": row[0],
                "name": row[1],
                "type": row[2],
                "price_range": row[3],
                "location": row[4],
                "environment": row[5]
            }
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

    def list_restaurants(self, filters=None) -> list:
        if filters:
            query = "SELECT * FROM restaurants WHERE "
            conditions = []
            values = []
            for key, value in filters.items():
                conditions.append(f"{key} = %s")
                values.append(value)
            query += " AND ".join(conditions)
        else:
            query = "SELECT * FROM restaurants"

        self.conn.connect()
        self.conn.cursor.execute(query)
        rows = self.conn.cursor.fetchall()
        self.conn.disconnect()

        restaurants = []
        for row in rows:
            restaurants.append(
                {
                    "id": row[0],
                    "name": row[1],
                    "type": row[2],
                    "price_range": row[3],
                    "location": row[4],
                    "environment": row[5]
                }
            )
        return restaurants