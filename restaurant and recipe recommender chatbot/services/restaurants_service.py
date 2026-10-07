
class RestaurantService:
    def __init__(self, repo, embedding_service):
        self.repo = repo
        self.embedding_service = embedding_service

    def add_restaurant(self, name, cuisine, price_range, location, description):
        status = self.repo.upsert_restaurant(name, cuisine, price_range, location, description)
        if "successfully" in status:
            # Assuming the restaurant ID is returned or can be fetched after insertion
            restaurant_id = self.repo.get_last_inserted_id()  # You need to implement this method in your repo
            self.embedding_service.enqueue_embedding_job(
                entity_type="restaurant",
                entity_id=restaurant_id,
                modality="text",
                payload={"name": name, "cuisine": cuisine, "description": description}
            )
        return status

    def update_restaurant_info(self, restaurant_id, name, cuisine, price_range, location, description):
        status = self.repo.update_restaurant(restaurant_id, name, cuisine, price_range, location, description)
        if "successfully" in status:
            self.embedding_service.enqueue_embedding_job(
                entity_type="restaurant",
                entity_id=restaurant_id,
                modality="text",
                payload={"name": name, "cuisine": cuisine, "description": description}
            )
        return status

    def get_restaurant_info(self, restaurant_id):
        restaurant = self.repo.get_restaurant(restaurant_id)
        return restaurant

    def get_restaurant_list(self, filters=None):
        results = self.repo.list_restaurants(filters=filters)
        return results

