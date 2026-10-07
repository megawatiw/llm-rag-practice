
class MenuItemsService:
    def __init__(self, menu_items_repository):
        self.menu_items_repository = menu_items_repository

    def get_all_menu_items(self):
        return self.menu_items_repository.get_all()

    def get_menu_item_by_id(self, item_id):
        return self.menu_items_repository.get_by_id(item_id)

    def create_menu_item(self, menu_item_data):
        return self.menu_items_repository.create(menu_item_data)

    def update_menu_item(self, item_id, menu_item_data):
        return self.menu_items_repository.update(item_id, menu_item_data)

    def delete_menu_item(self, item_id):
        return self.menu_items_repository.delete(item_id)