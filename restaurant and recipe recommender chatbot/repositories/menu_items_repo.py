
class MenuItemsRepo:
    def __init__(self, db):
        self.db = db

    def get_all_menu_items(self):
        return self.db.query("SELECT * FROM menu_items")

    def get_menu_item_by_id(self, item_id):
        # TODO: fix parameters
        return self.db.query("SELECT * FROM menu_items WHERE id = ?", (item_id,))

    def add_menu_item(self, name, price):
        # TODO: fix parameters
        self.db.execute("INSERT INTO menu_items (name, price) VALUES (?, ?)", (name, price))

    def update_menu_item(self, item_id, name, price):
        # TODO: fix parameters
        self.db.execute("UPDATE menu_items SET name = ?, price = ? WHERE id = ?", (name, price, item_id))

    def delete_menu_item(self, item_id):
        self.db.execute("DELETE FROM menu_items WHERE id = ?", (item_id,))