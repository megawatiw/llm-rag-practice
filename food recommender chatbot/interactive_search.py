from shared_functions import *

food_items = []

def suggest_related_searches(search_results):
    if not search_results or len(search_results) == 0:
        return
    
    cuisines = list(set([r['cuisine_type'] for r in search_results]))
    for cuisine in cuisines[:3]:  # Limit to 3 suggestions
        print(f"   • Try '{cuisine} dishes' for more {cuisine} options")

    avg_calories = sum([r['food_calories_per_serving'] for r in search_results]) / len(search_results)
    if avg_calories > 350:
        print("   • Try 'low calorie' for lighter options")
    else:
        print("   • Try 'hearty meal' for more substantial dishes")


def handle_food_search(collection, query):
    print(f"\n🔍 Searching for '{query}'...")
    print("   Please wait...")

    results = perform_similarity_search(collection, query, 5)

    if not results or len(results) == 0:
        print("❌ No matching foods found.")
        print("💡 Try different keywords like:")
        print("   • Cuisine types: 'Italian', 'American'")
        print("   • Ingredients: 'chocolate', 'flour', 'cheese'")
        print("   • Descriptors: 'sweet', 'baked', 'dessert'")
        return

    print(f"\n✅ Found {len(results)} recommendations:")
    print("=" * 60)

    for i, item in enumerate(results, 1):
        percentage_score = item['similarity_score']*100
        print(f"\n{i}. 🍽️  {item['food_name']}")
        print(f"   📊 Match Score: {percentage_score:.1f}%")
        print(f"   🏷️  Cuisine: {item['cuisine_type']}")
        print(f"   🔥 Calories: {item['food_calories_per_serving']} per serving")
        print(f"   📝 Description: {item['food_description']}")

        if i < len(results):
            print("   " + "-" * 50)
    
    print("=" * 60)
    suggest_related_searches(results)


def show_help_menu():
    print("\n📖 HELP MENU")
    print("-" * 30)
    print("Search Examples:")
    print("  • 'chocolate dessert' - Find chocolate desserts")
    print("  • 'Italian food' - Find Italian cuisine")
    print("  • 'sweet treats' - Find sweet desserts")
    print("  • 'baked goods' - Find baked items")
    print("  • 'low calorie' - Find lower-calorie options")
    print("\nCommands:")
    print("  • 'help' - Show this help menu")
    print("  • 'quit' - Exit the system")

def interactive_food_chatbot(collection):
    print("\n" + "="*50)
    print("🤖 INTERACTIVE FOOD SEARCH CHATBOT")
    print("="*50)
    print("Commands:")
    print("  • Type any food name or description to search")
    print("  • 'help' - Show available commands")
    print("  • 'quit' or 'exit' - Exit the system")
    print("  • Ctrl+C - Emergency exit")
    print("-" * 50)

    while True:
        try:
            user_input = input("\n🔍 Search for food: ").strip()

            if not user_input:
                print("   Please enter a search term or 'help' for commands")
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using the Food Recommendation System!")
                print("   Goodbye!")
                break
            elif user_input.lower() in ['help', 'h']:
                show_help_menu()
            else:
                handle_food_search(collection, user_input)
        except KeyboardInterrupt:
            print("\n\n👋 System interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error processing request: {e}")


def main():
    try:
        print("🍽️  Interactive Food Recommendation System")
        print("=" * 50)
        print("Loading food database...")

        global food_items
        food_items = load_food_data("./FoodDataSet.json")
        print(f"✅ Loaded {len(food_items)} food items successfully")

        collection = create_similarity_search_collection(
            "interactive_food_search",
            {'description': 'A collection for interactive food search'}
        )
        populate_similarity_collection(collection, food_items)

        interactive_food_chatbot(collection)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()