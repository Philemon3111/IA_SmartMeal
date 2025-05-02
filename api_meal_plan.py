import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import random
from flask import Flask, jsonify, request
import re
from collections import defaultdict
from fractions import Fraction
app = Flask(__name__)




# Charger les données sauvegardées
try:
    df = pd.read_pickle("recipes_df_v2.pkl")
    with open("label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    model = tf.keras.models.load_model("meal_plan_model.keras")
    print("Model and data loaded successfully")
except FileNotFoundError as e:
    print(f"Error: Missing file - {e}")
    exit(1)

# print("DataFrame columns:", df.columns.tolist())
# print("\nFirst few rows:")
# print(df.head())

from deep_translator import GoogleTranslator

def translate_francais_meal_plan(data):
    translator = GoogleTranslator(source='auto', target='fr')
    for day, meals in data.items():
        for meal in meals:
            meal["items"] = [translator.translate(item) for item in meal.get("items", [])]
            meal["ingredients"] = [translator.translate(ing) for ing in meal.get("ingredients", [])]
    return data
# Liste des ingrédients non-végans (sans butter)
NON_VEGAN_INGREDIENTS = [
    "meat", "beef", "pork", "chicken", "turkey", "fish", "salmon", "tuna", "shrimp", "crab", "lobster",
    "milk", "cheese", "cream", "yogurt", "egg", "eggs", "honey", "gelatin"
]

# Dictionnaire pour les allergènes avec mots-clés précis
ALLERGEN_KEYWORDS = {
    "lait": ["milk", "cheese", "cream", "yogurt", "whey"],
    "œufs": ["egg", "eggs", "mayonnaise"],
    "moutarde": ["mustard", "mustard seed"],
    "cacahuètes": ["peanut", "peanut oil", "peanut butter"],
    "fruits à coque": ["almond", "hazelnut", "walnut", "cashew", "pistachio"]
}

# Normaliser les ingrédients au chargement
def normalize_ingredient(ingredient):
    translations = {
        "lait": "milk",
        "œufs": "egg",
        "moutarde": "mustard",
        "cacahuètes": "peanut",
        "fruits à coque": "nut"
    }
    for fr, en in translations.items():
        ingredient = ingredient.replace(fr, en)
    return ingredient.lower()

df["ingredients"] = df["ingredients"].apply(lambda x: [normalize_ingredient(ing) for ing in x])
# print("Exemple d'ingrédients normalisés :", df["ingredients"].iloc[:5].tolist())

# Fonction pour vérifier si une recette respecte les contraintes
def is_recipe_valid(recipe, allergies, diet, max_calories=None):
    ingredients = [ing.lower() for ing in recipe["ingredients"]]
    
    # Vérifier les allergies
    for allergen, is_allergic in allergies.items():
        if is_allergic:
            keywords = ALLERGEN_KEYWORDS.get(allergen, [allergen.lower()])
            if any(any(keyword in ing for keyword in keywords) for ing in ingredients):
                print(f"Recette '{recipe['title']}' rejetée pour allergène : {allergen}")
                return False
    
    # Vérifier le régime végan
    if diet.lower() == "végan":
        for ing in ingredients:
            if any(non_vegan in ing for non_vegan in NON_VEGAN_INGREDIENTS):
                print(f"Recette '{recipe['title']}' rejetée pour ingrédient non-végan : {ing}")
                return False
    
    # Vérifier les calories
    if max_calories and recipe["calories"] > max_calories:
        print(f"Recette '{recipe['title']}' rejetée pour calories : {recipe['calories']} > {max_calories}")
        return False
    
    return True

# Fonction pour générer un plan de repas personnalisé
def generate_meal_plan(preferences):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    meal_plan = {}
    
    # Extraire les préférences
    allergies = preferences.get("allergy", {})
    diet = preferences.get("diet", "none")
    goal = preferences.get("goal", "none")
    number_of_meals = preferences.get("number_of_meals", 6)
    grocery_day = preferences.get("grocery_day", "Monday")
    max_calories = preferences.get("max_calories", None)
    
    # Ajuster les jours à partir du grocery_day
    try:
        start_idx = days.index(grocery_day)
        days = days[start_idx:] + days[:start_idx]
    except ValueError:
        pass
    
    # Déterminer le seuil de calories pour "lose weight"
    if goal.lower() == "lose weight" and max_calories is None:
        max_calories = np.percentile(df["calories"], 75)  # 75e percentile pour plus de recettes
        print(f"Seuil de calories pour 'lose weight' (75e percentile) : {max_calories}")
    elif max_calories is not None:
        print(f"Seuil de calories spécifié par l'utilisateur : {max_calories}")
    else:
        print("Aucune contrainte de calories appliquée")
    
    # Préfiltrer les recettes valides
    valid_recipes = df[df.apply(lambda row: is_recipe_valid(row, allergies, diet, max_calories), axis=1)]
    print(f"Nombre de recettes valides après filtrage : {len(valid_recipes)}")
    if valid_recipes.empty:
        reasons = []
        if any(allergies.values()):
            reasons.append("Allergies too restrictive")
        if diet.lower() == "végan":
            reasons.append("No vegan recipes available")
        if max_calories:
            reasons.append(f"No recipes with calories below {max_calories}")
        return {"error": "No recipes match the preferences", "details": reasons}
    
    # Répartir les repas sur les jours
    meals_per_day = max(1, number_of_meals // len(days))
    remaining_meals = number_of_meals % len(days)
    
    used_titles = set()
    for i, day in enumerate(days):
        num_meals = meals_per_day + (1 if i < remaining_meals else 0)
        meals = []
        
        for _ in range(num_meals):
            type_plat = random.choice(["entrée", "plat principal", "dessert"])
            recettes_type = valid_recipes[valid_recipes["type_plat"] == type_plat]
            if recettes_type.empty:
                print(f"Aucune recette valide pour {type_plat} le {day}")
                continue
            recette = recettes_type.sample(n=1).iloc[0]
            if recette["title"] in used_titles:
                continue
            meals.append({
                "items": [recette["title"] or "Untitled dish"],
                "calories": int(recette["calories"]),
                "servings": int(recette["servings"]),
                "time": int(recette["time"]),
                "ingredients": recette["ingredients"],
                "NER": recette["NER"]
                })
            used_titles.add(recette["title"])
        
        meal_plan[day] = meals
    
    # Rééquilibrer si trop peu de repas
    total_meals = sum(len(meals) for meals in meal_plan.values())
    if total_meals < number_of_meals:
        print(f"Avertissement : Seulement {total_meals} repas générés sur {number_of_meals} demandés")
        for day in days:
            if not meal_plan[day] and total_meals < number_of_meals:
                recettes_type = valid_recipes[valid_recipes["type_plat"] == random.choice(["entrée", "plat principal", "dessert"])]
                if not recettes_type.empty:
                    recette = recettes_type.sample(n=1).iloc[0]
                    if recette["title"] not in used_titles:
                        meal_plan[day].append({
                            "items": [recette["title"] or "Untitled dish"],
                            "calories": int(recette["calories"]),
                            "servings": int(recette["servings"]),
                            "time": int(recette["time"]),
                            "ingredients": recette["ingredients"],
                            "NER": recette["NER"]
                        })
                        used_titles.add(recette["title"])
                        total_meals += 1
    
    return meal_plan

# Fonction pour générer une liste de courses (inchangée)
def generate_shopping_list(meal_plan):
    ingredients_set = {}
    
    def parse_quantity(qty_str):
        try:
            qty_str = qty_str.replace("g", "").replace("ml", "").replace("unités", "").strip()
            if "/" in qty_str:
                num, denom = qty_str.split("/")
                return float(num.strip()) / float(denom.strip())
            return float(qty_str)
        except (ValueError, ZeroDivisionError):
            return 100.0
    
    for day, meals in meal_plan.items():
        for meal in meals:
            title = meal["items"][0]
            recette = df[df["title"] == title]
            if not recette.empty:
                ingredients = recette.iloc[0]["ingredients"]
                for ingredient in ingredients:
                    parts = ingredient.split(",")[0].split()
                    quantity = next((p for p in parts if any(c.isdigit() for c in p) or "/" in p), "100")
                    unit = next((p for p in parts if p in ["g", "ml", "unités"]), "g")
                    name = " ".join(p for p in parts if p != quantity and p != unit)
                    key = name.strip()
                    
                    if key in ingredients_set:
                        current_qty = parse_quantity(ingredients_set[key])
                        new_qty = parse_quantity(quantity)
                        ingredients_set[key] = f"{current_qty + new_qty}{unit}"
                    else:
                        ingredients_set[key] = f"{quantity}{unit}"
    
    return ingredients_set

# Routes existantes
@app.route('/meal_plan', methods=['GET'])
def get_meal_plan():
    meal_plan = generate_meal_plan({})  # Appel sans préférences
    return jsonify(meal_plan)

def clean_and_categorize_ingredients(meal_plan):
    ingredients_list = set()

    for day, meals in meal_plan.items():
        for meal in meals:
            ner_items = meal.get("NER", [])
            ingredients = meal.get("ingredients", [])

            ingredients_text = " ".join(ingredients).lower()

            for ner in ner_items:
                if ner.lower() in ingredients_text:
                    ingredients_list.add(ner)
    # Remove invalid entries (errors, empty strings, etc.)
    cleaned = [
        ing for ing in ingredients_list 
        if isinstance(ing, str) 
        and not ing.startswith(('Error', 'N', 'O', 'Votre choix'))
    ]
    
    # Define category keywords (case-insensitive)
    categories = {
        "oils_fats": [
            "huile", "huile d'olive", "huile de maïs", "huile de sésame", "huile végétale", 
            "lard", "margarine", "mayonnaise", "shortening végétal"
        ],
        "fruits_vegetables": [
            "abricots", "ail", "ananas", "artichauts", "artichauts gelé", "aubergine", 
            "banane", "bananes", "bleuets", "bleuets frais", "brocoli", "brocoli frais", 
            "brocoli gelé", "carotte", "carotte râpée", "carottes", "céleri", "cerise noire", 
            "cerises", "champignon", "champignons", "champignons frais", "chou", "chou rouge", 
            "chou vert", "chou-fleur", "citron", "citron vert", "citrons", "citrouille", 
            "concombre", "concombres", "courge jaune", "courgettes", "dates", "échalotes", 
            "épinard", "fraises", "fraises fraîches", "framboise", "framboises", "gingembre frais", 
            "haricots verts", "haricots verts gelés", "laitue romaine", "légumes", "légumes mélangés", 
            "légumes mélangés surgelés", "légumes verts", "mangue", "navets", "oignon", 
            "oignon blanc", "oignon frais", "oignon rouge", "oignon vert", "olive", 
            "olives", "orange", "oranges", "pomme", "pomme de terre", "pommes", "pommes de terre", 
            "pommes de terre rouges", "pommes fraîches", "pommes vertes", "potiron", "pruneaux", 
            "raisin", "raisins", "raisins secs", "rhubarbe", "tomate", "tomate fraîche", "tomates"
        ],
        "meat_seafood": [
            "agneau", "anchois", "bacon", "bœuf", "bœuf haché maigre", "crabe", "crevette", 
            "dinde", "escalopes de poulet", "filet de poisson", "filet de porc", "jambon", 
            "morceau de poulet blanc", "palourdes", "poisson blanc ferme", "poitrine de boeuf", 
            "poitrines de poulet", "porc", "poulet", "saucisse", "saumon", "thon", "viande hachée", "steak"
        ],
        "dairy_eggs": [
            "babeurre", "beurre", "beurre non salé", "blanc d'oeuf", "blancs d'œufs", 
            "crème", "crème condensée", "crème fouettée", "crème fraîche", "crème légère", 
            "crème sure commerciale", "fromage", "fromage blanc", "fromage cheddar", 
            "fromage à la crème", "jaune d'oeuf", "jaunes d'oeuf", "lait", "lait condensé", 
            "lait de coco", "lait en poudre", "œuf", "Œufs", "yaourt", "yaourt nature non gras"
        ],
        "herbs_spices": [
            "aneth", "aneth frais", "anis", "basilic", "basilic doux", "cannelle", "cardamome", 
            "cayenne", "ciboulette", "coriandre", "cumin", "curcuma", "curry", "estragon", 
            "feuille de laurier", "gingembre en poudre", "marjolaine", "origan", "paprika", 
            "persil", "persil frais", "piment", "poivre", "poivre blanc", "poivre de Cayenne", 
            "romarin", "safran", "sel", "thym", "vanille"
        ],
        "grains_cereals": [
            "Arborio", "avoine", "avoine de cuisson", "avoine roulée", "biscuits", "chapelure", 
            "corn flakes", "farine", "farine de blé entier", "farine de maïs", "gruau", 
            "macaroni", "nouilles", "nouilles aux œufs", "orge perlée", "pâtes", "riz", 
            "riz brun", "riz sauvage", "spaghetti", "pain"
        ],
        "legumes_nuts": [
            "amande", "amandes", "arachide", "arachides", "haricots", "haricots blancs", 
            "haricots de Lima", "haricots pinto", "haricots rouges", "haricots verts", 
            "lentilles", "noix", "noix de cajou", "noix de coco", "noix de pécan", "pacanes", 
            "petits pois", "pois verts", "soja"
        ],
        "baking_sweets": [
            "bicarbonate de soude", "cacao", "cacao en poudre", "chocolat", "chocolat au lait", 
            "chocolat non sucré", "confiture", "gâteau", "gâteau au chocolat", "gélatine", 
            "guimauves", "miel", "sucre", "sucre en poudre", "vanille", "mélasse"
        ],
        "condiments_sauces": [
            "barbecue sauce", "ketchup", "moutarde", "sauce", "sauce au poivre", 
            "sauce tomate", "sauce worcestershire", "vinaigre", "vinaigre de cidre"
        ],
        "beverages": [
            "bière", "café", "coca-cola", "jus d'ananas", "jus de citron", "jus de tomate", 
            "thé", "vin blanc", "vin rouge"
        ],
        "canned_packaged": [
            "bouillon de bœuf", "bouillon de poulet", "concentré de jus d'orange", 
            "conserve d'abricot", "cornichons", "purée de tomates", "soupe aux champignons"
        ],
        "miscellaneous": [
            "colorant alimentaire", "extrait de vanille", "levure", "poudre à pâte", 
            "sel de céleri", "semoule"
        ]
    }
        
    # Categorize ingredients
    categorized = {key: [] for key in categories}
    uncategorized = []
    
    for ingredient in cleaned:
        lower_ing = ingredient.lower()
        found = False
        
        for category, keywords in categories.items():
            if any(keyword in lower_ing for keyword in keywords):
                categorized[category].append(ingredient)
                found = True
                break
                
        if not found:
            uncategorized.append(ingredient)
    
    # Sort each category alphabetically
    for category in categorized:
        categorized[category] = sorted(categorized[category], key=lambda x: x.lower())
    
    return categorized, uncategorized



def parse_quantity_min_1(qty_str):
    try:
        qty_str = qty_str.strip()
        if re.match(r'^\d+ \d+/\d+$', qty_str):  # e.g., "2 1/4"
            parts = qty_str.split()
            value = float(parts[0]) + float(Fraction(parts[1]))
        elif re.match(r'^\d+/\d+$', qty_str):  # e.g., "1/2"
            value = float(Fraction(qty_str))
        else:
            match = re.match(r'^([\d.]+)', qty_str)
            if match:
                value = float(match.group(1))
            else:
                return 0
        return max(1, int(round(value)))  # Ensure minimum 1
    except:
        return 0

def extract_quantities(meal_plan, categorized):
    quantities = {
        "meat_seafood": defaultdict(list),
        "dairy_eggs": defaultdict(list),
        "fruits_vegetables": defaultdict(list)
    }

    for day, meals in meal_plan.items():
        for meal in meals:
            ingredient_lines = meal.get("ingredients", [])

            for line in ingredient_lines:
                line_lower = line.lower()

                for category in quantities:
                    for item in categorized.get(category, []):
                        if item.lower() in line_lower:
                            # Try to extract quantity with unit first
                            match = re.search(r"(\d+(?:[.,]\d+)?)(?:\s*)(pt|lb|kg|g|ml|l|cuillères?|tsp|oz|pkg|tranches?)?", line_lower, re.IGNORECASE)
                            if match:
                                number_str = match.group(1).replace(',', '.')
                                unit = match.group(2).lower() if match.group(2) else "count"  # default to "count" if no unit
                                try:
                                    number_val = float(number_str)
                                    quantities[category][item].append((number_val, unit))
                                except ValueError:
                                    pass  # skip invalid number
                            else:
                                # No quantity found at all — fallback to default count = 1
                                quantities[category][item].append((1.0, "count"))

    return quantities

def flatten_quantities(quantities):
    flattened = {}

    for category, items in quantities.items():
        flattened[category] = {}

        for item, qty_list in items.items():
            if category == "meat_seafood":
                total_grams = 0
                total_packs = 0

                for value, unit in qty_list:
                    unit = unit.lower()
                    if unit in ["g", "gram", "grams"]:
                        total_grams += value
                    elif unit in ["kg", "kilogram", "kilograms"]:
                        total_grams += value * 1000
                    elif unit in ["lb", "lbs", "pound", "pounds"]:
                        total_grams += value * 453.592
                    else:
                        # Treat as pack (pkg, tranches, etc.)
                        total_packs += value

                result = {}
                if total_grams > 0:
                    result["grams"] = int(round(total_grams))
                if total_packs > 0:
                    result["packs"] = int(round(total_packs))

                flattened[category][item] = result

            else:
                # For other categories, just sum values regardless of unit
                total = defaultdict(float)
                for value, unit in qty_list:
                    total[unit] += value
                flattened[category][item] = {
                    unit: int(round(amount)) for unit, amount in total.items()
                }

    return flattened


@app.route('/shopping_list', methods=['POST'])
def get_shopping_list():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    meal_plan = request.get_json()
    shopping_list = clean_and_categorize_ingredients(meal_plan)
    return jsonify(shopping_list, flatten_quantities(extract_quantities(meal_plan, shopping_list[0])))

# Route pour un plan de repas personnalisé
@app.route('/custom_meal_plan', methods=['POST'])
def get_custom_meal_plan():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    preferences = request.get_json()
    
    # Valider les préférences
    required_fields = ["allergy", "diet", "goal", "number_of_meals", "grocery_day"]
    for field in required_fields:
        if field not in preferences:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    # Vérifier les allergies
    if not isinstance(preferences["allergy"], dict):
        return jsonify({"error": "Allergy must be a dictionary"}), 400
    
    # Vérifier le régime
    if preferences["diet"].lower() not in ["végan", "none"]:
        return jsonify({"error": "Unsupported diet"}), 400
    
    # Vérifier l'objectif
    if preferences["goal"].lower() not in ["lose weight", "maintain", "gain weight"]:
        return jsonify({"error": "Unsupported goal"}), 400
    
    # Vérifier le nombre de repas
    if not isinstance(preferences["number_of_meals"], int) or preferences["number_of_meals"] < 1:
        return jsonify({"error": "Invalid number_of_meals"}), 400
    
    # Vérifier max_calories (facultatif)
    if "max_calories" in preferences:
        if not isinstance(preferences["max_calories"], (int, float)) or preferences["max_calories"] <= 0:
            return jsonify({"error": "Invalid max_calories"}), 400
    
    # Générer le plan de repas
    meal_plan = generate_meal_plan(preferences)
    
    if "error" in meal_plan:
        return jsonify(meal_plan), 400
    
    return jsonify(meal_plan)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)