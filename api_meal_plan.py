import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import random
import requests
from flask import Flask, jsonify, request
import re
from collections import defaultdict
from fractions import Fraction
import unicodedata

app = Flask(__name__)

# Charger les données sauvegardées
try:
    df = pd.read_pickle("recipes_df_v2.pkl")
    with open("label_encoder_v2.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("scaler_v2.pkl", "rb") as f:
        scaler = pickle.load(f)
    model = tf.keras.models.load_model("meal_plan_model_v2.keras")
    print("Model and data loaded successfully")
except FileNotFoundError as e:
    print(f"Error: Missing file - {e}")
    exit(1)

# Liste des ingrédients non-végétariens (viandes, poissons, gélatine)
NON_VEGETARIAN_INGREDIENTS = [
    "meat", "beef", "pork", "chicken", "turkey", "fish", "salmon", "tuna", "shrimp", "crab", "lobster",
    "viande", "bœuf", "boeuf", "porc", "poulet", "dinde", "poisson", "saumon", "thon", "crevette", "crabe",
    "homard", "gelatin", "gélatine", "gelatine"
]

# Liste des ingrédients exclus pour le régime végan (laitiers, œufs, miel)
VEGAN_EXCLUDED_INGREDIENTS = [
    "milk", "cheese", "cream", "yogurt", "egg", "eggs", "honey",
    "lait", "fromage", "crème", "creme", "yaourt", "œuf", "oeuf", "œufs", "oeufs", "miel"
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

# Fonction pour vérifier si une recette respecte les contraintes
def is_recipe_valid(recipe, allergies, diet, max_calories=None):
    ingredients = [ing.lower() for ing in recipe["ingredients"]]
    
    # Vérifier les allergènes
    for allergen, is_allergic in allergies.items():
        if is_allergic:
            keywords = ALLERGEN_KEYWORDS.get(allergen, [allergen.lower()])
            if any(any(keyword in ing for keyword in keywords) for ing in ingredients):
                print(f"Recette '{recipe['title']}' rejetée pour allergène : {allergen}")
                return False
    
    # Vérifier le régime végétarien
    if diet.lower() == "végétarien":
        for ing in ingredients:
            if any(non_veg in ing for non_veg in NON_VEGETARIAN_INGREDIENTS):
                print(f"Recette '{recipe['title']}' rejetée pour ingrédient non-végétarien : {ing}")
                return False
    
    # Vérifier le régime végan
    if diet.lower() == "végan":
        for ing in ingredients:
            if any(non_vegan in ing for non_vegan in NON_VEGETARIAN_INGREDIENTS + VEGAN_EXCLUDED_INGREDIENTS):
                print(f"Recette '{recipe['title']}' rejetée pour ingrédient non-végan : {ing}")
                return False
    
    # Vérifier les calories
    if max_calories and recipe["calories"] > max_calories:
        print(f"Recette '{recipe['title']}' rejetée pour calories : {recipe['calories']} > {max_calories}")
        return False
    
    return True

# Fonction pour générer un plan de repas
def generate_meal_plan(preferences=None, inventory_ingredients=None):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    meal_plan = {}
    weekly_selected_indices = set()
    
    if preferences:
        allergies = preferences.get("allergy", {})
        diet = preferences.get("diet", "none")
        goal = preferences.get("goal", "none")
        number_of_meals = preferences.get("number_of_meals", 6)
        grocery_day = preferences.get("grocery_day", "Monday")
        max_calories = preferences.get("max_calories", None)
        
        try:
            start_idx = days.index(grocery_day)
            days = days[start_idx:] + days[:start_idx]
        except ValueError:
            pass
        
        if goal.lower() == "lose weight" and max_calories is None:
            max_calories = np.percentile(df["calories"], 90)
            print(f"Seuil de calories pour 'lose weight' (90e percentile) : {max_calories}")
        elif max_calories is not None:
            print(f"Seuil de calories spécifié par l'utilisateur : {max_calories}")
        
        valid_recipes = df[df.apply(lambda row: is_recipe_valid(row, allergies, diet, max_calories), axis=1)]
        print(f"Nombre de recettes valides après filtrage : {len(valid_recipes)}")
        if valid_recipes.empty:
            reasons = []
            if any(allergies.values()):
                reasons.append("Allergies too restrictive")
            if diet.lower() == "végan":
                reasons.append("No vegan recipes available")
            if diet.lower() == "végétarien":
                reasons.append("No vegetarian recipes available")
            if max_calories:
                reasons.append(f"No recipes with calories below {max_calories}")
            print(f"Raisons du filtrage vide : {reasons}")
            return {"error": "No recipes match the preferences", "details": reasons}
        
        meals_per_day = max(1, number_of_meals // len(days))
        remaining_meals = number_of_meals % len(days)
    else:
        valid_recipes = df
        number_of_meals = 6
        meals_per_day = 1
        remaining_meals = 0
    
    # Calculer les scores d'ingrédients si inventory_ingredients est fourni
    ingredient_scores = None
    if inventory_ingredients:
        ingredient_scores = np.zeros(len(valid_recipes))
        inventory_ingredients = [normalize_ingredient(ing) for ing in inventory_ingredients]
        for idx, recipe in valid_recipes.iterrows():
            recipe_ingredients = [ing.lower() for ing in recipe["ingredients"]]
            matches = sum(1 for inv_ing in inventory_ingredients if any(inv_ing in recipe_ing for recipe_ing in recipe_ingredients))
            ingredient_scores[valid_recipes.index.get_loc(recipe.name)] = matches
        # Normaliser les scores (0 à 1)
        max_score = ingredient_scores.max() if ingredient_scores.max() > 0 else 1
        ingredient_scores = ingredient_scores / max_score
        print(f"Scores d'ingrédients calculés : min={ingredient_scores.min()}, max={ingredient_scores.max()}")

    for i, day in enumerate(days):
        num_meals = meals_per_day + (1 if i < remaining_meals else 0) if preferences else random.randint(1, 2)
        meals = []
        daily_selected_indices = set()
        
        for _ in range(num_meals):
            type_plat = random.choice(["entrée", "plat principal", "dessert"])
            try:
                type_plat_encoded = encoder.transform([type_plat])[0]
                X_input = scaler.transform([[type_plat_encoded, random.randint(200, 800), random.randint(15, 60)]])
                prediction = model.predict(X_input, verbose=0)[0]
                print(f"Taille de prediction pour {type_plat} le {day}: {len(prediction)}")
                
                if preferences or inventory_ingredients:
                    valid_indices = list(valid_recipes.index)
                    print(f"Taille de valid_indices pour {type_plat} le {day}: {len(valid_indices)}")
                    if len(valid_indices) == 0:
                        print(f"Aucune recette valide pour {type_plat} le {day}")
                        continue
                    # Vérifier les indices valides
                    valid_indices = [i for i in valid_indices if i < len(df)]
                    if len(valid_indices) == 0:
                        print(f"Aucun indice valide après vérification pour {type_plat} le {day}")
                        continue
                    valid_probs = prediction[valid_indices][:len(df)-1]  # -1 pour limiter à 998
                    print(f"Taille de valid_probs pour {type_plat} le {day}: {len(valid_probs)}")
                    if len(valid_probs) != len(valid_indices):
                        print(f"Erreur : valid_probs ({len(valid_probs)}) et valid_indices ({len(valid_indices)}) ont des tailles différentes")
                        recette_index = random.choice(valid_indices)
                    elif valid_probs.sum() == 0 or np.isnan(valid_probs).any():
                        print(f"Avertissement : Probabilités invalides pour {type_plat} le {day}, sélection aléatoire")
                        recette_index = random.choice(valid_indices)
                    else:
                        valid_probs = valid_probs / valid_probs.sum()
                        # Ajuster les probabilités avec les scores d'ingrédients
                        if ingredient_scores is not None:
                            valid_scores = ingredient_scores[:len(valid_indices)]
                            valid_probs = valid_probs * (0.5 + 0.5 * valid_scores)  # Combiner probabilités et scores
                            valid_probs = valid_probs / valid_probs.sum()  # Renormaliser
                        sequential_indices = list(range(len(valid_indices)))
                        for _ in range(10):
                            seq_index = np.random.choice(sequential_indices, p=valid_probs)
                            recette_index = valid_indices[seq_index]
                            if recette_index not in weekly_selected_indices:
                                break
                        else:
                            seq_index = np.random.choice(sequential_indices, p=valid_probs)
                            recette_index = valid_indices[seq_index]
                else:
                    valid_indices = list(range(len(df)-1))  # -1 pour 0 à 998
                    print(f"Taille de valid_indices pour {type_plat} le {day}: {len(valid_indices)}")
                    if len(prediction) == 0:
                        print(f"Aucune prédiction disponible pour {type_plat} le {day}")
                        continue
                    prediction = prediction[:len(df)-1] / prediction.sum()  # -1 pour 0 à 998
                    for _ in range(10):
                        recette_index = np.random.choice(valid_indices, p=prediction)
                        if recette_index not in weekly_selected_indices:
                            break
                    else:
                        recette_index = np.random.choice(valid_indices, p=prediction)
                
                if recette_index not in daily_selected_indices and recette_index < len(df):
                    daily_selected_indices.add(recette_index)
                    weekly_selected_indices.add(recette_index)
                    recette = df.iloc[recette_index]
                    meals.append({
                        "items": [recette["title"] or "Plat sans titre"],
                        "calories": int(recette["calories"]),
                        "servings": int(recette["servings"]),
                        "time": int(recette["time"]),
                        "ingredients": recette["ingredients"],
                        "NER": recette["NER"]
                    })
                    print(f"Recette sélectionnée : {recette['title']} (index {recette_index}) pour {type_plat} le {day}")
                else:
                    print(f"Index {recette_index} déjà utilisé ou invalide pour {type_plat} le {day}")
            except (ValueError, IndexError) as e:
                print(f"Erreur lors de la sélection de recette pour {type_plat} le {day}: {e}")
                continue
        
        meal_plan[day] = meals
    
    return meal_plan

# Routes existantes
@app.route('/meal_plan', methods=['GET'])
def get_meal_plan():
    meal_plan = generate_meal_plan({})
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


def normalize_name(name):
    return unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode().lower().strip()

def subtract_inventory(shopping_list, inventory):
    # Flatten inventory into normalized name → (quantity, unit)
    inv_map = {}

    for item in inventory.get("grocery", []) + inventory.get("fresh_produce", []):
        name = normalize_name(item["name"])
        quantity = float(item["quantity"].replace(',', '.'))
        unit = item["type_quantity"].lower()
        inv_map[name] = (quantity, unit)

    # Adjust the shopping list
    for category, items in shopping_list.items():
        for item, data in list(items.items()):  # Use list() to allow removal
            norm_item = normalize_name(item)

            if norm_item in inv_map:
                inv_qty, inv_unit = inv_map[norm_item]

                # Meat/seafood special case
                if category == "meat_seafood":
                    if "grams" in data and inv_unit in ["g", "gram", "grams"]:
                        data["grams"] = max(0, data["grams"] - inv_qty)
                        if data["grams"] == 0:
                            del data["grams"]
                    if "packs" in data and inv_unit in ["pack", "packs", "pkg"]:
                        data["packs"] = max(0, data["packs"] - inv_qty)
                        if data["packs"] == 0:
                            del data["packs"]

                else:
                    for unit in list(data.keys()):
                        if normalize_name(unit) == normalize_name(inv_unit) or (unit == "count" and inv_unit == ""):
                            data[unit] = max(0, data[unit] - inv_qty)
                            if data[unit] == 0:
                                del data[unit]

            # Remove item if it's now empty
            if not data:
                del shopping_list[category][item]

    return shopping_list

@app.route('/shopping_list', methods=['POST'])
def get_shopping_list():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    meal_plan = data.get("meal_plan")
    inventory = data.get("inventory", {})

    if not meal_plan:
        return jsonify({"error": "Missing meal_plan in request"}), 400

    categorized, _ = clean_and_categorize_ingredients(meal_plan)
    extracted = extract_quantities(meal_plan, categorized)
    flattened = flatten_quantities(extracted)
    if (inventory != {}):
        final_list = subtract_inventory(flattened, inventory)
    else:
        final_list = flattened
    return jsonify({
        "categorized": categorized,
        "shopping_list": final_list
    })

@app.route('/custom_meal_plan', methods=['POST'])
def get_custom_meal_plan():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    preferences = request.get_json()
    
    # Valider les préférences
    required_fields = ["allergy", "diet", "goal", "number_of_meals", "grocery_day"]
    for field in required_fields:
        if field not in preferences:
            return jsonify({"error": "Missing required field: {field}"}), 400
    
    if not isinstance(preferences["allergy"], dict):
        return jsonify({"error": "Allergy must be a dictionary"}), 400
    
    if preferences["diet"].lower() not in ["végan", "végétarien", "none"]:
        return jsonify({"error": "Unsupported diet"}), 400
    
    if preferences["goal"].lower() not in ["lose weight", "maintain", "gain weight"]:
        return jsonify({"error": "Unsupported goal"}), 400
    
    if not isinstance(preferences["number_of_meals"], int) or preferences["number_of_meals"] < 1:
        return jsonify({"error": "Invalid number_of_meals"}), 400
    
    if "max_calories" in preferences:
        if not isinstance(preferences["max_calories"], (int, float)) or preferences["max_calories"] <= 0:
            return jsonify({"error": "Invalid max_calories"}), 400
    
    # Générer le plan de repas
    meal_plan = generate_meal_plan(preferences)
    
    if "error" in meal_plan:
        return jsonify(meal_plan), 400
    
    return jsonify(meal_plan)

@app.route('/optimized_meal_plan', methods=['POST'])
def get_optimized_meal_plan():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    
    # Valider les champs requis
    required_fields = ["inventory_id", "user_id", "grocery", "fresh_produce"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    if not isinstance(data["inventory_id"], int):
        return jsonify({"error": "inventory_id must be an integer"}), 400
    
    if not isinstance(data["user_id"], int):
        return jsonify({"error": "user_id must be an integer"}), 400
    
    if not isinstance(data["grocery"], list):
        return jsonify({"error": "grocery must be a list"}), 400
    
    if not isinstance(data["fresh_produce"], list):
        return jsonify({"error": "fresh_produce must be a list"}), 400
    
    # Extraire les ingrédients de grocery et fresh_produce
    inventory_ingredients = []
    for item in data["grocery"] + data["fresh_produce"]:
        if not isinstance(item, dict) or "name" not in item:
            return jsonify({"error": "Each grocery or fresh_produce item must be a dictionary with a 'name' field"}), 400
        inventory_ingredients.append(item["name"])
    
    if not inventory_ingredients:
        return jsonify({"error": "No ingredients provided in grocery or fresh_produce"}), 400
    
    print(f"Ingrédients de l'inventaire : {inventory_ingredients}")
    
    # Générer le plan de repas avec les ingrédients de l'inventaire
    meal_plan = generate_meal_plan(inventory_ingredients=inventory_ingredients)
    
    if "error" in meal_plan:
        return jsonify(meal_plan), 400
    
    return jsonify(meal_plan)

# Route POST pour un plan de repas optimisé avec préférences
@app.route('/optimized_preferences_meal_plan', methods=['POST'])
def get_optimized_preferences_meal_plan():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400    
    
    data = request.get_json()

    preferences = data.get("preferences")
    inventory = data.get("inventory", {})

    # Valider les champs requis
    required_fields = ["inventory_id", "user_id", "grocery", "fresh_produce"]
    for field in required_fields:
        if field not in inventory:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    if not isinstance(inventory["inventory_id"], int):
        return jsonify({"error": "inventory_id must be an integer"}), 400
    
    if not isinstance(inventory["user_id"], int):
        return jsonify({"error": "user_id must be an integer"}), 400
    
    if not isinstance(inventory["grocery"], list):
        return jsonify({"error": "grocery must be a list"}), 400
    
    if not isinstance(inventory["fresh_produce"], list):
        return jsonify({"error": "fresh_produce must be a list"}), 400
    
    # Extraire les ingrédients de grocery et fresh_produce
    inventory_ingredients = []
    for item in inventory["grocery"] + inventory["fresh_produce"]:
        if not isinstance(item, dict) or "name" not in item:
            return jsonify({"error": "Each grocery or fresh_produce item must be a dictionary with a 'name' field"}), 400
        inventory_ingredients.append(item["name"])
    
    if not inventory_ingredients:
        return jsonify({"error": "No ingredients provided in grocery or fresh_produce"}), 400
    
    print(f"Ingrédients de l'inventaire : {inventory_ingredients}")
    
    # Récupérer les préférences via l'API
    # user_id = data["user_id"]
    # api_url = f"https://smartmeal-backend.onrender.com/preferences/id?user_id={user_id}"
    # try:
    #     response = requests.get(api_url, timeout=5)
    #     response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP
    #     preferences = response.json()
    #     print(f"Préférences récupérées pour user_id {user_id}: {preferences}")
    # except requests.exceptions.RequestException as e:
    #     return jsonify({"error": f"Failed to fetch preferences: {str(e)}"}), 500
    
    # Valider les préférences
    required_fields = ["allergy", "diet", "goal", "number_of_meals", "grocery_day"]
    for field in required_fields:
        if field not in preferences:
            return jsonify({"error": f"Missing required preference field: {field}"}), 400
    
    if not isinstance(preferences["allergy"], dict):
        return jsonify({"error": "Allergy must be a dictionary"}), 400
    
    if preferences["diet"].lower() not in ["végan", "végétarien", "none"]:
        return jsonify({"error": "Unsupported diet"}), 400
    
    if preferences["goal"].lower() not in ["lose weight", "maintain", "gain weight"]:
        return jsonify({"error": "Unsupported goal"}), 400
    
    if not isinstance(preferences["number_of_meals"], int) or preferences["number_of_meals"] < 1:
        return jsonify({"error": "Invalid number_of_meals"}), 400
    
    if "max_calories" in preferences:
        if not isinstance(preferences["max_calories"], (int, float)) or preferences["max_calories"] <= 0:
            return jsonify({"error": "Invalid max_calories"}), 400
    
    # Générer le plan de repas avec les préférences et les ingrédients
    meal_plan = generate_meal_plan(preferences=preferences, inventory_ingredients=inventory_ingredients)
    
    if "error" in meal_plan:
        return jsonify(meal_plan), 400
    
    return jsonify(meal_plan)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)