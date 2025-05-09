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
    df = pd.read_pickle("recipes_fix.pkl")
    with open("label_fix.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("scaler_fix.pkl", "rb") as f:
        scaler = pickle.load(f)
    model = tf.keras.models.load_model("meal_plan_model_fix.keras")
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
                        "preparation": recette["instructions"],
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

# def clean_and_categorize_ingredients(meal_plan):
#     ingredients_list = set()

#     for day, meals in meal_plan.items():
#         for meal in meals:
#             ner_items = meal.get("NER", [])
#             ingredients = meal.get("ingredients", [])

#             ingredients_text = " ".join(ingredients).lower()

#             for ner in ner_items:
#                 if ner.lower() in ingredients_text:
#                     ingredients_list.add(ner)
#     # Remove invalid entries (errors, empty strings, etc.)
#     cleaned = [
#         ing for ing in ingredients_list 
#         if isinstance(ing, str) 
#         and not ing.startswith(('Error', 'N', 'O', 'Votre choix'))
#     ]
    
#     # Define category keywords (case-insensitive)
#     categories = {
#         "oils_fats": [
#             "huile", "huile d'olive", "huile de maïs", "huile de sésame", "huile végétale", 
#             "lard", "margarine", "mayonnaise", "shortening végétal"
#         ],
#         "fruits_vegetables": [
#             "abricots", "ail", "ananas", "artichauts", "artichauts gelé", "aubergine", 
#             "banane", "bananes", "bleuets", "bleuets frais", "brocoli", "brocoli frais", 
#             "brocoli gelé", "carotte", "carotte râpée", "carottes", "céleri", "cerise noire", 
#             "cerises", "champignon", "champignons", "champignons frais", "chou", "chou rouge", 
#             "chou vert", "chou-fleur", "citron", "citron vert", "citrons", "citrouille", 
#             "concombre", "concombres", "courge jaune", "courgettes", "dates", "échalotes", 
#             "épinard", "fraises", "fraises fraîches", "framboise", "framboises", "gingembre frais", 
#             "haricots verts", "haricots verts gelés", "laitue romaine", "légumes", "légumes mélangés", 
#             "légumes mélangés surgelés", "légumes verts", "mangue", "navets", "oignon", 
#             "oignon blanc", "oignon frais", "oignon rouge", "oignon vert", "olive", 
#             "olives", "orange", "oranges", "pomme", "pomme de terre", "pommes", "pommes de terre", 
#             "pommes de terre rouges", "pommes fraîches", "pommes vertes", "potiron", "pruneaux", 
#             "raisin", "raisins", "raisins secs", "rhubarbe", "tomate", "tomate fraîche", "tomates"
#         ],
#         "meat_seafood": [
#             "agneau", "anchois", "bacon", "bœuf", "bœuf haché maigre", "crabe", "crevette", 
#             "dinde", "escalopes de poulet", "filet de poisson", "filet de porc", "jambon", 
#             "morceau de poulet blanc", "palourdes", "poisson blanc ferme", "poitrine de boeuf", 
#             "poitrines de poulet", "porc", "poulet", "saucisse", "saumon", "thon", "viande hachée", "steak"
#         ],
#         "dairy_eggs": [
#             "babeurre", "beurre", "beurre non salé", "blanc d'oeuf", "blancs d'œufs", 
#             "crème", "crème condensée", "crème fouettée", "crème fraîche", "crème légère", 
#             "crème sure commerciale", "fromage", "fromage blanc", "fromage cheddar", 
#             "fromage à la crème", "jaune d'oeuf", "jaunes d'oeuf", "lait", "lait condensé", 
#             "lait de coco", "lait en poudre", "œuf", "Œufs", "yaourt", "yaourt nature non gras"
#         ],
#         "herbs_spices": [
#             "aneth", "aneth frais", "anis", "basilic", "basilic doux", "cannelle", "cardamome", 
#             "cayenne", "ciboulette", "coriandre", "cumin", "curcuma", "curry", "estragon", 
#             "feuille de laurier", "gingembre en poudre", "marjolaine", "origan", "paprika", 
#             "persil", "persil frais", "piment", "poivre", "poivre blanc", "poivre de Cayenne", 
#             "romarin", "safran", "sel", "thym", "vanille"
#         ],
#         "grains_cereals": [
#             "Arborio", "avoine", "avoine de cuisson", "avoine roulée", "biscuits", "chapelure", 
#             "corn flakes", "farine", "farine de blé entier", "farine de maïs", "gruau", 
#             "macaroni", "nouilles", "nouilles aux œufs", "orge perlée", "pâtes", "riz", 
#             "riz brun", "riz sauvage", "spaghetti", "pain"
#         ],
#         "legumes_nuts": [
#             "amande", "amandes", "arachide", "arachides", "haricots", "haricots blancs", 
#             "haricots de Lima", "haricots pinto", "haricots rouges", "haricots verts", 
#             "lentilles", "noix", "noix de cajou", "noix de coco", "noix de pécan", "pacanes", 
#             "petits pois", "pois verts", "soja"
#         ],
#         "baking_sweets": [
#             "bicarbonate de soude", "cacao", "cacao en poudre", "chocolat", "chocolat au lait", 
#             "chocolat non sucré", "confiture", "gâteau", "gâteau au chocolat", "gélatine", 
#             "guimauves", "miel", "sucre", "sucre en poudre", "vanille", "mélasse"
#         ],
#         "condiments_sauces": [
#             "barbecue sauce", "ketchup", "moutarde", "sauce", "sauce au poivre", 
#             "sauce tomate", "sauce worcestershire", "vinaigre", "vinaigre de cidre"
#         ],
#         "beverages": [
#             "bière", "café", "coca-cola", "jus d'ananas", "jus de citron", "jus de tomate", 
#             "thé", "vin blanc", "vin rouge"
#         ],
#         "canned_packaged": [
#             "bouillon de bœuf", "bouillon de poulet", "concentré de jus d'orange", 
#             "conserve d'abricot", "cornichons", "purée de tomates", "soupe aux champignons"
#         ],
#         "miscellaneous": [
#             "colorant alimentaire", "extrait de vanille", "levure", "poudre à pâte", 
#             "sel de céleri", "semoule"
#         ]
#     }
        
#     # Categorize ingredients
#     categorized = {key: [] for key in categories}
#     uncategorized = []
    
#     for ingredient in cleaned:
#         lower_ing = ingredient.lower()
#         found = False
        
#         for category, keywords in categories.items():
#             if any(keyword in lower_ing for keyword in keywords):
#                 categorized[category].append(ingredient)
#                 found = True
#                 break
                
#         if not found:
#             uncategorized.append(ingredient)
    
#     # Sort each category alphabetically
#     for category in categorized:
#         categorized[category] = sorted(categorized[category], key=lambda x: x.lower())
    
#     return categorized, uncategorized



# def parse_quantity_min_1(qty_str):
#     try:
#         qty_str = qty_str.strip()
#         if re.match(r'^\d+ \d+/\d+$', qty_str):  # e.g., "2 1/4"
#             parts = qty_str.split()
#             value = float(parts[0]) + float(Fraction(parts[1]))
#         elif re.match(r'^\d+/\d+$', qty_str):  # e.g., "1/2"
#             value = float(Fraction(qty_str))
#         else:
#             match = re.match(r'^([\d.]+)', qty_str)
#             if match:
#                 value = float(match.group(1))
#             else:
#                 return 0
#         return max(1, int(round(value)))  # Ensure minimum 1
#     except:
#         return 0

# def extract_quantities(meal_plan, categorized):
#     quantities = {
#         "meat_seafood": defaultdict(list),
#         "dairy_eggs": defaultdict(list),
#         "fruits_vegetables": defaultdict(list)
#     }

#     for day, meals in meal_plan.items():
#         for meal in meals:
#             ingredient_lines = meal.get("ingredients", [])

#             for line in ingredient_lines:
#                 line_lower = line.lower()

#                 for category in quantities:
#                     for item in categorized.get(category, []):
#                         if item.lower() in line_lower:
#                             # Try to extract quantity with unit first
#                             match = re.search(r"(\d+(?:[.,]\d+)?)(?:\s*)(pt|lb|kg|g|ml|l|cuillères?|tsp|oz|pkg|tranches?)?", line_lower, re.IGNORECASE)
#                             if match:
#                                 number_str = match.group(1).replace(',', '.')
#                                 unit = match.group(2).lower() if match.group(2) else "count"  # default to "count" if no unit
#                                 try:
#                                     number_val = float(number_str)
#                                     quantities[category][item].append((number_val, unit))
#                                 except ValueError:
#                                     pass  # skip invalid number
#                             else:
#                                 # No quantity found at all — fallback to default count = 1
#                                 quantities[category][item].append((1.0, "count"))

#     return quantities

# def flatten_quantities(quantities):
#     flattened = {}

#     for category, items in quantities.items():
#         flattened[category] = {}

#         for item, qty_list in items.items():
#             if category == "meat_seafood":
#                 total_grams = 0
#                 total_packs = 0

#                 for value, unit in qty_list:
#                     unit = unit.lower()
#                     if unit in ["g", "gram", "grams"]:
#                         total_grams += value
#                     elif unit in ["kg", "kilogram", "kilograms"]:
#                         total_grams += value * 1000
#                     elif unit in ["lb", "lbs", "pound", "pounds"]:
#                         total_grams += value * 453.592
#                     else:
#                         # Treat as pack (pkg, tranches, etc.)
#                         total_packs += value

#                 result = {}
#                 if total_grams > 0:
#                     result["grams"] = int(round(total_grams))
#                 if total_packs > 0:
#                     result["packs"] = int(round(total_packs))

#                 flattened[category][item] = result

#             else:
#                 # For other categories, just sum values regardless of unit
#                 total = defaultdict(float)
#                 for value, unit in qty_list:
#                     total[unit] += value
#                 flattened[category][item] = {
#                     unit: int(round(amount)) for unit, amount in total.items()
#                 }

#     return flattened


# def normalize_name(name):
#     return unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode().lower().strip()

# def subtract_inventory(shopping_list, inventory):
#     # Flatten inventory into normalized name → (quantity, unit)
#     inv_map = {}

#     for item in inventory.get("grocery", []) + inventory.get("fresh_produce", []):
#         name = normalize_name(item["name"])
#         quantity = float(item["quantity"].replace(',', '.'))
#         unit = item["type_quantity"].lower()
#         inv_map[name] = (quantity, unit)

#     # Adjust the shopping list
#     for category, items in shopping_list.items():
#         for item, data in list(items.items()):  # Use list() to allow removal
#             norm_item = normalize_name(item)

#             if norm_item in inv_map:
#                 inv_qty, inv_unit = inv_map[norm_item]

#                 # Meat/seafood special case
#                 if category == "meat_seafood":
#                     if "grams" in data and inv_unit in ["g", "gram", "grams"]:
#                         data["grams"] = max(0, data["grams"] - inv_qty)
#                         if data["grams"] == 0:
#                             del data["grams"]
#                     if "packs" in data and inv_unit in ["pack", "packs", "pkg"]:
#                         data["packs"] = max(0, data["packs"] - inv_qty)
#                         if data["packs"] == 0:
#                             del data["packs"]

#                 else:
#                     for unit in list(data.keys()):
#                         if normalize_name(unit) == normalize_name(inv_unit) or (unit == "count" and inv_unit == ""):
#                             data[unit] = max(0, data[unit] - inv_qty)
#                             if data[unit] == 0:
#                                 del data[unit]

#             # Remove item if it's now empty
#             if not data:
#                 del shopping_list[category][item]

#     return shopping_list

# @app.route('/shopping_list', methods=['POST'])
# def get_shopping_list():
#     if not request.is_json:
#         return jsonify({"error": "Request must be JSON"}), 400

#     data = request.get_json()

#     meal_plan = data.get("meal_plan")
#     inventory = data.get("inventory", {})

#     if not meal_plan:
#         return jsonify({"error": "Missing meal_plan in request"}), 400

#     categorized, _ = clean_and_categorize_ingredients(meal_plan)
#     extracted = extract_quantities(meal_plan, categorized)
#     flattened = flatten_quantities(extracted)
#     if (inventory != {}):
#         final_list = subtract_inventory(flattened, inventory)
#     else:
#         final_list = flattened
#     return jsonify({
#         "categorized": categorized,
#         "shopping_list": final_list
#     })

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
            return jsonify({"error": "Missing required field: {field}"}), 400
    
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



CATEGORY_RULES = {
    "huiles": {"unit": "l", "default": 0.5, "conversions": {"ml": 0.001}},
    "fruits": {"unit": "g", "default": 100, "conversions": {}},
    "legumes": {"unit": "g", "default": 100, "conversions": {}},
    "viande": {"unit": "g", "default": 400, "conversions": {}},
    "poisson": {"unit": "g", "default": 400, "conversions": {}},
    "lait": {"unit": "l", "default": 0.5, "conversions": {"ml": 0.001}},
    "oeuf": {"unit": "number", "default": 1, "conversions": {}},
    "herbes": {"unit": "g", "default_per_cuillere": 0.5, "conversions": {}},
    "epices": {"unit": "g", "default_per_cuillere": 0.5, "conversions": {}},
    "cereales": {"unit": "g", "default": 100, "conversions": {}},
    "noix": {"unit": "g", "default": 100, "conversions": {}},
    "patisserie": {"unit": "g", "default": 100, "conversions": {}},
    "sauce": {"unit": "ml", "default_per_cuillere": 0.25, "conversions": {}},
    "boisson": {"unit": "l", "default": 0.5, "conversions": {"ml": 0.001}},
    "autres": {"unit": "number", "default": 1, "conversions": {}}
}

categories = {
    "huiles": ["Huile", "Huile Crisco", "Huile Mazola", "Huile Wesson", "Huile d'arachide", "Huile d'olive", "Huile de cannelle", "Huile de carthame", "Huile de clou de girofle", "Huile de cuisson", "Huile de maïs", "Huile de maïs soufflé", "Huile de menthe poivrée", "Huile de salade", "Huile de sésame", "Huile végétale", "Crisco", "Crisco fondu", "Shortening", "Shortening au goût de beurre", "Shortening végétal", "Graisse", "Graisse de bacon", "Graisse de viande fondue", "Saindoux", "Matière grasse", "Matière grasse aromatisée au beurre", "Matière grasse solide", "Matière grasse végétale"],
    "fruits": ["Abricots", "Ananas", "Ananas en dés", "Banane", "Bananes", "Cantaloup", "Cerise au marasquin", "Cerises", "Cerises au marasquin", "Cerises confites", "Cerises marasquin", "Cerises noires dénoyautées", "Citron", "Citron vert", "Citrons", "Citrouille", "Compote de pommes", "Concentré d'ananas", "Concentré de jus d'orange", "Concentré de jus de pomme", "Coquilles", "Dattes", "Fraises", "Fraises fraîches", "Fraises nettoyées", "Fraises surgelées", "Framboise rouge", "Framboises", "Framboises congelées", "Fruit", "Fruits", "Fruits confits", "Fruits confits mélangés", "Gelée de cerise", "Gelée de fraise", "Gelée de fraise Jell-O", "Gelée de groseille", "Gelée de pomme", "Gelée de raisin", "Grenadine", "Jus d'ananas", "Jus d'orange", "Jus de canneberge", "Jus de cerise", "Jus de citron", "Jus de citron vert", "Jus de framboise", "Jus de mandarine", "Jus de pomme", "Jus de pruneau", "Jus de pêche", "Jus de raisin blanc", "Kakis", "Mandarines", "Mangue", "Moitiés de poires", "Morceaux d'ananas", "Mûres fraîches", "Nectar d'abricot", "Orange", "Oranges", "Oranges mandarines", "Pamplemousse", "Pêche", "Pêches", "Pêches tranchées", "Pomme", "Pomme Granny Smith", "Pomme Red Delicious", "Pommes", "Pommes fraîches", "Pommes jaunes", "Pommes non pelées", "Pommes pelées", "Pommes rouges", "Pommes vertes", "Pommes à cuire", "Pruneaux", "Pulpe de banane", "Pulpe de kaki", "Quartiers de citron", "Raisins", "Raisins blancs", "Raisins secs", "Raisins verts", "Raisins violets", "Rhubarbe", "Segments d'orange mandarine", "Segments de mandarine", "Tranches d'ananas"],
    "legumes": ["Ail", "Artichauts", "Aubergine", "Betteraves", "Brocoli", "Brocoli frais", "Brocoli surgelé", "Brocolis", "Carotte", "Carotte râpée", "Carottes", "Carottes pour bébé", "Céleri", "Champignons", "Champignons frais", "Chou", "Chou rouge", "Chou vert", "Chou-fleur", "Choucroute", "Châtaignes d'eau", "Concombre", "Concombres", "Coquilles Creamette", "Coquilles moyennes", "Courge", "Courge d'été", "Courge jaune", "Courge jaune d'été", "Courges jaunes", "Courgette", "Courgettes", "Courgettes râpées", "Cosses de pois", "Cosses de pois surgelées", "Épinards", "Épinards hachés", "Épinards surgelés", "Épinards à la crème", "épinards congelés", "Fenouil séché", "Feuilles de navet", "Fleurettes de brocoli", "Fleurons de brocoli", "Germes de soja", "Haricots", "Haricots B & M", "Haricots Northern", "Haricots au porc", "Haricots beurre", "Haricots blancs", "Haricots cuits", "Haricots de Lima", "Haricots de Lima jaunes", "Haricots de Lima verts", "Haricots en sauce tomate", "Haricots et porc", "Haricots frits", "Haricots jaunes", "Haricots navy", "Haricots pinto", "Haricots rouges", "Haricots rouges Ranch Style", "Haricots verts", "Haricots verts surgelés", "Laitue", "Laitue romaine", "Laitue râpée", "Légumes", "Légumes Veg-All", "Légumes chop suey", "Légumes mélangés", "Légumes mélangés Veg-All", "Légumes mélangés congelés", "Légumes mélangés surgelés", "Légumes verts", "Maïs", "Maïs congelé", "Maïs doré", "Maïs en crème", "Maïs en grains", "Maïs en grains entiers", "Maïs entier", "Maïs hominy jaune", "Maïs jaune", "Maïs surgelé", "Maïs à la crème", "Navets", "Oignon", "Oignon blanc", "Oignon frais", "Oignon frais émincé", "Oignon rouge", "Oignon vert", "Oignon violet", "Oignons", "Oignons doux", "Oignons frits", "Oignons jaunes", "Oignons verts", "Okra", "Olive noires", "Olives", "Olives noires", "Onion", "Patates douces", "Petits pois", "Petits pois anglais", "Petits pois congelés", "Petits pois doux", "Petits pois verts", "Petits pois verts congelés", "Pieds de champignons", "Piment", "Piment de Cayenne", "Piment de Cayenne rouge", "Piment de la Jamaïque", "Piment fort", "Piment jalapeño", "Piment rouge", "Piment vert", "Piments", "Piments doux", "Piments forts", "Piments jalapeño", "Piments jalapeños", "Piments rouges", "Piments verts", "Poireaux", "Pois", "Pois aux yeux noirs", "Pois cassés verts", "Pois chiches", "Pois de campagne", "Pois mange-tout", "Pois à vache", "Poivron", "Poivron rouge", "Poivron vert", "Poivrons", "Poivrons rouges", "Poivrons rouges doux", "Poivrons verts", "Poivrons verts doux", "Pomme de terre", "Pommes de terre", "Pommes de terre O'Brien", "Pommes de terre blanches", "Pommes de terre irlandaises", "Pommes de terre nouvelles", "Pommes de terre rissolées", "Pommes de terre rissolées congelées", "Pommes de terre rouges", "Pousses de bambou", "Radis", "Tomate", "Tomate fraîche", "Tomates", "Tomates Ro-Tel", "Tomates assaisonnées à l'italienne", "Tomates italiennes", "Tomates à l'italienne", "Tomates étuvées", "Tiges de céleri", "Échalotes"],
    "viande": ["Agneau", "Bacon", "Bacon canadien", "Beefogetti", "Bifteck", "Bifteck de flanc", "Bœuf", "Bœuf en conserve", "Bœuf fumé", "Bœuf haché", "Bœuf haché extra maigre", "Bœuf haché faible en gras", "Bœuf haché maigre", "Bœuf haché rond", "Bœuf maigre haché", "Bœuf salé", "Bœuf séché", "Bœuf à ragoût", "Cubes de bœuf chuck", "Corned-beef", "Côtelettes de porc", "Demi-Poitrine Poulet", "Dinde", "Dinde hachée", "Escalopes de poulet", "Filet de porc", "Filets de porc", "Hachis de bœuf salé", "Hachis de corned-beef", "Hamburger", "Hamburger maigre", "Hot-dogs", "Jambon", "Jambon cuit", "Jambon fumé", "Jambon râpé", "Kielbasa", "Morceaux de Poulet", "Morceaux de bacon", "Morceaux de poulet", "Os de jambon", "Os à soupe charnus", "Pepperoni", "Pointe de poitrine", "Poitrine de bœuf", "Poitrine de dinde", "Poitrine de poulet", "Poitrines de Poulet", "Poitrines de poulet", "Poitrines de poulet désossées", "Porc", "Porc et haricots", "Porc haché", "Porc salé", "Poule", "Poulet", "Poulet blanc", "Rôti de Bœuf", "Rôti de palette", "Rôti de porc", "Salami", "Saucisse", "Saucisse d'été", "Saucisse de porc", "Saucisse de porc hachée", "Saucisse douce", "Saucisse fumée", "Saucisse piquante", "Saucisse épicée", "Saucisses de Francfort", "Saucisses italiennes douces", "Steak", "Steaks", "Viande de bœuf en conserve", "Viande de poulet blanc", "Viande hachée", "Viande à ragoût"],
    "poisson": ["Aiglefin", "Chair de crabe", "Chair de crabe imitée", "Crabe", "Crevettes", "Fausse chair de crabe", "Filet de Colin", "Filet de poisson", "Flet", "Huîtres", "Liquide d'huîtres", "Mulet", "Palourdes", "Poisson blanc ferme", "Pétoncles", "Queues d'écrevisses", "Saumon", "Saumon rose", "Thon", "Têtes de poisson"],
    "lait": ["Babeurre", "Babeurre sans gras", "Beurre", "Beurre d'arachide", "Beurre de cacahuète", "Beurre de cacahuète croquant", "Beurre de cacahuète lisse", "Beurre non salé", "Crème", "Crème Carnation", "Crème aigre", "Crème de champignons", "Crème de céleri", "Crème de guimauve", "Crème de menthe", "Crème de tartre", "Crème en poudre non laitière", "Crème fouettée", "Crème glacée", "Crème liquide", "Crème légère", "Crème moitié-moitié", "Crème sure", "Crème à fouetter", "Crème épaisse", "Fromage", "Fromage Cheddar", "Fromage Feta", "Fromage Monterey Jack", "Fromage Mozzarella", "Fromage Muenster", "Fromage Parmesan", "Fromage Provolone", "Fromage Ricotta", "Fromage Romano", "Fromage Suisse", "Fromage Velveeta", "Fromage américain", "Fromage au piment", "Fromage bleu", "Fromage cheddar", "Fromage cheddar allégé", "Fromage cheddar fort", "Fromage cottage", "Fromage cottage faible en gras", "Fromage fort", "Fromage lite-line", "Fromage mozzarella", "Fromage parmesan", "Fromage ricotta", "Fromage romano", "Fromage râpé", "Fromage suisse", "Fromage à l'ail", "Fromage à la crème", "Fromage à la crème Philadelphia", "Fromage à pizza", "Lait", "Lait Borden", "Lait Eagle Brand", "Lait Pet", "Lait aigre", "Lait chaud", "Lait concentré", "Lait concentré sucré", "Lait de coco", "Lait doux", "Lait en poudre", "Lait entier", "Lait faible en gras", "Lait froid", "Lait sucré", "Lait tiède", "Lait écrémé", "Lait écrémé évaporé", "Lait évaporé", "Margarine", "Margarine Parkay", "Milnot", "Mozzarella", "Parmesan", "Velveeta", "Yaourt", "Yaourt au citron", "Yaourt aux fruits", "Yaourt nature sans gras", "Yaourt sans gras"],
    "oeuf": ["Blanc d'œuf", "Blancs d'œufs", "Jaune d'œuf", "Jaunes d'œufs", "Oeuf", "Œuf", "Œufs", "Substitut d'œuf"],
    "herbes": ["Aneth", "Aneth frais", "Basilic", "Basilic doux", "Basilic estragon", "Ciboulette", "Coriandre", "Estragon", "Feuille de laurier", "Feuilles de laurier", "Marjolaine", "Origan", "Persil", "Persil frais", "Persil vert", "Romarin", "Sauge", "Thym"],
    "epices": ["Accent", "Ail", "Anis", "Anisette", "Assaisonnement Salad Supreme", "Assaisonnement aromatisé barbecue", "Assaisonnement italien", "Assaisonnement pour chili", "Assaisonnement pour pizza", "Assaisonnement pour salade", "Assaisonnement pour steak", "Assaisonnement pour tacos", "Assaisonnement pour volaille", "Arôme d'amande", "Arôme d'orange", "Arôme de citron", "Arôme de courge musquée", "Arôme de noix de beurre", "Arôme de noix de coco", "Arôme de rhum", "Arôme de vanille", "Beau Monde", "Bitters", "Cacao", "Cacao en poudre", "Cannelle", "Cannelle moulue", "Cardamome", "Cayenne", "Clous de girofle", "Clous de girofle moulus", "Colorant", "Colorant achiote", "Colorant alimentaire", "Colorant alimentaire jaune", "Colorant alimentaire rouge", "Cumin", "Cumin moulu", "Curcuma", "Curry", "Extrait d'amande", "Extrait d'orange", "Extrait de citron", "Extrait de rhum", "Extrait de vanille", "Fenouil séché", "Fumée liquide", "Gingembre", "Gingembre en poudre", "Gingembre frais", "Gingembre moulu", "Gingembre râpé", "Gousse d'ail", "Gousses d'ail", "Graine de moutarde", "Graines de carvi", "Graines de cumin", "Graines de céleri", "Graines de lin", "Graines de pavot", "Graines de sésame", "Graines de sésame blanches", "Graines de tournesol", "Grains de poivre", "Mélange Hidden Valley Ranch", "Mélange Shake 'n Bake", "Mélange barbecue Shake 'N Bake", "Mélange d'assaisonnement pour tacos", "Mélange d'oignons", "Mélange pour sauce brune", "Mélange pour sauce stroganoff", "Mélange pour vinaigrette", "Mélange à vinaigrette", "Moutarde", "Moutarde de Dijon", "Moutarde sèche", "Moutarde séchée", "Moutarde à gros grains", "Muscade", "Muscade moulue", "Old Bay", "Paprika", "Paprika espagnol", "Piment de Cayenne", "Piment de Cayenne rouge", "Piment de la Jamaïque", "Piment de la Jamaïque moulu", "Poivre", "Poivre blanc", "Poivre blanc moulu", "Poivre citronné", "Poivre de Cayenne", "Poivre moulu", "Poivre noir", "Poivre noir moulu", "Poivre rouge moulu", "Poudre d'ail", "Poudre d'oignon", "Poudre de chili", "Poudre de curry", "Poudre de céleri", "Poudre de zeste de citron", "Quatre-épices", "Raifort", "Safran", "Sel", "Sel assaisonné", "Sel d'ail", "Sel d'oignon", "Sel de céleri", "Sel gemme", "Sel gros", "Sel à l'ail", "Tabasco", "Vanille", "Vinaigre", "Vinaigre blanc", "Vinaigre de cidre", "Vinaigre de vin", "Vinaigre de vin rouge", "Vinaigre noir", "Vinaigre rouge", "Worcestershire", "Zeste de citron", "Épice du Moyen-Orient", "Épice jerk", "Épices pour tarte aux pommes", "Épices pour tarte à la citrouille", "Épices à marinade", "Épices à marinades"],
    "cereales": ["Avoine", "Avoine à cuisson rapide", "Biscuits", "Biscuits Graham", "Biscuits Oreo", "Biscuits Ritz", "Biscuits au babeurre", "Biscuits au beurre", "Biscuits de riz soufflé", "Biscuits feuilletés", "Biscuits froids", "Biscuits graham", "Biscuits salés", "Biscuits sandwich au chocolat", "Biscuits soda", "Bisquick", "Bretzels", "Brioches surgelées", "Bâtonnets de pain", "Chapelure", "Chapelure italienne", "Cheerios", "Corn Chex", "Corn flakes", "Cornflakes", "Craquelins Graham", "Craquelins de seigle", "Crackers", "Crackers Ritz", "Croissants", "Croissants en pâte", "Croûte de biscuits Graham", "Croûte de tarte", "Croûte de tarte aux biscuits Graham", "Croûte de tarte en biscuits Graham", "Croûte à pizza", "Croûte à tarte", "Croûte à tarte crue", "Croûtes de biscuits Graham", "Croûtes à tarte", "Croûtes à tarte en biscuits Graham", "Croûtons", "Croûtons assaisonnés", "Farine", "Farine White Lily", "Farine autolevante", "Farine de blé complet", "Farine de maïs", "Farine tout usage", "Farine à gâteau", "Farine à lever", "Farine à pain", "Flocons d'avoine", "Flocons de maïs", "Flocons de noix de coco", "Germe de blé grillé", "Mélange Bisquick", "Mélange Gâteau au chocolat", "Mélange Gâteau blanc", "Mélange de farce", "Mélange de farce assaisonné aux herbes", "Mélange de farce aux herbes", "Mélange pour biscuits au babeurre", "Mélange pour pain de maïs", "Mélange à biscuits", "Mélange à farce", "Mélange à gâteau", "Mélange à gâteau au beurre", "Mélange à gâteau au chocolat", "Mélange à gâteau blanc", "Mélange à gâteau jaune", "Mélange à gâteau suprême au citron", "Mélange à gâteau à la fraise", "Mélange à muffins de maïs", "Mélange à pain de maïs", "Miettes de biscuits Graham", "Miettes de biscuits Ritz", "Miettes de biscuits graham", "Miettes de cornflakes", "Miettes de crackers", "Miettes de craquelins", "Miettes de pain", "Miettes de pain de seigle", "Muffins anglais", "Nouilles", "Nouilles Ramen", "Nouilles aux épinards", "Nouilles aux œufs", "Nouilles chinoises", "Nouilles chow mein", "Nouilles larges", "Nouilles à dumplings aux œufs", "Nouilles à lasagne", "Orge perlé", "Pain", "Pain blanc", "Pain de blé complet", "Pain de maïs", "Pain de mie", "Pain de seigle", "Pain français", "Pain grillé", "Pain rassis", "Pains au levain", "Pains de levure", "Pasta", "Pâtes", "Pâtes en spirale", "Petits pains", "Petits pains réfrigérés", "Pretzels", "Rice Chex", "Rice Krispies", "Rigatoni", "Riz", "Riz Minute blanc", "Riz arborio", "Riz brun", "Riz espagnol", "Riz instantané", "Riz sauvage", "Riz à l'espagnole", "Semoule de maïs", "Spaghetti", "Spaghettis", "Spaghettis fins", "Tortilla de maïs", "Tortillas", "Tortillas de farine", "Tortillas de maïs", "Tranches de Pain", "Vermicelles", "Wheat Chex", "Wontons chinois", "Wrappers wonton"],
    "noix": ["Amande", "Amandes", "Amandes effilées", "Amandes moulues", "Arachides", "Beurre d'arachide", "Beurre de cacahuète", "Beurre de cacahuète croquant", "Beurre de cacahuète lisse", "Cacahuètes", "Noix", "Noix de cajou", "Noix de coco", "Noix de coco Angel Flake", "Noix de coco congelée", "Noix de coco en flocons", "Noix de coco râpée", "Noix de muscade", "Noix de pécan", "Noix moulues", "Noix mélangées", "Noix noires", "Pépites de caramel", "Pépites de caramel au beurre", "Pépites de chocolat", "Pépites de chocolat mi-sucré", "Pépites de chocolat sucré", "Pépites de citron", "Pépites de céréales", "Tasses de beurre de cacahuète", "Écorce d'amande"],
    "patisserie": ["Barres Heath", "Bonbons M&M", "Bonbons multicolores", "Bonbons à l'orange", "Bonbons à la menthe poivrée", "Chips de maïs", "Chips de tortilla", "Choco-bake", "Chocolat", "Chocolat au lait", "Chocolat mi-sucré", "Chocolat non sucré", "Chocolat à cuire", "Confiture d'abricot", "Confiture d'ananas", "Confiture de mûres", "Cool Whip", "Cool whip", "Doritos", "Dream Whip", "Fruits de mer", "Garniture a dessert", "Garniture au caramel", "Garniture fouetté", "Garniture fouettée", "Garniture fouettée non laitière", "Garniture pour tarte", "Garniture pour tarte aux cerises", "Garniture pour tarte aux myrtilles", "Garniture pour tarte à la pistache", "Garniture à la fraise", "Gaufrettes au chocolat", "Gaufrettes à la vanille", "Gelée de cerise", "Gelée de fraise", "Gelée de fraise Jell-O", "Gelée de groseille", "Gelée de pomme", "Gelée de raisin", "Gingembre", "Gingembre en poudre", "Gingembre frais", "Gingembre moulu", "Gingembre râpé", "Glaçage Orange-Citron", "Glaçage aux fraises", "Glaçage noix de coco et pécan", "Glaçage à la vanille", "Guimauves", "Gâteau des anges", "Gélatine", "Gélatine aromatisée", "Gélatine au citron", "Gélatine au citron vert", "Gélatine d'abricot", "Gélatine de fraise", "Gélatine non aromatisée", "Gélatine sans saveur", "Gélatine à l'orange", "Gélatine à la fraise", "Gélatine à la lime", "Gélatine à saveur d'orange", "Gélatine à saveur de cerise", "Jell-O", "Jell-O au citron", "Jell-O au citron vert", "Jell-O à la fraise", "Jell-O à la framboise", "Jello au citron", "Jello à l'orange", "Jello à la fraise", "Life Savers", "Marshmallow Fluff", "Marshmallows", "Mints Frango", "Mini-guimauves", "Mélange pouding au chocolat", "Mélange pour pudding au chocolat", "Mélange à pudding instantané à la vanille", "Mélange à pudding à la vanille", "Nourriture pour bébé aux abricots", "Oreos", "Pâte de biscuits", "Pâte de tomate", "Pâte phyllo", "Pâte sucrée de base", "Pâte à biscuits", "Pâte à biscuits au sucre", "Pâte à croissants", "Pâte à pizza réfrigérée", "Pâte à tarte", "Pouding au citron", "Pouding instantané", "Pouding instantané au chocolat", "Pouding instantané au citron", "Pouding instantané à la noix de coco", "Pouding instantané à la pistache", "Pouding instantané à la vanille", "Pouding à la vanille", "Pouding à la vanille instantané", "Pots de purée de prunes pour bébés", "Pudding au beurre écossais", "Pudding au chocolat", "Pudding instantané au chocolat", "Pudding instantané à la vanille", "Pudding à la vanille", "Relish sucrée", "Sirop", "Sirop Karo", "Sirop blanc", "Sirop de chocolat", "Sirop de grenadine", "Sirop de maïs", "Sirop de maïs blanc", "Sirop de maïs blanc Karo", "Sirop de maïs léger", "Sirop sundae au chocolat", "Sucre", "Sucre blanc", "Sucre brun", "Sucre brun clair", "Sucre glace", "Sucre granulé", "Sucre à la cannelle", "Twinkies", "bonbons Skor"],
    "sauce": ["BBQ au bœuf", "BBQ au porc", "Bac*Os", "Bouillon de bœuf", "Bouillon de jambon", "Bouillon de poulet", "Bouillon de poulet condensé", "Bouillon de poulet instantané", "Cheez Whiz", "Chicken Tonight", "Chili Hormel", "Consommé", "Consommé de bœuf", "Consommé de poulet", "Cubes de bouillon de poulet", "Farce Stove Top", "Farce au poulet", "Farce aux herbes", "Fond de tarte", "Fonds de tarte", "Fonds de tarte en biscuits Graham", "Guacamole", "Ketchup", "Manwich", "Mayonnaise", "Mayonnaise Miracle Whip", "Mayonnaise sans gras", "Miracle Whip", "Mélange de pouding instantané à la pistache", "Mélange de soupe aux légumes", "Mélange pour bouillon", "Mélange pour croûte", "Mélange pour gâteau au beurre", "Mélange pour gâteau blanc deluxe", "Mélange pour gâteau jaune", "Mélange pour muffins au maïs", "Mélange pour muffins de maïs", "Mélange pour sauce brune", "Mélange pour sauce stroganoff", "Mélange pour soupe de légumes", "Mélange pour soupe à l'oignon", "Mélange pour vinaigrette", "Mélange à soupe au poulet", "Mélange à soupe à l'oignon", "Mélange à vinaigrette", "Pâte de tomate", "Purée de prunes", "Purée de pêches", "Purée de tomates", "Ragu", "Ro-Tel", "Rotel", "Salsa", "Salsa taco", "Sauce", "Sauce Alfredo légère", "Sauce Ragu", "Sauce Tabasco", "Sauce Worcestershire", "Sauce aigre-douce", "Sauce au fromage", "Sauce au piment fort", "Sauce au piment liquide", "Sauce au piment rouge", "Sauce au poulet", "Sauce aux canneberges", "Sauce barbecue", "Sauce caramel", "Sauce chili", "Sauce enchilada", "Sauce picante", "Sauce piquante", "Sauce pour steak", "Sauce salsa", "Sauce salsa au piment vert", "Sauce soja", "Sauce taco", "Sauce tamari", "Sauce tomate", "Sauce à croquettes", "Sauce à pizza", "Sauce à spaghetti", "Sauce à spaghettis", "Sauce à la crevette", "Sauce à la crème", "Sauce à la crème d'oignon", "Sauce à la crème de champignons", "Sauce à la crème de céleri", "Sauce à la crème de poulet", "Sauce à la tomate", "Soupe au brocoli et au fromage", "Soupe au bœuf au chili", "Soupe au céleri", "Soupe au fromage", "Soupe au fromage Cheddar", "Soupe au poulet", "Soupe aux champignons", "Soupe aux légumes", "Soupe aux tomates", "Soupe d'oignon", "Soupe de haricots Campbell's", "Soupe de nouilles", "Soupe de poulet", "Soupe de tomate", "Soupe à l'oignon", "Soupe à la crevette", "Soupe à la crème", "Soupe à la crème d'oignon", "Soupe à la crème de champignons", "Soupe à la crème de céleri", "Soupe à la crème de poulet", "Soupe à la tomate", "Stove Top", "Trempette au guacamole", "Vinaigrette", "Vinaigrette Miracle Whip", "Vinaigrette Ranch", "Vinaigrette Thousand Island", "Vinaigrette italienne", "Vinaigrette pour salade", "Vinaigrette russe"],
    "boisson": ["7-Up", "Bière", "Bourbon", "Brandy", "Café", "Café instantané", "Café instantané en poudre", "Café noir", "Chablis", "Champagne", "Coca-Cola", "Cocktail de fruits", "Cocktail de jus de canneberge", "Cognac", "Country Time Lemonade", "Eau", "Eau bouillante", "Eau chaude", "Eau froide", "Eau minérale pétillante", "Eau pétillante", "Eau tiède", "Ginger ale", "Jus de tomate", "Kool-Aid", "Kool-Aid cerise", "Limonade Country Time", "Limonade congelée", "Limonade surgelée", "Mountain Dew", "Punch Sangaree", "Rhum", "Rhum Meyers", "Sherry", "Soda citron-lime", "Soda club", "Sprite", "Tang", "Thé instantané", "Triple Sec", "V8", "Vin blanc", "Vin de cuisson sherry", "Vin rouge", "Vin rouge espagnol", "Vodka", "Whisky", "Xérès"],
    "autres": ["Alun", "Bleu de lessive", "Bol en verre transparent", "Cire de paraffine", "Cube de glace", "Glace", "Glace pilée", "Glace à la vanille", "Paraffine", "Spray de cuisson", "Spray de cuisson végétal", "Spray de cuisson végétal antiadhésif", "Édulcorant"]
}

def parse_quantity(ingredient_str):
    """Parse quantity from ingredient string"""
    parts = ingredient_str.split()
    for part in parts:
        if part.replace('.', '').isdigit():
            return float(part)
    return 1

def parse_unit(ingredient_str):
    """Parse unit from ingredient string"""
    units = ["g", "kg", "ml", "l", "cuillère", "cuillères", "tasse", "tasses"]
    parts = ingredient_str.split()
    for part in parts:
        if part in units:
            if part in ["cuillère", "cuillères"]:
                return "cuillere"
            return part
    return ""

def get_category(item):
    """Find which category an item belongs to"""
    for category, items in categories.items():
        if item in items:
            return category
    return "autres"

def convert_quantity(qty, from_unit, to_unit, category):
    """Convert between units"""
    if from_unit == to_unit:
        return qty
    
    if from_unit in CATEGORY_RULES[category]["conversions"]:
        return qty * CATEGORY_RULES[category]["conversions"][from_unit]
    
    return qty
def get_standard_quantity(ingredient_str, category):
    """Get quantity in standard units for the category"""
    qty = parse_quantity(ingredient_str)
    unit = parse_unit(ingredient_str)
    
    # Handle spoon measurements first
    if category in ["herbes", "epices", "sauce"] and unit == "cuillere":
        return qty * CATEGORY_RULES[category].get("default_per_cuillere", 0.5)
    
    # Handle case where no unit is specified
    if qty == 1 and unit == "":
        return CATEGORY_RULES[category].get("default", 
               CATEGORY_RULES[category].get("default_per_cuillere", 1))
    
    # Handle unit conversions
    target_unit = CATEGORY_RULES[category]["unit"]
    return convert_quantity(qty, unit, target_unit, category)

def clean_and_categorize_ingredients(meal_plan):
    """Categorize all ingredients in the meal plan"""
    categorized = defaultdict(list)
    all_ingredients = set()
    
    for day_data in meal_plan.values():
        for meal in day_data:
            for ingredient in meal["NER"]:
                category = get_category(ingredient)
                categorized[category].append(ingredient)
                all_ingredients.add(ingredient)
    
    return dict(categorized), all_ingredients

def extract_quantities(meal_plan, categorized):
    """Extract quantities for each ingredient"""
    quantities = defaultdict(list)
    
    for day_data in meal_plan.values():
        for meal in day_data:
            for ingredient, ingredient_str in zip(meal["NER"], meal["ingredients"]):
                category = get_category(ingredient)
                std_qty = get_standard_quantity(ingredient_str, category)
                quantities[category].append({
                    "name": ingredient,
                    "quantity": std_qty,
                    "unit": CATEGORY_RULES[category]["unit"]
                })
    
    return dict(quantities)

def flatten_quantities(extracted):
    """Combine quantities for the same ingredients"""
    flattened = defaultdict(list)
    
    for category, items in extracted.items():
        item_counts = defaultdict(float)
        for item in items:
            item_counts[item["name"]] += item["quantity"]
        
        for name, qty in item_counts.items():
            flattened[category].append(f"{name}: {qty} {CATEGORY_RULES[category]['unit']}")
    
    return dict(flattened)

def subtract_inventory(flattened, inventory):
    """Subtract inventory quantities from shopping list"""
    inventory_items = {}
    for category in ["grocery", "fresh_produce"]:
        for item in inventory.get(category, []):
            inventory_items[item["name"].lower()] = float(item["quantity"])
    
    final_list = defaultdict(list)
    
    for category, items in flattened.items():
        for item_str in items:
            name, rest = item_str.split(":", 1)
            current_qty = float(rest.split()[0])
            unit = rest.split()[1]
            
            inv_qty = inventory_items.get(name.strip().lower(), 0)
            remaining_qty = max(0, current_qty - inv_qty)
            
            if remaining_qty > 0:
                final_list[category].append(f"{name}: {remaining_qty} {unit}")
    
    return dict(final_list)

@app.route('/shopping_list', methods=['POST'])
def get_shopping_list():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    meal_plan = data.get("meal_plan")
    inventory = data.get("inventory", {})

    if not meal_plan:
        return jsonify({"error": "Missing meal_plan in request"}), 400

    # Get categorized ingredients (we still need this for extract_quantities)
    categorized, _ = clean_and_categorize_ingredients(meal_plan)
    
    # Process quantities
    extracted = extract_quantities(meal_plan, categorized)  # Now passing both required arguments
    flattened = flatten_quantities(extracted)
    
    # Subtract inventory if provided
    final_list = subtract_inventory(flattened, inventory) if inventory else flattened
    
    # Clean the output by removing "number" units
    cleaned_list = {}
    for category, items in final_list.items():
        cleaned_items = []
        for item in items:
            if ": " in item:
                name, quantity = item.split(": ", 1)
                if quantity.endswith(" number"):
                    cleaned_items.append(f"{name}: {quantity[:-7]}")  # Remove " number"
                else:
                    cleaned_items.append(item)
            else:
                cleaned_items.append(item)
        cleaned_list[category] = cleaned_items
    
    return jsonify(cleaned_list)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
