import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import random
from flask import Flask, jsonify, request

app = Flask(__name__)




# Charger les données sauvegardées
try:
    df = pd.read_pickle("recipes_df.pkl")
    with open("label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    model = tf.keras.models.load_model("meal_plan_model.keras")
    print("Model and data loaded successfully")
except FileNotFoundError as e:
    print(f"Error: Missing file - {e}")
    exit(1)

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
print("Exemple d'ingrédients normalisés :", df["ingredients"].iloc[:5].tolist())

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
                "ingredients": recette["ingredients"]
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
                            "ingredients": recette["ingredients"]
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

@app.route('/shopping_list', methods=['POST'])
def get_shopping_list():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    meal_plan = request.get_json()
    shopping_list = generate_shopping_list(meal_plan)
    return jsonify(shopping_list)

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