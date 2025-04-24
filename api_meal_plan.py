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
    print("Modèle et données chargés avec succès")
except FileNotFoundError as e:
    print(f"Erreur : Fichier manquant - {e}")
    exit(1)

# Fonction pour générer un plan de repas
# Dans api_meal_plan.py
def generate_meal_plan():
    days = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi"]
    meal_plan = {}
    
    for day in days:
        num_meals = random.randint(1, 2)
        meals = []
        used_titles = set()
        for _ in range(num_meals):
            type_plat = random.choice(["entrée", "plat principal", "dessert"])
            recettes_type = df[df["type_plat"] == type_plat]
            if recettes_type.empty:
                continue
            recette = recettes_type.sample(n=1).iloc[0]
            if recette["title"] in used_titles:
                continue
            used_titles.add(recette["title"])
            
            meals.append({
                "items": [recette["title"] or "Plat sans titre"],
                "calories": int(recette["calories"]),
                "servings": int(recette["servings"]),
                "time": int(recette["time"]),
                "ingredients": recette["ingredients"]  # Ajout des ingrédients
            })
        meal_plan[day] = meals
    
    return meal_plan
# Fonction pour générer une liste de courses
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

# Routes de l'API
@app.route('/meal_plan', methods=['GET'])
def get_meal_plan():
    meal_plan = generate_meal_plan()
    return jsonify(meal_plan)

@app.route('/shopping_list', methods=['POST'])
def get_shopping_list():
    if not request.is_json:
        return jsonify({"error": "Requête doit être au format JSON"}), 400
    meal_plan = request.get_json()
    shopping_list = generate_shopping_list(meal_plan)
    return jsonify(shopping_list)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)