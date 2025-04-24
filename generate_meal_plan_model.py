import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
import random
import pickle

# Charger le fichier JSON
with open('recipes_raw_nosource_fn.json', 'r') as file:
    data = json.load(file)

# Créer une liste de recettes avec des types de plats fictifs et estimations
recipes_data = []
for recipe_id, recipe in data.items():
    title = recipe.get("title", "")
    if title is None or not isinstance(title, str):
        title = ""
    
    type_plat = "dessert"
    if title:
        if "Dip" in title or "Eggplant" in title:
            type_plat = "entrée"
        elif any(s in title for s in ["Crab", "Beans", "Noodle", "Sushi", "Fries"]):
            type_plat = "plat principal"
    
    calories = int(random.randint(200, 800))
    servings = int(random.randint(1, 4))
    time = int(random.randint(15, 60))
    
    recipes_data.append({
        "type_plat": type_plat,
        "title": title,
        "instructions": recipe.get("instructions", "Instructions manquantes"),
        "ingredients": recipe.get("ingredients", []),
        "calories": calories,
        "servings": servings,
        "time": time
    })

# Convertir en DataFrame
df = pd.DataFrame(recipes_data)

# Encoder les types de plats
encoder = LabelEncoder()
df["type_plat_encoded"] = encoder.fit_transform(df["type_plat"])

# Préparer les entrées (X) et sorties (y)
X = df["type_plat_encoded"].values.reshape(-1, 1)
y = np.arange(len(df))

# Construire le modèle
model = models.Sequential([
    layers.Input(shape=(1,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(len(df), activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Entraîner le modèle
model.fit(X, y, epochs=50, verbose=1)

# Sauvegarder le modèle
model.save("meal_plan_model.keras")
print("Modèle sauvegardé sous 'meal_plan_model.keras'")

# Sauvegarder le DataFrame et l'encoder
df.to_pickle("recipes_df.pkl")
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)
print("DataFrame et encoder sauvegardés")

# Fonction pour générer un plan de repas
def generate_meal_plan():
    days = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi"]
    meal_plan = {}
    
    for day in days:
        num_meals = random.randint(1, 2)
        meals = []
        for _ in range(num_meals):
            type_plat = random.choice(["entrée", "plat principal", "dessert"])
            try:
                type_plat_encoded = encoder.transform([type_plat])[0]
                prediction = model.predict(np.array([[type_plat_encoded]]), verbose=0)
                recette_index = np.argmax(prediction)
                recette = df.iloc[recette_index]
                
                meals.append({
                    "items": [recette["title"] or "Plat sans titre"],
                    "calories": int(recette["calories"]),
                    "servings": int(recette["servings"]),
                    "time": int(recette["time"])
                })
            except ValueError:
                continue
        meal_plan[day] = meals
    
    return meal_plan

# Fonction pour générer une liste de courses
def generate_shopping_list(meal_plan):
    ingredients_set = {}
    
    def parse_quantity(qty_str):
        """Convertir une quantité (y compris fractions) en float ou retourner une valeur par défaut."""
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

# Tester les fonctions
meal_plan = generate_meal_plan()
print(json.dumps(meal_plan, ensure_ascii=False, indent=2))

shopping_list = generate_shopping_list(meal_plan)
print(json.dumps(shopping_list, ensure_ascii=False, indent=2))