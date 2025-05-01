import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder, StandardScaler
import random
import pickle

# Charger le fichier JSON
with open('recipes_fr.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Listes de mots-clés pour classification
entree_keywords = [
    "soupe", "salade", "velouté", "crème", "maïs", "dip", "bruschetta", "tartine", 
    "terrine", "pâté", "carpaccio", "ceviche", "gazpacho", "entrée", "amuse-bouche", 
    "canapé", "tapas", "antipasto", "crudités", "rillettes", "quiche", "tarte salée"
]
plat_principal_keywords = [
    "poulet", "bœuf", "porc", "agneau", "veau", "poisson", "crevette", "saumon", 
    "thon", "pâte", "riz", "gratin", "rôti", "steak", "filet", "côte", "cuisse", 
    "lasagne", "pizza", "burger", "tajine", "curry", "wok", "plat", "ragoût", 
    "blanquette", "chili", "paella", "risotto"
]
dessert_keywords = [
    "gâteau", "tarte", "biscuit", "cookie", "brownie", "muffin", "crème", "pudding", 
    "flan", "clafoutis", "sorbet", "glace", "chocolat", "mousse", "macaron", "éclair", 
    "millefeuille", "pavlova", "trifle", "bonbon", "candy", "dessert", "sucré", 
    "crêpe", "gaufre", "beignet"
]

entree_ingredients = [
    "laitue", "tomate", "concombre", "radis", "carotte râpée", "avocat", 
    "fromage de chèvre", "anchois", "olives", "pain grillé", "croûtons", 
    "vinaigre", "moutarde", "herbes fraîches", "poivron", "artichaut", "asperge"
]
plat_principal_ingredients = [
    "poulet", "bœuf", "porc", "poisson", "crevette", "pomme de terre", "riz", 
    "pâtes", "lentilles", "haricots", "oignon", "ail", "carotte", "courgette", 
    "champignons", "sauce tomate", "crème fraîche", "fromage râpé", "curry", "paprika"
]
dessert_ingredients = [
    "sucre", "chocolat", "beurre", "farine", "œuf", "vanille", "crème", "lait", 
    "cacao", "fraise", "pomme", "banane", "noix", "amandes", "pépites de chocolat", 
    "sucre en poudre", "miel", "cannelle"
]

# Créer une liste de recettes avec des types de plats
recipes_data = []
for recipe in data:
    title = recipe.get("title", "").lower()
    if title is None or not isinstance(title, str):
        title = ""
    
    ingredients = recipe.get("ingredients", [])
    ingredients_lower = [ing.lower() for ing in ingredients]
    ner = recipe.get("NER", [])
    
    type_plat = "dessert"
    if any(keyword in title for keyword in entree_keywords):
        type_plat = "entrée"
    elif any(keyword in title for keyword in plat_principal_keywords):
        type_plat = "plat principal"
    elif any(keyword in title for keyword in dessert_keywords):
        type_plat = "dessert"
    else:
        entree_score = sum(1 for ing in ingredients_lower for ent_ing in entree_ingredients if ent_ing in ing)
        plat_score = sum(1 for ing in ingredients_lower for plat_ing in plat_principal_ingredients if plat_ing in ing)
        dessert_score = sum(1 for ing in ingredients_lower for dessert_ing in dessert_ingredients if dessert_ing in ing)
        
        max_score = max(entree_score, plat_score, dessert_score)
        if max_score > 0:
            if entree_score == max_score:
                type_plat = "entrée"
            elif plat_score == max_score:
                type_plat = "plat principal"
            else:
                type_plat = "dessert"
    
    calories = int(random.randint(200, 800))
    servings = int(random.randint(1, 4))
    time = int(random.randint(15, 60))
    
    recipes_data.append({
        "type_plat": type_plat,
        "title": recipe.get("title", ""),
        "instructions": " ".join(recipe.get("directions", ["Instructions manquantes"])),
        "ingredients": ingredients,
        "NER": ner,
        "calories": calories,
        "servings": servings,
        "time": time
    })

# Convertir en DataFrame
df = pd.DataFrame(recipes_data)

# Convertir les listes NER en chaînes pour gérer les doublons, en excluant None
df["NER_str"] = df["NER"].apply(lambda x: ",".join(sorted([item for item in x if item is not None])) if isinstance(x, list) else x)
# Supprimer les doublons en utilisant title et NER_str
df = df.drop_duplicates(subset=["title", "NER_str"])
# Supprimer la colonne temporaire
df = df.drop(columns=["NER_str"])

# Vérifier la répartition des types de plats
print("Répartition des types de plats :")
print(df["type_plat"].value_counts())

# Encoder les types de plats
encoder = LabelEncoder()
df["type_plat_encoded"] = encoder.fit_transform(df["type_plat"])

# Préparer les entrées avec plus de caractéristiques
X = df[["type_plat_encoded", "calories", "time"]].values
y = np.arange(len(df))

# Normaliser les caractéristiques
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Construire le modèle
model = models.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(df), activation="softmax")
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss="sparse_categorical_crossentropy", 
              metrics=["accuracy"])

# Entraîner le modèle
model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Sauvegarder le modèle
model.save("meal_plan_model_v2.keras")
print("Modèle sauvegardé sous 'meal_plan_model_v2.keras'")

# Sauvegarder le DataFrame, l'encoder et le scaler
df.to_pickle("recipes_df_v2.pkl")
with open("label_encoder_v2.pkl", "wb") as f:
    pickle.dump(encoder, f)
with open("scaler_v2.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("DataFrame, encoder et scaler sauvegardés")

# Fonction pour générer un plan de repas
def generate_meal_plan():
    days = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi"]
    meal_plan = {}
    weekly_selected_indices = set()
    
    for day in days:
        num_meals = random.randint(1, 2)
        meals = []
        daily_selected_indices = set()
        for _ in range(num_meals):
            type_plat = random.choice(["entrée", "plat principal", "dessert"])
            try:
                type_plat_encoded = encoder.transform([type_plat])[0]
                X_input = scaler.transform([[type_plat_encoded, random.randint(200, 800), random.randint(15, 60)]])
                prediction = model.predict(X_input, verbose=0)[0]
                for _ in range(10):
                    recette_index = np.random.choice(len(prediction), p=prediction)
                    if recette_index not in weekly_selected_indices:
                        break
                else:
                    recette_index = np.random.choice(len(prediction), p=prediction)
                
                if recette_index not in daily_selected_indices:
                    daily_selected_indices.add(recette_index)
                    weekly_selected_indices.add(recette_index)
                    recette = df.iloc[recette_index]
                    meals.append({
                        "items": [recette["title"] or "Plat sans titre"],
                        "calories": int(recette["calories"]),
                        "servings": int(recette["servings"]),
                        "time": int(recette["time"]),
                        "ingredients": recette["ingredients"]
                    })
            except ValueError:
                continue
        meal_plan[day] = meals
    
    return meal_plan

# Fonction pour générer une liste de courses basée sur NER, sans quantités
def generate_shopping_list(meal_plan):
    ingredients_set = set()  # Utiliser un ensemble pour éviter les doublons
    
    for day, meals in meal_plan.items():
        for meal in meals:
            title = meal["items"][0]
            recette = df[df["title"] == title]
            if not recette.empty:
                ner_items = recette.iloc[0]["NER"]  # Utiliser NER pour les ingrédients
                for ner_item in ner_items:
                    if ner_item and isinstance(ner_item, str) and ner_item.strip():  # Ignorer None, non-chaînes, ou chaînes vides
                        ingredients_set.add(ner_item.strip())
    
    return list(ingredients_set)  # Convertir en liste pour la sortie

# Tester les fonctions
meal_plan = generate_meal_plan()
print(json.dumps(meal_plan, ensure_ascii=False, indent=2))

shopping_list = generate_shopping_list(meal_plan)
print(json.dumps(shopping_list, ensure_ascii=False, indent=2))