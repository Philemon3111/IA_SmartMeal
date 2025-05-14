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
    "homard", "gelatin", "gélatine de porc", "gelatine", "Boeuf", "Porc", "Poulet", "Dinde", "Canard", "Oie", "Agneau", 
    "Veau", "Chevreau", "Lapin", "Cerf", "Sanglier", "Poisson", "Crevettes", "Crabe", "Homard", "Moules", "Huîtres", 
    "Calmar", "Poulpe", "Escargots", "Foie gras", "Saucisses", "Jambon", "Bacon", "Lardons", "Pâté de foie", "Rillettes", 
    "Gelée de viande", "Bouillon de poulet", "Bouillon de boeuf", "Bouillon de poisson", "Gélatine de porc", "Graisse animale", "Suif", 
    "Saindoux", "Présure animale"
]

VEGAN_SAFE_INGREDIENTS = ["lait d'amande","lait de coco","lait de soja","lait d'avoine","lait de riz","lait de noisette","lait de chanvre","lait de macadamia","lait de cajou",
                          "lait de noix","lait de quinoa","lait de sesame","lait de tournesol","lait de pistache","lait de noix du bresil","lait d'epeautre","lait de millet",
                          "creme de coco","creme de soja","creme d'amande","creme d'avoine","creme de cajou","creme de noisette","creme de riz","creme de chanvre",
                          "creme de macadamia","crème de coco","crème de soja","crème d'amande","crème d'avoine","crème de cajou","crème de noisette","crème de riz","crème de chanvre",
                          "crème de macadamia", "crème de champignons","agar-agar","carraghenane","pectine","gomme de guar","gomme xanthane","fecule de mais","fecule de tapioca","fecule de pomme de terre",
                          "arrow-root","kuzu","riz","quinoa","millet","sarrasin","amarante","teff","epeautre","kamut","farine de riz","farine de sarrasin","farine de quinoa",
                          "farine d'amarante","farine de teff","farine de mais","farine de pois chiche","farine de lentilles","lentilles rouges","lentilles vertes","lentilles noires",
                          "pois chiches","haricots noirs","haricots rouges","haricots blancs","haricots azuki","pois casses","feves","lupins","pommes","bananes","oranges","citrons",
                          "limes","poires","mangues","ananas","framboises","myrtilles","fraises","mures","cerises","peches","abricots","grenades","kiwis","dattes","figues","raisins",
                          "melons","pasteques","raisins secs","dattes sechees","abricots secs","figues sechees","pruneaux","cranberries sechees","baies de goji","mulberries",
                          "carottes","courgettes","aubergines","poivrons rouges","poivrons jaunes","poivrons verts","tomates","concombres","epinards","chou kale","laitue","roquette",
                          "chou-fleur","brocoli","chou de bruxelles","betteraves","panais","navets","radis","oignons","ail","poireaux","celeri","asperges","artichauts",
                          "champignons de paris","shiitake","pleurotes","amandes","noix de cajou","noix","noisettes","noix de pecan","noix du bresil","noix de macadamia",
                          "pistaches","graines de tournesol","graines de lin","graines de chia","graines de sesame","graines de courge","graines de chanvre","huile d'olive",
                          "huile de coco","huile de sesame","huile de tournesol","huile de colza","huile de lin","huile d'avocat","huile de noix","huile de pepins de raisin",
                          "sirop d'erable","sucre de coco","sirop d'agave","sucre de canne","melasse","stevia","sucre de datte","nectar de coco","tofu","tempeh","seitan",
                          "proteines de pois","proteines de soja texturees","edamame","basilic","persil","coriandre","menthe","thym","romarin","origan","ciboulette","aneth",
                          "curcuma","cumin","paprika","paprika fume","gingembre","cannelle","muscade","clous de girofle","piment de cayenne","poudre de chili","poivre noir",
                          "poivre blanc","cardamome","safran","ras el hanout","garam masala","sauce soja","tamari","miso","vinaigre de cidre","vinaigre balsamique","vinaigre de riz",
                          "moutarde","tahini","sauce sriracha","sauce harissa","lait de coco en conserve","puree de tomates","ketchup vegan","mayonnaise vegane","levure chimique",
                          "bicarbonate de soude","levure nutritionnelle","cacao en poudre","chocolat noir vegan","pepites de chocolat vegan","vanille","extrait de vanille",
                          "noix de coco rapee","farine de coco","farine d'amande","farine de chataigne","sirop de riz","aquafaba","pate de dattes","pate de miso","pate de curry",
                          "bouillon de legumes","vermicelles de riz","nouilles de soba","nouilles de riz","pates de ble dur","pate brisee vegane","pate feuilletee vegane",
                          "pain pita vegan","tortillas de mais","tortillas de ble","chapelure vegane","graines de pavot","polenta","couscous","bulgur","flocons d'avoine",
                          "granola vegan","muesli vegan","fruits confits","compote de pommes","puree de banane","puree de courge","lait condense de coco","beurre de cacahuete",
                          "beurre d'amande","beurre de noix de cajou","pate de sesame","pate de pistache","poudre d'amande","poudre de noisette","poudre de cacao","cafe","the",
                          "tisanes","eau de fleur d'oranger","eau de rose","sirop de grenadine vegan","jus de citron","jus de lime","jus d'orange","jus de pomme","jus de cranberry",
                          "gélatine à saveur de cerise", "gélatine de fraise", "gélatine à la fraise"]
# Liste des ingrédients exclus pour le régime végan (laitiers, œufs, miel)
VEGAN_EXCLUDED_INGREDIENTS = [
    "milk", "cheese", "cream", "yogurt", "egg", "eggs", "honey",
    "lait", "fromage", "crème", "creme", "yaourt", "œuf", "oeuf", "œufs", "oeufs", 
    "miel", "Lait", "Fromage", "Yaourt", "Crème", "Beurre", "Ghee", "Œufs", "Miel", 
    "Cire d'abeille", "Lactose", "Caséine", "Lactosérum", "Petit-lait", "Crème glacée", 
    "Chocolat au lait", "Mayonnaise classique", "Pâtisseries à base de beurre", "beurre", 
    "Sauces crémeuses", "Compléments alimentaires avec oméga-3 d'origine animale", 
    "Confitures avec gélatine", "Bonbons avec gélatine", "Médicaments avec gélatine"
]

NO_PORC_EXCLUDED_INGREDIENTS= ["Porc", "Jambon", "Bacon", "Saucisses de porc", "Lard", "Lardons", "Pâté de porc", 
                               "Rillettes de porc", "Salami de porc", "Chorizo de porc", "Prosciutto", "Pancetta", 
                               "Côtelettes de porc", "Rôti de porc", "Échine de porc", "Filet de porc", "Saindoux", 
                               "Graisse de porc", "Gélatine de porc", "Couenne de porc", "Pieds de porc", "Tête de porc", 
                               "Sauces à base de porc", "Bouillon de porc", "Charcuterie contenant du porc"]

KETO_EXCLUDED_INGREDIENT = ["Blé", "Riz", "Maïs", "Orge", "Avoine", "Quinoa", "Sarrasin", "Haricots", "Lentilles", "Pois chiches", 
                            "Pois", "Soja", "Bananes", "Raisins", "Mangues", "Ananas", "Pommes", "Oranges", "Poires", 
                            "Fruits secs", "Jus de fruits", "Pommes de terre", "Patates douces", "Panais", "Carottes", "Betteraves", 
                            "Sucre", "Miel", "Sirop d’érable", "Sirop d’agave", "Sirop de maïs", "Maltodextrine", "Dextrose", 
                            "Bonbons", "Biscuits", "Chips", "Craquelins", "Barres de céréales", "Ketchup", "Sauce barbecue", 
                            "Sauce teriyaki", "Sodas", "Boissons énergétiques", "Bières", "Vins sucrés", "Lait", "Yaourts sucrés", 
                            "Crèmes glacées", "Fromages fondus sucrés", "Huile de canola", "Huile de maïs", "Huile de soja", "Margarine", 
                            "Graisses trans", "Amidon", "Farine de blé", "Farine de maïs", "Aliments panés", "Sirop de glucose-fructose", 
                            "Maltitol", "Sorbitol"]

PALEO_EXCLUDED_INGREDIENT = ["Blé", "Riz", "Maïs", "Orge", "Avoine", "Quinoa", "Sarrasin", "Seigle", "Haricots", "Lentilles", 
                             "Pois chiches", "Pois", "Soja", "Tofu", "Lait", "Fromage", "Yaourt", "Crème", "Beurre", "Crème glacée", 
                             "Sucre", "Sirop de maïs", "Sirop d’agave", "Miel artificiel", "Aspartame", "Saccharine", "Pommes de terre", 
                             "Huile de canola", "Huile de canola", "Huile de maïs", "Huile de soja", "Huile de coton", "Huile de carthame", 
                             "Margarine", "Bonbons", "Biscuits", "Gâteaux", "Pâtisseries", "Chips", "Craquelins", "Sodas", "Jus de fruits", 
                             "Bières", "Vins sucrés", "Boissons énergétiques", "Aliments frits", "Fast-food", "Sauces transformées", 
                             "Ketchup", "Mayonnaise industrielle", "Farine de blé", "Farine de maïs", "Amidon", "Sirop de glucose-fructose", 
                             "Maltodextrine", "Dextrose", "Protéines de soja", "Protéines de lactosérum", "Lactosérum", "Caséine"]

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

df["ingredients"] = df["ingredients"].apply(lambda x: [ing.lower() for ing in x])

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
    if diet.lower() == "végétarien" :
        for ing in ingredients:
            if any(non_veg in ing for non_veg in NON_VEGETARIAN_INGREDIENTS):
                print(f"Recette '{recipe['title']}' rejetée pour ingrédient non-végétarien : {ing}")
                return False
    
    # Vérifier le régime végan
    if diet.lower() == "végan":
        for ing in ingredients:
            # Vérifier si l'ingrédient est dans VEGAN_SAFE_INGREDIENTS
            if any(safe_ing in ing for safe_ing in VEGAN_SAFE_INGREDIENTS):
                continue  # Ignorer les ingrédients végans sûrs
            if any(non_vegan in ing for non_vegan in NON_VEGETARIAN_INGREDIENTS + VEGAN_EXCLUDED_INGREDIENTS):
                print(f"Recette '{recipe['title']}' rejetée pour ingrédient non-végan : {ing}")
                return False
            
    # Vérifier le régime sans porc
    if diet.lower() == "sans porc":
        for ing in ingredients:
            if any(non_vegan in ing for non_vegan in NO_PORC_EXCLUDED_INGREDIENTS):
                print(f"Recette '{recipe['title']}' rejetée pour ingrédient sans porc : {ing}")
                return False
    
    # Vérifier le régime végan
    if diet.lower() == "keto":
        for ing in ingredients:
            if any(non_vegan in ing for non_vegan in KETO_EXCLUDED_INGREDIENT):
                print(f"Recette '{recipe['title']}' rejetée pour ingrédient non-keto : {ing}")
                return False
    # Vérifier le régime végan
    if diet.lower() == "paleo":
        for ing in ingredients:
            if any(non_vegan in ing for non_vegan in PALEO_EXCLUDED_INGREDIENT):
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
        
        # Vérifier que number_of_meals est un entier positif
        if not isinstance(number_of_meals, int) or number_of_meals < 1:
            print("Erreur : number_of_meals doit être un entier positif")
            return {"error": "Invalid number_of_meals, must be a positive integer"}
        
        try:
            start_idx = days.index(grocery_day)
            days = days[start_idx:] + days[:start_idx]
        except ValueError:
            pass
        
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
        
        meals_per_day = number_of_meals  # number_of_meals est le nombre de repas par jour
        print(f"Nombre de repas par jour : {meals_per_day}")
    else:
        valid_recipes = df
        meals_per_day = random.randint(1, 2)  # Valeur par défaut si preferences est None
        print(f"Nombre de repas par jour (par défaut) : {meals_per_day}")

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
        num_meals = meals_per_day
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
    
    #if preferences["diet"].lower() not in ["végan", "végétarien", "none"]:
    #    return jsonify({"error": "Unsupported diet"}), 400
    
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
