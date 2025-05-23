import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import random
# import requests
from flask import Flask, jsonify, request
import re
from collections import defaultdict
from fractions import Fraction
import unicodedata
from fuzzywuzzy import fuzz

app = Flask(__name__)

# Charger les données sauvegardées
dir = "version7/"
try:
    df = pd.read_pickle(dir + "recipes.pkl")
    with open(dir + "label.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open(dir + "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    model = tf.keras.models.load_model(dir + "meal_plan.keras")
    print("Model and data loaded successfully")
except FileNotFoundError as e:
    print(f"Error: Missing file - {e}")
    exit(1)

# Liste des ingrédients non-végétariens (viandes, poissons, gélatine)

NO_PORC_EXCLUDED_INGREDIENTS= ["porc","jambon","bacon","saucisses de porc","lard","lardons","pate de porc","rillettes de porc",
                               "salami de porc","chorizo de porc","prosciutto","pancetta","cotelettes de porc","roti de porc",
                               "echine de porc","filet de porc","saindoux","graisse de porc","gelatine de porc","couenne de porc",
                               "pieds de porc","tete de porc","sauces a base de porc","bouillon de porc","charcuterie contenant du porc"]

KETO_EXCLUDED_INGREDIENT = ["ble","riz","mais","orge","avoine","quinoa","sarrasin","haricots","lentilles","pois chiches","pois",
                            "soja","bananes","raisins","mangues","ananas","pommes","oranges","poires","fruits secs","jus de fruits",
                            "pommes de terre","patates douces","panais","carottes","betteraves","sucre","miel","sirop d'erable",
                            "sirop d'agave","sirop de mais","maltodextrine","dextrose","bonbons","biscuits","chips","craquelins",
                            "barres de cereales","ketchup","sauce barbecue","sauce teriyaki","sodas","boissons energetiques","bieres",
                            "vins sucres","lait","yaourts sucres","cremes glacees","fromages fondus sucres","huile de canola",
                            "huile de mais","huile de soja","margarine","graisses trans","amidon","farine de ble","farine de mais",
                            "aliments panes","sirop de glucose-fructose","maltitol","sorbitol"]

PALEO_EXCLUDED_INGREDIENT = ["ble","riz","mais","orge","avoine","quinoa","sarrasin","seigle","haricots","lentilles","pois chiches",
                             "pois","soja","tofu","lait","fromage","yaourt","creme","beurre","creme glacee","sucre","sirop de mais",
                             "sirop d'agave","miel artificiel","aspartame","saccharine","pommes de terre","huile de canola",
                             "huile de mais","huile de soja","huile de coton","huile de carthame","margarine","bonbons","biscuits",
                             "gateaux","patisseries","chips","craquelins","sodas","jus de fruits","bieres","vins sucres",
                             "boissons energetiques","aliments frits","fast-food","sauces transformees","ketchup",
                             "mayonnaise industrielle","farine de ble","farine de mais","amidon","sirop de glucose-fructose",
                             "maltodextrine","dextrose","proteines de soja","proteines de lactoserum","lactoserum","caseine"]


# Dictionnaire pour les allergènes avec mots-clés précis
ALLERGEN_KEYWORDS = {
    "lait": ["milk", "cheese", "cream", "yogurt", "whey", "lait", "fromage", "creme", "yaourt", "lactoserum", "petit-lait", "caseine", "lactose", "beurre", "ghee", "creme glacee", "lait en poudre", "lait condense", "lait evapore"],
    "oeufs": ["egg", "eggs", "mayonnaise", "oeuf", "oeufs", "albumen", "ovoproduit", "lecitine d'oeuf", "pate d'oeuf", "jaune d'oeuf", "blanc d'oeuf"],
    "moutarde": ["mustard", "mustard seed", "moutarde", "graine de moutarde", "poudre de moutarde", "huile de moutarde"],
    "cacahuetes": ["peanut", "peanut oil", "peanut butter", "cacahuete", "huile de cacahuete", "beurre de cacahuete", "arachide", "huile d'arachide"],
    "fruits a coque": ["almond", "hazelnut", "walnut", "cashew", "pistachio", "amande", "noisette", "noix", "noix de cajou", "pistache", "noix de pecan", "noix du bresil", "noix de macadamia", "pignon de pin", "chataigne", "huile d'amande", "huile de noisette", "beurre d'amande", "beurre de noix de cajou"],
    "gluten": ["wheat", "barley", "rye", "oats", "ble", "orge", "seigle", "avoine", "farine de ble", "farine d'orge", "farine de seigle", "malt", "amidon de ble", "semoule", "couscous", "bulgur", "epautre", "kamut", "triticale"],
    "soja": ["soy", "soya", "soybean", "soja", "graine de soja", "huile de soja", "lecitine de soja", "tofu", "tempeh", "miso", "sauce soja", "tamari", "edamame", "proteine de soja", "lait de soja", "creme de soja"],
    "poisson": ["fish", "salmon", "tuna", "cod", "poisson", "saumon", "thon", "morue", "anchois", "sardine", "hareng", "maquereau", "surimi", "caviar", "huile de poisson"],
    "crustaces": ["shrimp", "crab", "lobster", "crevette", "crabe", "homard", "ecrevisse", "langoustine", "moule", "huitre", "coquille saint-jacques", "calmar", "poulpe"],
    "sesame": ["sesame", "sesame seed", "Kuhlschmied", "tahini", "huile de sesame", "graines de sesame", "pate de sesame"],
    "celeri": ["celery", "celeri", "celeri-rave", "graine de celeri", "jus de celeri"]
}

chemin_vegan = "alimentliste/non_vegan.json"
# chemin_vegan = "alimentliste/non_vegan.json"
chemin_paleo = "alimentliste/non_keto.json"
chemin_sporc = "alimentliste/sans_porc.json"
chemin_keto = "alimentliste/non_paleo.json"
ner_vegan = None
# Vérifier si le fichier existe
if os.path.exists(chemin_vegan):
    # Ouvrir et lire le fichier JSON
    with open(chemin_vegan, 'r', encoding='utf-8') as fichier:
        ner_vegan = json.load(fichier)

chemin_vegetarien = "alimentliste/non_vegetarien.json"
ner_vegetarien = None
# Vérifier si le fichier existe
if os.path.exists(chemin_vegetarien):
    # Ouvrir et lire le fichier JSON
    with open(chemin_vegetarien, 'r', encoding='utf-8') as fichier:
        ner_vegetarien = json.load(fichier)

ner_sporc = None
# Vérifier si le fichier existe
if os.path.exists(chemin_sporc):
    # Ouvrir et lire le fichier JSON
    with open(chemin_sporc, 'r', encoding='utf-8') as fichier:
        ner_sporc = json.load(fichier)

ner_keto = None
# Vérifier si le fichier existe
if os.path.exists(chemin_keto):
    # Ouvrir et lire le fichier JSON
    with open(chemin_keto, 'r', encoding='utf-8') as fichier:
        ner_keto = json.load(fichier)

ner_paleo = None
# Vérifier si le fichier existe
if os.path.exists(chemin_paleo):
    # Ouvrir et lire le fichier JSON
    with open(chemin_paleo, 'r', encoding='utf-8') as fichier:
        ner_paleo = json.load(fichier)        
# Normaliser les ingrédients au chargement
# def normalize_ingredient(ingredient):
#     translations = {
#         "lait": "milk",
#         "œufs": "egg",
#         "moutarde": "mustard",
#         "cacahuètes": "peanut",
#         "fruits à coque": "nut"
#     }
#     for fr, en in translations.items():
#         ingredient = ingredient.replace(fr, en)
#     return ingredient.lower()

# df["ingredients"] = df["ingredients"].apply(lambda x: [normalize_ingredient(ing) for ing in x])

# Fonction pour vérifier si une recette respecte les contraintes
def is_recipe_valid(recipe, allergies, diet, max_calories=None):
    ingredients = [ing for ing in recipe["NER"]]
    
    # Vérifier les allergènes
    for allergen, is_allergic in allergies.items():
        if is_allergic:
            keywords = ALLERGEN_KEYWORDS.get(allergen, [allergen.lower()])
            if any(any(keyword in ing for keyword in keywords) for ing in ingredients):
                print(f"Recette '{recipe['title']}' rejetée pour allergène : {allergen}")
                return False
    
    # Vérifier le régime végétarien
    if diet.lower() == "végétarien" or diet.lower() == "vegetarian":
        for ing in ingredients:
            if any(non_veg in ing for non_veg in ner_vegetarien):
                print(f"Recette '{recipe['title']}' rejetée pour ingrédient non-végétarien : {ing}")
                return False
    
    # Vérifier le régime végan
    if diet.lower() == "végan":
        for ing in ingredients:
            if any(non_vegan in ing for non_vegan in ner_vegan + ner_vegetarien):
                print(f"Recette '{recipe['title']}' rejetée pour ingrédient non-végan : {ing}")
                return False
            
    # Vérifier le régime sans porc
    if diet.lower() == "sans porc":
        for ing in ingredients:
            if any(non_sporc in ing for non_sporc in ner_sporc):
                print(f"Recette '{recipe['title']}' rejetée pour ingrédient sans porc : {ing}")
                return False
    
    # Vérifier le régime keto
    if diet.lower() == "keto":
        for ing in ingredients:
            if any(non_keto in ing for non_keto in ner_keto):
                print(f"Recette '{recipe['title']}' rejetée pour ingrédient non-keto : {ing}")
                return False
    
    # Vérifier le régime végan
    if diet.lower() == "paleo":
        for ing in ingredients:
            if any(non_paleo in ing for non_paleo in ner_paleo):
                print(f"Recette '{recipe['title']}' rejetée pour ingrédient non-paléo : {ing}")
                return False

    # Vérifier les calories
    if max_calories and recipe["calories"] > max_calories:
        print(f"Recette '{recipe['title']}' rejetée pour calories : {recipe['calories']} > {max_calories}")
        return False
    
    return True

def normalize_string(s):
    """Normalise une chaîne : supprime accents, met en minuscules, gère singulier/pluriel."""
    # Supprimer accents
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    # Mettre en minuscules
    s = s.lower().strip()
    # Supprimer pluriels simples (ex. "oignons" -> "oignon")
    s = re.sub(r's$', '', s)
    return s

def prepare_inventory_ingredients(inventory):
    """Extrait et nettoie les ingrédients de l'inventaire."""
    ingredients = []
    if inventory:
        for category in ['grocery', 'fresh_produce']:
            if category in inventory:
                for item in inventory[category]:
                    name = item.get('name', '')
                    if name:
                        normalized_name = normalize_string(name)
                        if normalized_name not in ingredients:
                            ingredients.append(normalized_name)
    return ingredients

# Fonction pour générer un plan de repas
def generate_meal_plan(preferences=None, inventory_ingredients=None):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    meal_plan = {}
    weekly_selected_indices = set()
    
    if preferences:
        allergies = preferences.get("allergy", {})
        diet = preferences.get("diet", "none")
        goal = preferences.get("goal", "none")
        number_of_meals = preferences.get("number_of_meals", 6)
        grocery_day = preferences.get("grocery_day", "Monday")
        max_calories = preferences.get("max_calories", None)
        
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
        
        meals_per_day = number_of_meals
        print(f"Nombre de repas par jour : {meals_per_day}")
    else:
        valid_recipes = df
        meals_per_day = random.randint(1, 2)
        print(f"Nombre de repas par jour (par défaut) : {meals_per_day}")

    # Calculate ingredient scores if inventory_ingredients is provided
    ingredient_scores = None
    if inventory_ingredients:
        ingredient_scores = np.zeros(len(valid_recipes))
        for idx, recipe in valid_recipes.iterrows():
            recipe_ingredients = [ing for ing in recipe["ingredients"]]
            matches = 0
            matched_ingredients = []
            for inv_ing in inventory_ingredients:
                for recipe_ing in recipe_ingredients:
                    if fuzz.partial_ratio(inv_ing, recipe_ing) > 80:
                        matches += 1
                        matched_ingredients.append((inv_ing, recipe_ing))
                        break
            ingredient_scores[valid_recipes.index.get_loc(recipe.name)] = matches
            # print(f"Recette '{recipe['title']}': {matches} correspondances -> {matched_ingredients}")
        max_score = ingredient_scores.max() if ingredient_scores.max() > 0 else 1
        ingredient_scores = ingredient_scores / max_score
        # print(f"Scores d'ingrédients calculés : min={ingredient_scores.min()}, max={ingredient_scores.max()}, max_score brut={max_score}")

    valid_meal_types = list(encoder.classes_)
    # print(f"Valid meal types for selection: {valid_meal_types}")

    for day in days:
        num_meals = meals_per_day
        meals = []
        daily_selected_indices = set()

        for _ in range(num_meals+1):
            type_plat = random.choice(valid_meal_types)
            try:
                type_plat_encoded = encoder.transform([type_plat])[0]
                # Use dataset ranges for all 6 features
                X_input = scaler.transform([[
                    type_plat_encoded,
                    random.uniform(df["calories"].min(), df["calories"].max()),
                    random.uniform(df["lipide"].min(), df["lipide"].max()),
                    random.uniform(df["glucide"].min(), df["glucide"].max()),
                    random.uniform(df["proteine"].min(), df["proteine"].max()),
                    random.uniform(df["fibre"].min(), df["fibre"].max())
                ]])
                prediction = model.predict(X_input, verbose=0)[0]
                print(f"Taille de prediction pour {type_plat} le {day}: {len(prediction)}")
                
                # First branch - if preferences or inventory_ingredients
                if preferences or inventory_ingredients:
                    valid_indices = list(valid_recipes.index)
                    valid_indices = [i for i in valid_indices if i < len(df) and df.iloc[i]["type_plat"] == type_plat and i < len(prediction)]                    
                    print(f"Taille de valid_indices pour {type_plat} le {day}: {len(valid_indices)}")
                    
                    # Additional validation checks
                    valid_indices = [i for i in valid_indices if i < len(df) and i < len(prediction)]
                    if not valid_indices:
                        print(f"Aucune recette valide après vérification complète pour {type_plat} le {day}")
                        continue
                    
                    try:
                        valid_probs = prediction[valid_indices]
                    except IndexError as e:
                        print(f"Erreur d'index avec prediction array: {e}")
                        continue
                        
                    print(f"Taille de valid_probs pour {type_plat} le {day}: {len(valid_probs)}")
                    
                    if len(valid_probs) != len(valid_indices):
                        print(f"Erreur : valid_probs ({len(valid_probs)}) et valid_indices ({len(valid_indices)}) ont des tailles différentes")
                        try:
                            recette_index = random.choice(valid_indices)
                        except IndexError:
                            print(f"Impossible de sélectionner dans valid_indices vide pour {type_plat} le {day}")
                            continue
                    elif valid_probs.sum() == 0 or np.isnan(valid_probs).any():
                        print(f"Avertissement : Probabilités invalides pour {type_plat} le {day}, sélection aléatoire")
                        try:
                            recette_index = random.choice(valid_indices)
                        except IndexError:
                            print(f"Impossible de sélectionner dans valid_indices vide pour {type_plat} le {day}")
                            continue
                    else:
                        valid_probs = valid_probs / valid_probs.sum()
                        if ingredient_scores is not None:
                            valid_scores = ingredient_scores[:len(valid_indices)]
                            valid_probs = valid_probs * (0.5 + 0.5 * valid_scores)
                            valid_probs = valid_probs / valid_probs.sum()
                        
                        sequential_indices = list(range(len(valid_indices)))
                        for _ in range(10):
                            try:
                                seq_index = np.random.choice(sequential_indices, p=valid_probs)
                                recette_index = valid_indices[seq_index]
                                if recette_index not in daily_selected_indices:
                                    break
                            except (ValueError, IndexError) as e:
                                print(f"Erreur lors de la sélection aléatoire: {e}")
                                continue
                        else:
                            try:
                                seq_index = np.random.choice(sequential_indices, p=valid_probs)
                                recette_index = valid_indices[seq_index]
                            except (ValueError, IndexError) as e:
                                print(f"Erreur lors de la sélection aléatoire finale: {e}")
                                continue

                # Second branch - else case
                else:
                    valid_indices = list(range(len(df)))
                    valid_indices = [i for i in valid_indices if df.iloc[i]["type_plat"] == type_plat]
                    print(f"Taille de valid_indices pour {type_plat} le {day}: {len(valid_indices)}")
                    
                    if len(prediction) == 0:
                        print(f"Aucune prédiction disponible pour {type_plat} le {day}")
                        continue
                    
                    # Additional validation
                    valid_indices = [i for i in valid_indices if i < len(prediction)]
                    if not valid_indices:
                        print(f"Aucun indice valide après vérification pour {type_plat} le {day}")
                        continue
                    
                    try:
                        prediction = prediction / prediction.sum()
                    except (ValueError, ZeroDivisionError) as e:
                        print(f"Erreur de normalisation des probabilités: {e}")
                        continue
                    
                    for _ in range(10):
                        try:
                            recette_index = np.random.choice(valid_indices, p=prediction[valid_indices])
                            if recette_index not in daily_selected_indices:
                                break
                        except (ValueError, IndexError) as e:
                            print(f"Erreur lors de la sélection aléatoire: {e}")
                            continue
                    else:
                        try:
                            recette_index = np.random.choice(valid_indices, p=prediction[valid_indices])
                        except (ValueError, IndexError) as e:
                            print(f"Erreur lors de la sélection aléatoire finale: {e}")
                            continue

                # Final validation before adding to meals
                if recette_index >= len(df):
                    print(f"Index de recette invalide {recette_index} (taille df: {len(df)})")
                    continue

                try:
                    recette = df.iloc[recette_index]
                    daily_selected_indices.add(recette_index)
                    weekly_selected_indices.add(recette_index)
                    # print("ici")
                    # print(recette)
                    meals.append({
                        "items": [recette["title"] or "Plat sans titre"],
                        "calories": int(recette["calories"]),
                        "nutriments": {
                            "lipide": float(recette["lipide"]),
                            "glucide": float(recette["glucide"]),
                            "proteine": float(recette["proteine"]),
                            "fibre": float(recette["fibre"])
                        },
                        "ingredients": recette["ingredients"],
                        "preparation": recette["instructions"],
                        "NER": recette["NER"],
                        "type": recette["type_plat"],
                        "time": int(recette["time"]),
                        "servings": int(recette["servings"])
                    })
                except IndexError as e:
                    print(f"Erreur d'accès à la recette: {e}")
                    continue
                except KeyError as e:
                    print(f"Colonne manquante dans les données: {e}")
                    continue
                    # print(f"Recette sélectionnée : {recette['title']} (index {recette_index}) pour {type_plat} le {day}")
                else:
                    print(f"Index {recette_index} déjà utilisé ou invalide pour {type_plat} le {day}")
            except (ValueError, IndexError) as e:
                print(f"Erreur lors de la sélection de recette pour {type_plat} le {day}: {e}")
                continue
        
        meal_plan[day] = meals
    
    return meal_plan

# Routes existantes
# @app.route('/meal_plan', methods=['GET'])
# def get_meal_plan():
#     meal_plan = generate_meal_plan({})
#     return jsonify(meal_plan)

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

# @app.route('/custom_meal_plan', methods=['POST'])
# def get_custom_meal_plan():
#     if not request.is_json:
#         return jsonify({"error": "Request must be JSON"}), 400
    
#     preferences = request.get_json()
    
#     # Valider les préférences
#     required_fields = ["allergy", "diet", "goal", "number_of_meals", "grocery_day"]
#     for field in required_fields:
#         if field not in preferences:
#             return jsonify({"error": f"Missing required field: {field}"}), 400
    
#     if not isinstance(preferences["allergy"], dict):
#         return jsonify({"error": "Allergy must be a dictionary"}), 400
    
#     if preferences["diet"].lower() not in ["végan", "végétarien", "none"]:
#         return jsonify({"error": "Unsupported diet"}), 400
    
#     if preferences["goal"].lower() not in ["lose weight", "maintain", "gain weight"]:
#         return jsonify({"error": "Unsupported goal"}), 400
    
#     if not isinstance(preferences["number_of_meals"], int) or preferences["number_of_meals"] < 1:
#         return jsonify({"error": "Invalid number_of_meals"}), 400
    
#     if "max_calories" in preferences:
#         if not isinstance(preferences["max_calories"], (int, float)) or preferences["max_calories"] <= 0:
#             return jsonify({"error": "Invalid max_calories"}), 400
    
#     # Générer le plan de repas
#     meal_plan = generate_meal_plan(preferences)
    
#     if "error" in meal_plan:
#         return jsonify(meal_plan), 400
    
#     return jsonify(meal_plan)

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
    
    # if preferences["diet"].lower() not in ["végan", "végétarien", "none"]:
    #     return jsonify({"error": "Unsupported diet"}), 400
    
    # if preferences["goal"].lower() not in ["lose weight", "maintain", "gain weight"]:
    #     return jsonify({"error": "Unsupported goal"}), 400
    
    if not isinstance(preferences["number_of_meals"], int) or preferences["number_of_meals"] < 1:
        return jsonify({"error": "Invalid number_of_meals"}), 400
    
    if "max_calories" in preferences:
        if not isinstance(preferences["max_calories"], (int, float)) or preferences["max_calories"] <= 0:
            return jsonify({"error": "Invalid max_calories"}), 400
    
    # Générer le plan de repas avec les préférences et les ingrédients
    meal_plan = generate_meal_plan(preferences=preferences, inventory_ingredients=inventory_ingredients)
    
    if "error" in meal_plan:
        return jsonify(meal_plan), 400
    
    # print(meal_plan)
    return jsonify(meal_plan)


categories = {
    "huiles": ["Huile", "Huile d'olive", "Crisco", "Graisse", "Graisse de bacon", "Graisse de viande fondue", "Saindoux", "Shortening", "Spray de cuisson", "Aérosol de cuisson végétal", "Pam"],
    "fruits": ["Abricots", "Ananas", "Bananes", "Canneberges", "Cantaloup", "Cerises", "Citrons", "Citron vert", "Citrouille", "Cocktail de fruits", "Dattes", "Fraises", "Framboises", "Framboisese", "Kakis", "Mandarines", "Mangue", "Myrtilles", "Orange", "Oranges", "Pêches", "Poires", "Pommes", "Pruneaux", "Pulpe de Bananes", "Pulpe de kaki", "Raisins", "Raisins blancs", "Raisins secs", "Raisins verts", "Raisins violets", "Rhubarbe", "Segments d'orange mandarine", "Segments de mandarine", "Tranches d'ananas"],
    "légumes": ["Ail", "Artichauts", "Aubergine", "Betteraves", "Brocoli", "Carottes", "Céleri", "Champignons", "Chou", "Chou rouge", "Chou vert", "Chou-fleur", "Châtaignes d'eau", "Concombre", "Courge", "Courgettes", "Câpres", "Échalotes", "Épinards", "Épinards hachés", "Épinards à la crème", "Feuilles de navet", "Haricots", "Haricots rouges", "Haricots verts", "Laitue", "Maïs", "Maïs en crème", "Maïs entier", "Maïs à la crème", "Navets", "Oignon", "Oignon violet", "Oignons", "Oignons frits", "Oignons jaunes", "Oignons rouge", "Oignons verts", "Okra", "Olives", "Patates douces", "Petits pois", "Pois", "Pois chiches", "Pois mange-tout", "Pois à vache", "Poireaux", "Poivron rouge", "Poivron rouges", "Poivron vert", "Poivron verts", "Poivrons", "Pommes de terre", "Pousses de bambou", "Radis", "Tomates", "Zucchini"],
    "viandes": ["Agneau", "Bacon", "Bifteck", "Bœuf", "Bœuf haché", "Brisket", "Côtelettes de porc", "Dinde", "Foie de veau", "Hamburger", "Hot-dogs", "Jambon", "Os de jambon", "Os à soupe charnus", "Pepperoni", "Porc", "Rôti de Bœuf", "Rôti de palette", "Rôti de porc", "Salami", "Saucisse", "Saucisse d'été", "Saucisse de porc", "Saucisse douce", "Saucisse fumée", "Saucisse piquante", "Saucisse épicée", "Saucisses de Francfort", "Saucisses italiennes douces", "Steak", "Viande hachée", "Viande à ragoût"],
    "poissons": ["Aiglefin", "Anchois", "Chair de crabe", "Chair de crabe imitée", "Crabe", "Crevettes", "Filet de Colin", "Filet de poisson", "Flet", "Fruits de mer", "Huîtres", "Liquide d'huîtres", "Mulet", "Palourdes", "Poisson blanc ferme", "Pétoncles", "Queues d'écrevisses", "Saumon", "Soupe à la crevette", "Thon", "Têtes de poisson"],
    "produits laitiers solides (g)": ["Beurre", "Fromage", "Fromage Cheddar", "Fromage Feta", "Fromage Monterey Jack", "Fromage Mozzarella", "Fromage Muenster", "Fromage Parmesan", "Fromage Provolone", "Fromage Ricotta", "Fromage Romano", "Fromage Suisse", "Fromage Velveeta", "Fromage américain", "Fromage au piment", "Fromage bleu", "Fromage cottage", "Fromage râpé", "Fromage à l'ail", "Fromage à la crème", "Fromage à pizza", "Margarine", "Crème", "Crème Carnation", "Crème aigre", "Crème de champignons", "Crème de céleri", "Crème fouettée", "Crème légère", "Crème sure", "Crème épaisse", "Yaourt", "Yaourt au Citrons", "Yaourt aux fruits", "Lait en poudre"],
    "produits laitiers liquides (ml)": ["Babeurre", "Lait", "Lait concentré", "Lait de coco", "Lait écrémé", "Lait évaporé", "Crème liquide"],
    "oeufs": ["Œuf", "Blanc d'œuf", "Blancs d'œuf", "Jaunes d'Œuf", "Substitut d'œuf"],
    "herbes": ["Aneth", "Basilic", "Ciboulette", "Coriandre", "Estragon", "Feuilles de laurier", "Marjolaine", "Origan", "Persil", "Romarin", "Sauge", "Thym"],
    "épices": ["Ail en poudre", "Anis", "Anisette", "Cannelle", "Cardamome", "Cayenne", "Clous de girofle", "Cumin", "Curcuma", "Curry", "Gingembre", "Graine de moutarde", "Graines de carvi", "Graines de cumin", "Graines de céleri", "Graines de pavot", "Graines de sésame", "Graines de tournesol", "Muscade", "Paprika", "Piment", "Piment de Cayenne", "Piment de la Jamaïque", "Poivre", "Poivre blanc", "Poivre citronné", "Poivre de Cayenne", "Poivre noir", "Poivre rouge moulu", "Quatre-épices", "Racine de gingembre", "Raifort", "Safran", "Sel", "Sel assaisonné", "Sel d'ail", "Sel d'oignon", "Sel de céleri", "Sel gros", "Épice du Moyen-Orient", "Épice jerk", "Épices pour tarte aux pommes", "Épices pour tarte à la citrouille", "Épices à marinades"],
    "céréales": ["Avoine", "Biscuits", "Chapelure", "Corn Chex", "Cornflakes", "Craquelins Graham", "Craquelins de seigle", "Crackers", "Farine", "Farine à lever", "Germe de blé grillé", "Miettes de biscuits", "Miettes de cornflakes", "Miettes de crackers", "Miettes de craquelins", "Miettes de pain", "Orge perlé", "Pain", "Pain blanc", "Pain de blé complet", "Pain de maïs", "Pain de mie", "Pain de seigle", "Pain français", "Pain grillé", "Pain rassis", "Pains au levain", "Pains de levure", "Petits pains", "Riz", "Riz brun", "Semoule de maïs", "Tapioca", "Tortillas", "Tortillas de maïs", "Wontons chinois", "Wrappers wonton"],
    "noix": ["Amandes", "Arachides", "Cacahuètes", "Noix", "Noix de cajou", "Noix de coco", "Noix de pécan", "Noix moulues", "Noix mélangées", "Noix noires", "Pistaches"],
    "pâtisserie": ["Barres Heath", "Biscuits", "Bisquick", "Bits 'O Brickle", "Bonbons", "Bonbons à la menthe poivrée", "Cap'n Crunch", "Cheerios", "Chocolat", "Chocolat au lait", "Choco-bake", "Cool Whip", "Céréales", "Céréales de maïs et de riz", "Doritos", "Dream Whip", "Fleurs en bonbon", "Fruits confits", "Gaufrettes", "Gelée de Pommes", "Gelée de cerise", "Gelée de fraise", "Gelée de groseille", "Gelée de raisin", "Glaçage Orange-Citrons", "Glaçage aux fraises", "Glaçage noix de coco et pécan", "Glaçage à la vanille", "Guimauves", "Gâteau des anges", "Jell-O", "Jell-O à la fraise", "Jell-O à la framboise", "Jello au Citrons", "Life Savers", "Marshmallows", "Miel", "Mini-guimauves", "Mints Frango", "Mélasse", "Oreos", "Pépites de caramel", "Pépites de chocolat", "Pépites de céréales", "Pudding au beurre", "Pudding au chocolat", "Pudding instantané", "Pudding à la vanille", "Pâte de biscuits", "Pâte phyllo", "Pâte sucrée", "Pâte à biscuits", "Pâte à croissants", "Pâte à pizza", "Pâte à tarte", "Sucre", "Sucre brun", "Sucre glace", "Sucre granulé", "Sucre à la cannelle", "Twinkies", "Vers gélifiés"],
    "sauces": ["Concentré de Tomates", "Ketchup", "Mayonnaise", "Miracle Whip", "Moutarde", "Relish sucrée", "Sauce", "Sauce Alfredo légère", "Sauce Ragu", "Sauce Tabasco", "Sauce Tomates", "Sauce Worcestershire", "Sauce aigre-douce", "Sauce au fromage", "Sauce au poulet", "Sauce aux canneberges", "Sauce barbecue", "Sauce caramel", "Sauce chili", "Sauce enchilada", "Sauce piquante", "Sauce pour steak", "Sauce salsa", "Sauce soja", "Sauce taco", "Sauce tamari", "Sauce à Spaghettis", "Sauce à croquettes", "Sauce à pizza", "Soupe au brocoli et au fromage", "Soupe au boeuf au chili", "Soupe au céleri", "Soupe au fromage", "Soupe au poulet", "Soupe aux champignons", "Soupe aux légumes", "Soupe de Tomates", "Soupe de haricots", "Soupe de nouilles", "Soupe de poulet", "Soupe de tomate", "Soupe à l'oignon", "Soupe à la crevette", "Soupe à la crème", "Soupe à la crème d'oignon", "Soupe à la crème de champignons", "Soupe à la crème de céleri", "Soupe à la crème de poulet", "Tabasco", "Trempette au guacamole", "Vinaigre", "Vinaigre blanc", "Vinaigre de cidre", "Vinaigre de vin", "Vinaigre noir", "Vinaigre rouge", "Vinaigrette", "Vinaigrette Catalina", "Vinaigrette Ranch", "Vinaigrette Thousand Island", "Vinaigrette italienne", "Vinaigrette pour salade", "Vinaigrette russe", "Worcestershire"],
    "boissons": ["Bière", "Bourbon", "Brandy", "Café", "Chablis", "Cognac", "Ginger ale", "Jus d'ananas", "Jus d'orange", "Jus de Citrons", "Jus de Pommes", "Jus de Tomates", "Jus de cerise", "Jus de citron vert", "Jus de cornichon", "Jus de framboise", "Jus de mandarine", "Jus de pruneau", "Jus de pêche", "Punch Sangaree", "Rhum", "Sherry", "Soda club", "Triple Sec", "Vin blanc", "Vin rouge", "Vodka", "Xérès", "Eau"],
    "produits pour bébés": ["Nourriture pour bébé aux abricots", "Pots de purée de prunes pour bébés"],
    "produits chimiques/alimentaires": ["Alun", "Bicarbonate de soude", "Cire de paraffine", "Colorant alimentaire", "Fécule de maïs", "Gélatine", "Gélatine au Citrons", "Levure", "Levure chimique", "Paraffine"],
    "mélanges préparés": ["Beau Monde", "Beefogetti", "Chicken Tonight", "Chili Hormel", "Farce Stove Top", "Farce au poulet", "Farce aux herbes", "Manwich", "McCormick Salad Supreme", "Mélange Bisquick", "Mélange Gâteau au chocolat", "Mélange Gâteau blanc", "Mélange Hidden Valley Ranch", "Mélange Shake 'n Bake", "Mélange barbecue Shake 'N Bake", "Mélange d'assaisonnement pour tacos", "Mélange d'oignons", "Mélange de Pouding instantané", "Mélange de farce", "Mélange pour Biscuits", "Mélange pour bouillon", "Mélange pour croûte", "Mélange pour muffins de maïs", "Mélange pour pain de maïs", "Mélange pour pudding", "Mélange pour pudding au chocolat", "Mélange pour sauce brune", "Mélange pour soupe de légumes", "Mélange pour soupe à l'oignon", "Mélange pour vinaigrette", "Mélange à Pudding instantané", "Mélange à biscuits", "Mélange à farce", "Mélange à gâteau", "Mélange à soupe au poulet", "Old Bay", "Salad Supreme", "Season-All", "Spatini", "Stove Top"],
    "arômes et extraits": ["Arôme d'Amandes", "Arôme d'orange", "Arôme de Citrons", "Arôme de courge musquée", "Arôme de noix de beurre", "Arôme de noix de coco", "Arôme de rhum", "Arôme de vanille", "Extrait d'Amandes", "Extrait d'orange", "Extrait de Citrons", "Extrait de rhum", "Extrait de vanille", "Fumée liquide"],
    "conserves et concentrés": ["Compote de pommes", "Concentré de Tomates", "Concentré de jus d'orange", "Confiture d'abricot", "Confiture d'ananas", "Confiture de mûres", "Consommé", "Consommé de boeuf", "Consommé de poulet", "Purée de prunes", "Purée de tomates", "Pâte de Tomates"],
    "garnitures": ["Garniture a dessert", "Garniture au caramel", "Garniture fouettée", "Garniture pour tarte", "Garniture à la fraise"],
    "produits de snacking": ["Bac*Os", "Chips de Tortillas", "Chips de maïs", "Doritos", "Fritos", "Pretzels", "Tater Tots"],
    "pâtes et nouilles": ["Fettucini", "Linguine", "Macaronis", "Nouilles", "Nouilles Ramen", "Nouilles a lasagne", "Nouilles aux épinards", "Nouilles aux Oeuf", "Nouilles chinoises", "Nouilles chow mein", "Nouilles larges", "Nouilles à dumplings aux Oeuf", "Nouilles à lasagne", "Pasta", "Pâtes", "Pâtes en spirale", "Rigatoni", "Spaghettis", "Tortellini", "Vermicelles"],
    "produits végétariens": ["Chair de crabe imitée", "Fausse chair de crabe", "Veg-All"],
    "emballages alimentaires": ["Coquilles", "Croûte à pizza", "Croûtes à tarte", "Croûtons", "Fonds de tarte", "Wontons chinois", "Wrappers wonton"],
    "édulcorants": ["Édulcorant", "Sirop", "Sirop Karo", "Sirop blanc", "Sirop de chocolat", "Sirop de maïs", "Sirop de maïs Karo", "Sirop sundae au chocolat"],
    "produits divers": ["Écorce d'Amandes", "Écorce de pastèque", "Milnot", "Pépites de Citrons", "Pouding au Citrons", "Pouding instantané", "Pouding à la vanille", "Poudre d'ail", "Poudre d'oignon", "Poudre de chili", "Poudre de curry", "Poudre de céleri", "Rotel", "V8", "Velveeta"]
}

# oeuf == 50g
def parse_quantity(ingredient_str):
    """Parse quantity from ingredient string"""
    parts = ingredient_str.split()
    for part in parts:
        if part.replace('.', '').isdigit():
            return float(part)
    return 1

def get_category(item):
    """Find which category an item belongs to"""
    for category, items in categories.items():
        if item in items:
            return category
    return "autres"

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
                std_qty = parse_quantity(ingredient_str)
                quantities[category].append({
                    "name": ingredient,
                    "quantity": std_qty,
                    "unit": "g"
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
            no_unit = ["herbes", "epices"]
            if category in no_unit:
                flattened[category].append(f"{name}")
            elif category in ["huiles", "sauces"]:
                flattened[category].append(f"{name}: {qty} ml")
            else:
                flattened[category].append(f"{name}: {qty} g")

    
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
            no_unit = ["herbes", "epices"]
            if category not in no_unit:
                name, rest = item_str.split(":", 1)
                current_qty = float(rest.split()[0])
                unit = rest.split()[1]
                
                inv_qty = inventory_items.get(name.strip().lower(), 0)
                remaining_qty = max(0, current_qty - inv_qty)
                
                if remaining_qty > 0:
                    if category in ["huiles", "produits laitiers liquides (ml)"]:
                        final_list[category].append(f"{name}: {remaining_qty // 250 + (1 if remaining_qty % 1000 != 0 else 0)} bouteille")
                    else:
                        if name == "Œuf":
                            final_list[category].append(f"{name}: {int(remaining_qty/50)}")
                        elif name == "Eau":
                            continue
                        elif remaining_qty>1000 and unit == "g":
                            final_list[category].append(f"{name}: {remaining_qty/1000} kg")
                        else:
                            final_list[category].append(f"{name}: {remaining_qty} {unit}")
            else:
                final_list[category].append(f"{item_str}")
    
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

    categorized, _ = clean_and_categorize_ingredients(meal_plan)
    
    extracted = extract_quantities(meal_plan, categorized)  
    flattened = flatten_quantities(extracted)
    
    # Subtract inventory if provided
    final_list = subtract_inventory(flattened, inventory) if inventory else flattened
    
    # Clean the output by removing "number" units
    # cleaned_list = {}
    # for category, items in final_list.items():
    #     cleaned_items = []
    #     for item in items:
    #         if ": " in item:
    #             name, quantity = item.split(": ", 1)
    #             if quantity.endswith(" number"):
    #                 cleaned_items.append(f"{name}: {quantity[:-7]}")  # Remove " number"
    #             else:
    #                 cleaned_items.append(item)
    #         else:
    #             cleaned_items.append(item)
    #     cleaned_list[category] = cleaned_items
    
    return jsonify(final_list)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
