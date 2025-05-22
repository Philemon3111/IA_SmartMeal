import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder, StandardScaler
import random
import pickle

# Load the JSON file
with open('newrecipe.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Create a list of recipes with relevant fields
recipes_data = []
for recipe in data:
    title = recipe.get("title", "").lower()
    if title is None or not isinstance(title, str):
        title = ""
    
    ingredients = recipe.get("ingredients", [])
    ner = recipe.get("NER", [])
    calories = recipe.get("calories", 0)  # Use provided calories
    type_plat = recipe.get("type", "Dessert").lower()  # Use provided type, default to Dessert
    nutriments = recipe.get("nutriments", {})
    
    # Extract nutrient values, default to 0 if missing
    lipide = nutriments.get("lipide", 0.0)
    glucide = nutriments.get("glucide", 0.0)
    proteine = nutriments.get("proteine", 0.0)
    fibre = nutriments.get("fibre", 0.0)
    
    recipes_data.append({
        "type_plat": type_plat,
        "title": title,
        "instructions": " ".join(recipe.get("directions", ["Instructions missing"])),
        "ingredients": ingredients,
        "NER": ner,
        "calories": calories,
        "lipide": lipide,
        "glucide": glucide,
        "proteine": proteine,
        "fibre": fibre
    })

# Convert to DataFrame
df = pd.DataFrame(recipes_data)

# Convert NER lists to strings for deduplication, excluding None
df["NER_str"] = df["NER"].apply(lambda x: ",".join(sorted([item for item in x if item is not None])) if isinstance(x, list) else x)
# Remove duplicates based on title and NER_str
df = df.drop_duplicates(subset=["title", "NER_str"])
# Drop temporary column
df = df.drop(columns=["NER_str"])

# Check the distribution of meal types
print("Distribution of meal types:")
print(df["type_plat"].value_counts())

# Encode meal types
encoder = LabelEncoder()
df["type_plat_encoded"] = encoder.fit_transform(df["type_plat"])

# Prepare input features with calories and nutrients
X = df[["type_plat_encoded", "calories", "lipide", "glucide", "proteine", "fibre"]].values
y = np.arange(len(df))

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Build the model
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

# Train the model
model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Save the model
model.save("meal_plan_model_new.keras")
print("Model saved as 'meal_plan_model_new.keras'")

# Save the DataFrame, encoder, and scaler
df.to_pickle("recipes_new.pkl")
with open("label_new.pkl", "wb") as f:
    pickle.dump(encoder, f)
with open("scaler_new.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("DataFrame, encoder, and scaler saved")

# Function to generate a meal plan
def generate_meal_plan():
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    meal_plan = {}
    weekly_selected_indices = set()
    
    for day in days:
        num_meals = random.randint(1, 2)
        meals = []
        daily_selected_indices = set()
        for _ in range(num_meals):
            type_plat = random.choice(["entree", "plat", "dessert"])
            try:
                type_plat_encoded = encoder.transform([type_plat])[0]
                # Use average nutrient values for prediction input
                X_input = scaler.transform([[
                    type_plat_encoded,
                    random.randint(200, 4000),  # Broad calorie range
                    random.uniform(0, 200),     # Random lipide
                    random.uniform(0, 500),     # Random glucide
                    random.uniform(0, 300),     # Random proteine
                    random.uniform(0, 50)       # Random fibre
                ]])
                prediction = model.predict(X_input, verbose=0)[0]
                for _ in range(10):  # Try to avoid duplicates
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
                        "items": [recette["title"] or "Untitled dish"],
                        "calories": int(recette["calories"]),
                        "nutriments": {
                            "lipide": float(recette["lipide"]),
                            "glucide": float(recette["glucide"]),
                            "proteine": float(recette["proteine"]),
                            "fibre": float(recette["fibre"])
                        },
                        "ingredients": recette["ingredients"]
                    })
            except ValueError:
                continue
        meal_plan[day] = meals
    
    return meal_plan

# Function to generate a shopping list based on NER
def generate_shopping_list(meal_plan):
    ingredients_set = set()  # Use a set to avoid duplicates
    
    for day, meals in meal_plan.items():
        for meal in meals:
            title = meal["items"][0]
            recette = df[df["title"] == title]
            if not recette.empty:
                ner_items = recette.iloc[0]["NER"]  # Use NER for ingredients
                for ner_item in ner_items:
                    if ner_item and isinstance(ner_item, str) and ner_item.strip():  # Ignore None, non-strings, or empty strings
                        ingredients_set.add(ner_item.strip())
    
    return list(ingredients_set)  # Convert to list for output

# Test the functions
meal_plan = generate_meal_plan()
print(json.dumps(meal_plan, ensure_ascii=False, indent=2))

shopping_list = generate_shopping_list(meal_plan)
print(json.dumps(shopping_list, ensure_ascii=False, indent=2))