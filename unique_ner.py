import json

with open('newrecipe.json', 'r', encoding='utf-8') as f:
    data = json.load(f) 

all_ner = []
for recipe in data:
    if "NER" in recipe: 
        all_ner.extend(recipe["NER"])

unique_ner = list(set(all_ner))

unique_ner_sorted = sorted(unique_ner)

print("Unique NER items:")
for item in unique_ner_sorted:
    print(f"- {item}")

with open('unique_ner.json', 'w', encoding='utf-8') as f:
    json.dump(unique_ner_sorted, f, ensure_ascii=False, indent=2)

print("\nSaved to 'unique_ner.json'")