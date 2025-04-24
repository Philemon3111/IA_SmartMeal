import requests
from bs4 import BeautifulSoup
import time

# Liste pour stocker les URLs
recipe_urls = []

# URL de départ (ex. : catégorie desserts)
base_url = "https://www.marmiton.org/recettes/index/categorie/dessert?page={}"

# Parcourir plusieurs pages (ex. : 1 à 50)
for page in range(1, 51):
    url = base_url.format(page)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Vérifier les erreurs HTTP
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Trouver les liens des recettes
        links = soup.find_all("a", class_="recipe-card-link")
        for link in links:
            recipe_url = link["href"]
            if not recipe_url.startswith("http"):
                recipe_url = "https://www.marmiton.org" + recipe_url
            recipe_urls.append(recipe_url)
        
        print(f"Page {page} : {len(links)} URLs collectées")
        time.sleep(1)  # Délai pour éviter le blocage
    
    except requests.RequestException as e:
        print(f"Erreur sur la page {page} : {e}")
        break

# Sauvegarder les URLs
with open("recipe_urls.txt", "w") as f:
    for url in recipe_urls:
        f.write(url + "\n")
print(f"Total URLs collectées : {len(recipe_urls)}")