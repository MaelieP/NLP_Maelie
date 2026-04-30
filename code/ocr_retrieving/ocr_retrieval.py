import pandas as pd
import requests
import time
import json
import os
from tqdm import tqdm

# Paramètres

for annee in ['1988', '1993']:
    INPUT_FILE = f'data/brute/archelect_search_{annee}.csv'
    OUTPUT_FILE = f'data/ocr_retrieval/ocr_recovered_{annee}.json'

    TIMEOUT_SEC = 3 # Si ça ne répond pas en 3s, on considère que c'est mort
    DELAY_BETWEEN = 0.5 # Petit délai de courtoisie

    df = pd.read_csv(INPUT_FILE)

    # On vérifie si un fichier de progression existe déjà pour reprendre là où on s'est arrêté
    recovered_data = []
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            recovered_data = json.load(f)
        print(f"Reprise du projet : {len(recovered_data)} lignes déjà traitées.")

    # Extraction des URLs déjà traitées pour éviter les doublons
    processed_urls = {item['ocr_url'] for item in recovered_data}

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    # Boucle principale
    try:
        for index, row in tqdm(df.iterrows(), total=len(df)):
            url = row['ocr_url']
            
            # Sauter si déjà fait
            if url in processed_urls:
                continue
                
            text_content = None
            try:
                # Timeout court : si c'est instantané, ça passe. Sinon, on skip.
                response = requests.get(url, headers=headers, timeout=TIMEOUT_SEC)
                if response.status_code == 200:
                    text_content = response.text
            except Exception:
                text_content = None # Devient NaN/null dans le JSON

            # Stockage
            recovered_data.append({
                'original_index': index,
                'ocr_url': url,
                'text': text_content
            })

            # Sauvegarde toutes les 5 itérations (plus fréquent car rapide)
            if index % 5 == 0:
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(recovered_data, f, ensure_ascii=False, indent=2)
            
            time.sleep(DELAY_BETWEEN)

    except KeyboardInterrupt:
        print("\nInterruption manuelle. Sauvegarde des données acquises...")

    finally:
        # Sauvegarde finale quoi qu'il arrive
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(recovered_data, f, ensure_ascii=False, indent=2)
        print(f"Terminé. Données sauvegardées dans {OUTPUT_FILE}")