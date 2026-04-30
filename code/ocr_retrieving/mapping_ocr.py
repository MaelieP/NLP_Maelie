import pandas as pd
import json

# 1. Charger le dataset original
for annee in ['1988', '1993']:
    df_orig = pd.read_csv(f'data/brute/archelect_search_{annee}.csv')

    # 2. Charger les résultats OCR (le fichier JSON généré précédemment)
    with open(f'data/ocr_retrieval/ocr_recovered_{annee}.json', 'r', encoding='utf-8') as f:
        data_ocr = json.load(f)

    # Convertir le JSON en DataFrame
    df_ocr = pd.DataFrame(data_ocr)

    # 3. Fusionner les deux DataFrames
    # 'how=left' est crucial : on garde toutes les lignes du CSV original, 
    # et on ajoute le texte là où l'URL correspond.
    df_final = pd.merge(
        df_orig, 
        df_ocr[['ocr_url', 'text']], 
        on='ocr_url', 
        how='left'
    )

    # 4. Vérification rapide
    nb_succes = df_final['text'].notna().sum()
    print(f"Mapping terminé : {nb_succes} textes intégrés sur {len(df_final)} lignes.")

    # 5. Sauvegarder le résultat final
    df_final.to_csv(f'data/final/final_dirty/archelect_{annee}_complet.csv', index=False, encoding='utf-8')