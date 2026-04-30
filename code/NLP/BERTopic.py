import pandas as pd
import json

file = 'outputs/databases/lda_resultats_complet.csv'
df = pd.read_csv(file)

df = df.dropna(subset=['full_text']).copy()



useful_columns = ['annee', 'departement', 'departement-nom','titulaire-soutien',
                  'titulaire-liste', 'taux_chom', 'cleaned_blocks','full_text'
                  ]

#df = df[useful_columns]

#print(df.columns.tolist())

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

# 1. On prépare les documents (on enlève les éventuels NaN restants)
docs = df['full_text'].dropna().tolist()

# 2. On affine les mots vides (Stopwords) pour éviter les mots trop communs
from spacy.lang.fr.stop_words import STOP_WORDS
vectorizer_model = CountVectorizer(stop_words=list(STOP_WORDS))

# 3. Création et entraînement du modèle
umpa_model = UMAP(n_neighbors=15, n_components=5, metric='cosine', random_state=42)

topic_model = BERTopic(
    language="multilingual", 
    vectorizer_model=vectorizer_model,
    umap_model=umpa_model,
    verbose=True,
    calculate_probabilities=True,

)

topics, probs = topic_model.fit_transform(docs)

df['bertopic_id'] = topics

# Optionnel : Ajouter la probabilité du sujet dominant
import numpy as np
df['bertopic_probability'] = [json.dumps(p.tolist()) if p is not None else None for p in probs]
# 4. Voir les résultats
topic_model.get_topic_info().to_csv('outputs/topics_info.csv', index=False, encoding='utf-8')
df.to_csv('outputs/lda_bertopic_resultats_complet.csv', index=False, encoding='utf-8')

import os
import pandas as pd
from bertopic import BERTopic



# --- 2. Préparation du dossier de sortie ---
output_dir = 'outputs/graphs'
#os.makedirs(output_dir, exist_ok=True) # Crée le dossier s'il n'existe pas

print("Génération et sauvegarde des visualisations BERTopic...")

# ---------------------------------------------------------
# VISUALISATION 1 : La Carte des Distances Inter-Topics
# (Intertopic Distance Map)
# Montre la proximité s'émantique entre les thèmes.
# ---------------------------------------------------------
try:
    fig_distance = topic_model.visualize_topics()
    
    # Sauvegarde en HTML interactif (Recommandé)
    fig_distance.write_html(os.path.join(output_dir, 'bertopic_distance_map.html'))
    
    # Optionnel : Sauvegarde en image statique (PNG/SVG/PDF)
    # Nécessite l'installation de 'kaleido' : pip install kaleido
    # fig_distance.write_image(os.path.join(output_dir, 'bertopic_distance_map.png'))
    
    print(" - Carte des distances sauvegardée.")
except Exception as e:
    print(f"Erreur lors de la carte des distances : {e}")


# ---------------------------------------------------------
# VISUALISATION 2 : Le Nuage de Documents (UMAP 2D)
# Montre chaque tract comme un point, coloré par thème.
# ---------------------------------------------------------
try:
    # 'docs' doit être la liste des textes originaux utilisés pour le fit_transform
    fig_docs = topic_model.visualize_documents(docs, hide_annotations=True)
    
    # Sauvegarde en HTML interactif (Indispensable pour zoomer/survoler)
    fig_docs.write_html(os.path.join(output_dir, 'bertopic_documents_map.html'))
    
    print(" - Nuage de documents sauvegardé.")
except Exception as e:
    print(f"Erreur lors du nuage de documents : {e}")


# ---------------------------------------------------------
# VISUALISATION 3 : La Hiérarchie des Topics (Dendrogramme)
# Montre comment les thèmes se regroupent en super-catégories.
# ---------------------------------------------------------
try:
    fig_hierarchy = topic_model.visualize_hierarchy()
    
    fig_hierarchy.write_html(os.path.join(output_dir, 'bertopic_hierarchy.html'))
    
    print(" - Hiérarchie des topics sauvegardée.")
except Exception as e:
    print(f"Erreur lors de la hiérarchie : {e}")

print(f"\nSauvegarde terminée dans le dossier : {output_dir}")