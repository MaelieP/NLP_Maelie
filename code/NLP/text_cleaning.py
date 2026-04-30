import pandas as pd
import spacy
import re
import fasttext
import os
from tqdm import tqdm

# --- INITIALISATION ---
# Chargement de SpaCy
try:
    nlp = spacy.load('fr_core_news_sm')
except OSError:
    os.system("python -m spacy download fr_core_news_sm")
    nlp = spacy.load('fr_core_news_sm')

# Chargement de FastText (assurez-vous que le fichier est dans le bon dossier)
lang_model = fasttext.load_model("models/lid.176.ftz")

def is_french(text, threshold=0.6):
    """Vérifie si le texte est du français via FastText."""
    if not text.strip():
        return False
    # On nettoie les retours à la ligne pour la détection
    predictions = lang_model.predict(text.replace('\n', ' '))
    lang = predictions[0][0].replace('__label__', '')
    score = predictions[1][0]
    return lang == 'fr' and score >= threshold

def get_clean_french_blocks_final(raw_text):
    """
    Fusionne le pré-filtrage statistique, la détection de langue 
    et la validation par mots-outils.
    """
    if pd.isna(raw_text):
        return []

    # 1. Découpage en blocs (au moins deux sauts de ligne)
    blocks = re.split(r'\n\s*\n+', str(raw_text))
    
    final_blocks = []
    
    for block in blocks:
        block_stripped = block.strip()
        
        # --- FILTRE 1 : Longueur et Nettoyage de base ---
        if len(block_stripped) < 30:
            continue
            
        # --- FILTRE 2 : Ratio de caractères alphabétiques ---
        # On s'assure que ce n'est pas une ligne de symboles ou de chiffres
        alphas = re.findall(r'[a-zA-ZÀ-ÿ]', block_stripped)
        alpha_ratio = len(alphas) / len(block_stripped)
        if alpha_ratio < 0.65:
            continue

        # --- FILTRE 3 : Détection de langue (FastText) ---
        if not is_french(block_stripped):
            continue

        # --- FILTRE 4 : Densité de mots-outils (SpaCy) ---
        # On nettoie les sauts de ligne internes avant l'analyse finale
        clean_content = " ".join(block_stripped.split())
        doc = nlp(clean_content)
        
        stopwords_count = sum(1 for token in doc if token.is_stop)
        stopword_ratio = stopwords_count / len(doc) if len(doc) > 0 else 0
        
        # Validation finale : le bloc doit avoir une structure de phrase (mots-outils)
        if stopword_ratio > 0.15: # Au moins 15% de mots-outils pour du texte suivi
            final_blocks.append(clean_content)
            
    return final_blocks

# --- APPLICATION SUR LE DATAFRAME ---

df = pd.read_csv('data/final/final_dirty/archelect_1993_complet.csv')


tqdm.pandas(desc="Nettoyage des textes")

# On crée une nouvelle colonne avec la liste des blocs propres
df['cleaned_blocks'] = df['text'].progress_apply(get_clean_french_blocks_final)

df.to_csv('data/final/final_clean/archelect_1993_cleaned.csv', index=False, encoding='utf-8')