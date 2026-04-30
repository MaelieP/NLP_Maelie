import pandas as pd
import spacy
import re
import fasttext
import os
from tqdm import tqdm

# --- INITIALISATION ---
try:
    nlp = spacy.load('fr_core_news_sm')
except OSError:
    os.system("python -m spacy download fr_core_news_sm")
    nlp = spacy.load('fr_core_news_sm')

lang_model = fasttext.load_model("models/lid.176.ftz")

def is_french(text, threshold=0.6):
    if not text.strip():
        return False
    predictions = lang_model.predict(text.replace('\n', ' '))
    lang = predictions[0][0].replace('__label__', '')
    score = float(predictions[1][0])
    return lang == 'fr' and score >= threshold

def get_clean_french_blocks_final(raw_text):
    if pd.isna(raw_text):
        return []

    blocks = re.split(r'\n\s*\n+', str(raw_text))
    final_blocks = []
    
    for block in blocks:
        block_stripped = block.strip()
        
        if len(block_stripped) < 30:
            continue
            
        alphas = re.findall(r'[a-zA-ZÀ-ÿ]', block_stripped)
        alpha_ratio = len(alphas) / len(block_stripped)
        if alpha_ratio < 0.65:
            continue

        if not is_french(block_stripped):
            continue

        clean_content = " ".join(block_stripped.split())
        doc = nlp(clean_content)
        
        stopwords_count = sum(1 for token in doc if token.is_stop)
        stopword_ratio = stopwords_count / len(doc) if len(doc) > 0 else 0
        
        if stopword_ratio > 0.15:
            final_blocks.append(clean_content)
            
    return final_blocks

# --- APPLICATION SUR LES DEUX ANNÉES ---
for annee in ['1988', '1993']:
    input_path = f'data/final_dirty/archelect_{annee}_complet.csv'
    output_path = f'data/final_clean/archelect_{annee}_clean.csv'  # sans 'd'
    
    print(f"\n--- Traitement {annee} ---")
    df = pd.read_csv(input_path)
    
    tqdm.pandas(desc=f"Nettoyage des textes {annee}")
    df['cleaned_blocks'] = df['text'].progress_apply(get_clean_french_blocks_final)
    
    os.makedirs('data/final_clean', exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Sauvegardé : {output_path}")
