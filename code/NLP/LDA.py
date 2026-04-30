import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
import spacy
from tqdm import tqdm
import ast
import numpy as np

tqdm.pandas()  # Pour ajouter une barre de progression aux opérations pandas

nlp = spacy.load('fr_core_news_sm')

def get_full_lda_distribution(model, corpus, num_topics):
    # Initialise une matrice vide (Documents x Topics)
    matrix = np.zeros((len(corpus), num_topics))
    
    for i, bow in enumerate(corpus):
        # minimum_probability=0 force Gensim à donner un score même s'il est de 0.0001
        doc_topics = model.get_document_topics(bow, minimum_probability=0)
        for topic_id, prob in doc_topics:
            matrix[i, topic_id] = prob
    return matrix



def preprocess_LDA(text):
    if pd.isna(text):
        return []
    
    doc = nlp(text)

    tokens = [token.lemma_ for token in doc 
              if token.pos_ in ['NOUN', 'ADJ', 'VERB'] 
              and not token.is_stop 
              and len(token.lemma_) > 2]
    return tokens

file = 'outputs/databases/lda_bertopic_resultats_complet.csv'
df = pd.read_csv(file)
"""df['processed_text'] = df['processed_text'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
)"""
# Prétraitement des documents
df['processed_text'] = df['full_text'].progress_apply(preprocess_LDA)
df.to_csv('outputs/databases/lda_bertopic_resultats_complet.csv', index=False, encoding='utf-8')

print(df['processed_text'].head())
# Créer le dictionnaire (liste de tous les mots uniques retenus)
id2word = corpora.Dictionary(df['processed_text'])

# Optionnel : Filtrer les mots trop rares ou trop fréquents
# Ex: mots présents dans < 5 documents ou > 50% du corpus
id2word.filter_extremes(no_below=5, no_above=0.6)

# Créer le Corpus : transformation des textes en "Bag of Words" (ID du mot, fréquence)
corpus = [id2word.doc2bow(text) for text in df['processed_text']]


lda_model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    num_topics=10, # À ajuster selon vos besoins
    random_state=42,
    update_every=1,
    chunksize=100,
    passes=10,
    alpha='auto',
    per_word_topics=True
)

topics_data = []
for i, topic in lda_model.show_topics(formatted=False, num_topics=10):
    words = ", ".join([w[0] for w in topic])
    topics_data.append({"topic_id": i, "words": words})

df_topics_lda = pd.DataFrame(topics_data)
df_topics_lda.to_csv('outputs/lda_definition_themes.csv', index=False)

def get_main_topic(bow):
    topic_probs = lda_model.get_document_topics(bow)
    # On trie pour avoir le topic avec la plus haute probabilité
    topic_probs.sort(key=lambda x: x[1], reverse=True)
    return topic_probs[0] if topic_probs else (None, 0)

# Appliquer pour chaque ligne
df['lda_results'] = [get_main_topic(c) for c in corpus]
df['lda_topic_id'] = df['lda_results'].apply(lambda x: x[0])
df['lda_topic_score'] = df['lda_results'].apply(lambda x: x[1])

lda_matrix = get_full_lda_distribution(lda_model, corpus, num_topics=10)

# 2. L'ajouter au DataFrame (une colonne par topic pour plus de clarté)
for i in range(10):
    df[f'lda_prob_topic_{i}'] = lda_matrix[:, i]

# 3. Créer votre score d'intensité pour le chômage (Topic 5)
df['lda_intensity_chomage'] = df['lda_prob_topic_5']

# Sauvegarde finale
df.drop(columns=['lda_results']).to_csv('outputs/databases/lda_bertopic_resultats_complet.csv', index=False)