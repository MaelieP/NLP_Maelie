import pandas as pd
import numpy as np
import ast



def get_total_chomage_intensity(prob_row, target_ids=[3, 9, 12, 14, 15]):
    # Si BERTopic a généré des probabilités pour chaque document
    if isinstance(prob_row, (list, np.ndarray)):
        # On fait la somme des probas pour nos 5 thèmes cibles
        # On vérifie que l'index existe dans la liste pour éviter les erreurs
        return sum(prob_row[i] for i in target_ids if i < len(prob_row))
    return 0.0

# On applique cela à TOUTES les lignes du DataFrame
file = 'outputs/lda_bertopic_resultats_complet.csv'
df = pd.read_csv(file, dtype={'lda_topic_id': 'int64', 'bertopic_id': 'int64'})
df['bertopic_probability'] = df['bertopic_probability'].apply(ast.literal_eval)


# Pour le LDA (comparaison simple)
df['lda_mention_chomage'] = np.where(df['lda_topic_id'] == 5, 1, 0)
df = df[df['annee'] == 1993]

# Pour BERTopic (comparaison avec une liste)
df['bertopic_mention_chomage'] = np.where(df['bertopic_id'].isin([3, 9, 12, 14, 15]), 1, 0)

df['lda_score_chomage'] = np.where(df['lda_topic_id'] == 5, df['lda_topic_score'], 0.0)
target_topics = [3, 9, 12, 14, 15]
#df['bertopic_score_chomage'] = df['bertopic_probability'].apply(lambda x: get_bertopic_sum(x, target_topics))
df['bertopic_intensity_chomage'] = df['bertopic_probability'].apply(get_total_chomage_intensity)


df.to_csv('outputs/databases/lda_bertopic_resultats_complet.csv', index=False, encoding='utf-8')