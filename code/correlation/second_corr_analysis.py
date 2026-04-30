import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ast

# 1. Préparation des fonctions
def get_bertopic_sum(prob_list, target_ids):
    if prob_list is None or not isinstance(prob_list, (list, np.ndarray)):
        return 0
    return sum(prob_list[i] for i in target_ids if i < len(prob_list))

def map_political_orientation(df):
    col = df['titulaire-soutien'].str.lower().fillna('')
    conditions = [
        col.str.contains('parti socialiste|radicaux de gauche|verts'),
        col.str.contains('communiste|lutte ouvrière|parti ouvrier'),
        col.str.contains('rassemblement pour la république|union pour la démocratie française|nouveaux écologistes'),
        col.str.contains('front national|alliance populaire')
    ]
    choices = ['Gauche', 'Extrême Gauche', 'Droite', 'Extrême Droite']
    df['orientation'] = np.select(conditions, choices, default='Autre/Inconnu')
    return df

# 2. Chargement et Traitement
file = 'outputs/databases/lda_bertopic_resultats_complet.csv'
df = pd.read_csv(file)

# Conversion de la colonne probas (stockée en string dans le CSV)
df['bertopic_probability'] = df['bertopic_probability'].apply(ast.literal_eval)

# Calcul des intensités (Scores continus)
df['lda_intensity_chomage'] = df['lda_topic_score'].where(df['lda_topic_id'] == 5, 0.0)
target_topics = [3, 9, 12, 14, 15]
df['bertopic_intensity_chomage'] = df['bertopic_probability'].apply(lambda x: get_bertopic_sum(x, target_topics))

# Mapping politique
df = map_political_orientation(df)

# 3. Agrégation par département, orientation ET année
stats_geo_pol = df.groupby(['annee', 'departement', 'orientation']).agg({
    'taux_chom': 'mean',
    'lda_intensity_chomage': 'mean',
    'bertopic_intensity_chomage': 'mean'
}).reset_index()

# 4. Affichage des corrélations (Comparaison LDA vs BERTopic)
for annee in [1988, 1993]:
    print(f"\n--- ANALYSE CORRÉLATIONS {annee} ---")
    df_year = stats_geo_pol[stats_geo_pol['annee'] == annee]
    for pol in ['Extrême Gauche', 'Gauche', 'Droite', 'Extrême Droite']:
        sub = df_year[df_year['orientation'] == pol]
        if not sub.empty:
            c_lda = sub[['taux_chom', 'lda_intensity_chomage']].corr().iloc[0, 1]
            c_bert = sub[['taux_chom', 'bertopic_intensity_chomage']].corr().iloc[0, 1]
            print(f"[{pol}] LDA: {c_lda:.3f} | BERTopic: {c_bert:.3f}")

# 5. Visualisation : Comparaison des deux années (BERTopic)
sns.set_theme(style="whitegrid")
g = sns.lmplot(
    data=stats_geo_pol[stats_geo_pol['orientation'] != 'Autre/Inconnu'],
    x='taux_chom', 
    y='bertopic_intensity_chomage', 
    hue='orientation',
    col='annee', # Un graphique pour 1988, un pour 1993
    palette='muted',
    height=5, aspect=1.2,
    scatter_kws={'alpha':0.4}
)

# Ajustements cosmétiques
g.set_axis_labels("Taux de chômage réel (%)", "Intensité thématique (BERTopic)")
g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("Réactivité du discours au chômage : 1988 vs 1993", fontsize=16)

plt.show()