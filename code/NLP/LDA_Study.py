import pandas as pd

df_lda = pd.read_csv('outputs/databases/lda_resultats_complet.csv', sep=',', encoding='utf-8')

print(df_lda['id'].value_counts())