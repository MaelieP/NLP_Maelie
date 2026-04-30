import pandas as pd
import os

df = pd.read_csv('data/taux_chom/id_codes_ts_with_means.csv', sep=',', dtype={'ID': str})
df_text = pd.read_csv('data/final_clean/archelect_full_clean.csv', sep=',', dtype={'departement': str})

df['code_departement'] = df['Région'].str[:2]
df = df.dropna(subset=['mean_1988', 'mean_1993'])

df_text_1988 = df_text[df_text['annee'] == 1988]
df_text_1993 = df_text[df_text['annee'] == 1993]

df_text_1988['taux_chom'] = df_text_1988['departement'].map(df.set_index('code_departement')['mean_1988'])
df_text_1993['taux_chom'] = df_text_1993['departement'].map(df.set_index('code_departement')['mean_1993'])

df_full = pd.concat([df_text_1988, df_text_1993], ignore_index=True).dropna(subset=['taux_chom'])

print(df_full.columns.tolist())
if 'full_text.1' in df_full.columns:
    df_full = df_full.drop(columns=['full_text.1'])


os.makedirs('data/full', exist_ok=True)
df_full.to_csv('data/full/archelect_full_clean_with_chom.csv', index=False, encoding='utf-8')
df.to_csv('data/taux_chom/id_codes_ts_final.csv', index=False, sep=',', encoding='utf-8')