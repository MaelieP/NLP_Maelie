import pandas as pd

df = pd.read_csv('data/taux_chom/id_codes_ts.csv', sep=',')

df['ID'] = '00' + df['ID'].astype(str)

#print(df.head())
# On cible les 4 dernières lignes ( -4: ) sur la colonne 'ID'
# .str[1:] permet de garder la chaîne à partir du deuxième caractère (index 1)
df.iloc[-4:, df.columns.get_loc('ID')] = df.iloc[-4:]['ID'].str[1:]

print(df.tail(6))
df.to_csv('data/taux_chom/id_codes_ts.csv', index=False, sep=',', encoding='utf-8')
