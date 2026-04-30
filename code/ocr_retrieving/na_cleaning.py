import pandas as pd
import ast
import numpy as np

df_1988 = pd.read_csv('data/final_clean/archelect_1988_clean.csv')
df_1993 = pd.read_csv('data/final_clean/archelect_1993_clean.csv')

print(df_1988['cleaned_blocks'].isna().sum(), len(df_1988))
#print(df_1988['cleaned_blocks'][250])
print(df_1993['cleaned_blocks'].isna().sum(), len(df_1993))
#print(df_1993['cleaned_blocks'][250])

df_1993['cleaned_blocks'] = df_1993['cleaned_blocks'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
df_1993['cleaned_blocks'] = df_1993['cleaned_blocks'].apply(lambda x: x if len(x) > 0 else np.nan)
df_1988['cleaned_blocks'] = df_1988['cleaned_blocks'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
df_1988['cleaned_blocks'] = df_1988['cleaned_blocks'].apply(lambda x: x if len(x) > 0 else np.nan)  

df_1988 = df_1988.dropna(subset=['cleaned_blocks'])
df_1993 = df_1993.dropna(subset=['cleaned_blocks'])

df_1988['fulle_text'] = df_1988['cleaned_blocks'].apply(lambda blocks: " ".join(blocks))
df_1993['fulle_text'] = df_1993['cleaned_blocks'].apply(lambda blocks: " ".join(blocks))

df_1988.rename(columns={'fulle_text': 'full_text'}, inplace=True)
df_1993.rename(columns={'fulle_text': 'full_text'}, inplace=True)

df_1988['annee'] = 1988
df_1993['annee'] = 1993

df_full = pd.concat([df_1988, df_1993], ignore_index=True)

df_1988.to_csv('data/final_clean/archelect_1988_clean.csv', index=False, encoding='utf-8')
df_1993.to_csv('data/final_clean/archelect_1993_clean.csv', index=False, encoding='utf-8')
df_full.to_csv('data/final_clean/archelect_full_clean.csv', index=False, encoding='utf-8')

