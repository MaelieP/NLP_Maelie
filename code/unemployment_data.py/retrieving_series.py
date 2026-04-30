import pandas as pd
import pynsee
from tqdm import tqdm

df_id = pd.read_csv('data/taux_chom/id_codes_ts.csv', sep=',', dtype={'ID': str})

#print(str(df_id['ID'].tolist()))

id_list = df_id['ID'].tolist()

list_data_1988 = []
list_data_1993 = []

for id in tqdm(id_list, desc = "Processing IDs"):
    data = pynsee.get_series(id)
    data_1988 = data[data['TIME_PERIOD'].str.contains('1988')]
    data_1993 = data[data['TIME_PERIOD'].str.contains('1993')]
    mean_1988 = data_1988['OBS_VALUE'].mean()
    mean_1993 = data_1993['OBS_VALUE'].mean()
    list_data_1988.append(mean_1988)
    list_data_1993.append(mean_1993)

df_id['mean_1988'] = list_data_1988
df_id['mean_1993'] = list_data_1993

df_id.to_csv('data/taux_chom/id_codes_ts_with_means.csv', index=False, sep=',', encoding='utf-8')
