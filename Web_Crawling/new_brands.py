import pandas as pd
from tqdm import tqdm

# dfmomo = pd.read_csv("other_momo.csv", encoding='utf-8', index_col='Unnamed: 0').fillna(0)[['name','momo']]
# df_shopee = pd.read_csv("new_shopee.csv", encoding='utf-8', index_col='Unnamed: 0').fillna(0)[['shopee']]
# # df_shopee_2 = pd.read_csv("other_shopee_2.csv", encoding='utf-8', index_col='Unnamed: 0').fillna(0)
# # df_shopee = pd.concat([df_shopee_1, df_shopee_2], axis= 0, ignore_index=False)[['shopee']]
# df_human = pd.read_excel('孤單的品牌.xlsx', index_col="Column1").fillna(0)[['human']]
# df_human.columns

# new = dfmomo
# new = new.merge(df_shopee, how='outer', left_index=True, right_index=True).fillna(0)
# new = new.merge(df_human, how='outer', left_index=True, right_index=True).fillna(0)
# new['new_brands'] = None

# for i in tqdm(new.index):
#     if new['momo'][i] != 0:
#         new['new_brands'][i] = new['momo'][i]
#     elif new['shopee'][i] != 0:
#         new['new_brands'][i] = new['shopee'][i]
#     elif new['human'][i] != 0:
#         new['new_brands'][i] = new['human'][i]
#     else:
#          new['new_brands'][i] = 0

    
# new.to_csv('new_brands_v2.csv', encoding='utf-8')


df = pd.read_excel('new_brands_v3.xlsx', index_col="Column1").fillna(0)
df.to_csv('new_brands_v3.csv', encoding='utf-8')

