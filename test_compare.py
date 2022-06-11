import pandas as pd
import numpy as np
from tqdm import tqdm
from pandas.core.common import SettingWithCopyWarning
import warnings

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

df = pd.read_csv('test_data_with_score.csv', encoding='utf-8', index_col='Unnamed: 0')



for i in tqdm(np.arange(0.5,1,0.05)):
    col = str(i)[:4]
    df[col] = df['score'].map(lambda x: 1 if x < i else 0)

df.to_csv('test.csv', encoding='utf-8')



