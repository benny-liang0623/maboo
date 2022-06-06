import pandas as pd
from sklearn import metrics

df = pd.read_csv("Brand/資料/false.csv", encoding="utf-8")
print(df.head(1))

print(metrics.classification_report(df['Actual Tags'],df['Predicted Tags']))