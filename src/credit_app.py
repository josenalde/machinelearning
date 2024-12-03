import requests
import pandas as pd

df = pd.read_csv('datasets/Credit.csv')

# forma manual de label encoder
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype('category').cat.codes

# apenas 10 linhas para batch de predição
X = df.iloc[0:10, 0:20].to_json(orient='split')

y_pred = requests.post(url='http://localhost:2345/invocations',
                       headers={'Content-Type': 'application/json'}, data=X)

print(y_pred)

print(y_pred.text)
