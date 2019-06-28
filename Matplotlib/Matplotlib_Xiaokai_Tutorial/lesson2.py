import pandas as pd

df = pd.read_csv('./data/000001.csv')

#print(df.head())

#print(df.columns)

#print(df.info())

#print(df.shape)

#print(df.iloc[0])

print(df.values.shape)

print(df.values)
