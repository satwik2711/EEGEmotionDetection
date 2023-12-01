import pandas as pd
data = pd.read_csv("features.csv")
data.pop('label')
data.to_csv("features.csv", index=False)