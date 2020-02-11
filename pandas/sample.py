import pandas as pd
from sklearn import datasets

pd.set_option('display.max_rows', 150)
iris = datasets.load_iris()
df1 = pd.DataFrame(iris['data'])
print(df1)

pd.set_option('display.max_columns', 4096)

olivetti_faces = datasets.fetch_olivetti_faces()
df2 = pd.DataFrame(olivetti_faces['data']).head()

print(df2)

