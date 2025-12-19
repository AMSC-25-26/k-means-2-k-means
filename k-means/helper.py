
from pandas import DataFrame, read_csv

iris_dataset: DataFrame = read_csv("data/dataset-iris.csv")
iris_clusters: DataFrame = iris_dataset[iris_dataset.columns[-1]]
iris_clusters.to_csv("data/iris-clusters.csv", index=False)

wine_dataset: DataFrame = read_csv("data/dataset-wine.csv", sep=";")
wine_clusters: DataFrame = wine_dataset[wine_dataset.columns[-1]]
wine_clusters.to_csv("data/wine-clusters.csv", index=False)