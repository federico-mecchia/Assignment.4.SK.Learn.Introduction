"""
"Assignment 4 - SK Learn Introduction" - Part 2
Iris Dataset
"""


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from itertools import islice


def load_iris_data():

    iris_data_raw = load_iris()
    iris_data = iris_data_raw.data
    iris_target = iris_data_raw.target
    iris_data_df = pd.DataFrame(
        iris_data,
        columns=["Petal_length", "Petal_Width", "Sepal_Length", "Sepal_Width"],
    )
    iris_target_df = pd.DataFrame(iris_target, columns=["Species"])
    return iris_data_df, iris_target_df, iris_data, iris_target


def plot_elbow_and_prediction(range_value, iris_data, iris_target):

    Error = []
    for i in islice(range_value, 0, None, 1):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit_predict(iris_data, iris_target)
        Error.append(kmeans.inertia_)
        plot_cluster_graph_step(iris_data, kmeans.labels_, i)

    fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 5))
    ax.plot(range_value, Error)
    ax.set_title("Elbow plot")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Error value")
    plt.show()


def plot_cluster_graph_step(iris_data, iris_target, i):

    fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(10, 5))
    ax[0].scatter(iris_data[:, 0], iris_data[:, 1], c=iris_target, cmap="gist_rainbow")
    ax[0].set_title("Petal_Width/Petal_length")
    ax[0].set_xlabel("Petal_length")
    ax[0].set_ylabel("Petal_Width")
    fig.suptitle("Number {} cluster ".format(i))
    ax[1].scatter(iris_data[:, 2], iris_data[:, 3], c=iris_target, cmap="gist_rainbow")
    ax[1].set_title("Sepal_Width/Sepal_length")
    ax[1].set_xlabel("Sepal_length")
    ax[1].set_ylabel("Sepal_Width")


def plot_original_graph_(iris_data, iris_target):

    fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(10, 5))
    ax[0].scatter(iris_data[:, 0], iris_data[:, 1], c=iris_target, cmap="gist_rainbow")
    ax[0].set_title("Petal_Width/Petal_length")
    ax[0].set_xlabel("Petal_length")
    ax[0].set_ylabel("Petal_Width")
    fig.suptitle("Original")
    ax[1].scatter(iris_data[:, 2], iris_data[:, 3], c=iris_target, cmap="gist_rainbow")
    ax[1].set_title("Sepal_Width/Sepal_length")
    ax[1].set_xlabel("Sepal_length")
    ax[1].set_ylabel("Sepal_Width")


if __name__ == "__main__":

    iris_data_df, iris_target_df, iris_data, iris_target = load_iris_data()
    plot_original_graph_(iris_data, iris_target)
    range_value = range(2, 10)
    plot_elbow_and_prediction(range_value, iris_data, iris_target)


"""

Comment

In order to solve the second part of "Assignment 4 - SK Learn
Introduction" (the part regarding the Iris dataset) first I import "pandas"
as "pd", "matplotlib.pyplot" as "plt", "KMeans" from "sklearn.cluster",
"load_iris" from "sklearn.datasets" and "islice" from "itertools".
First I define "load_iris_data()". To this regard, I set "iris_data_raw"
equal to "load_iris()" in order to consider the Iris dataset, I set
"iris_data" equal to "iris_data_raw.data" and then I also set "iris_target"
equal to "iris_data_raw.target". Moreover, I set "iris_data_df" equal to
"pd.DataFrame()" and inside the brackets I include "iris_data" and I also
set "columns" equal to "["Petal_length", "Petal_Width", "Sepal_Length",
"Sepal_Width"]". In addition to this, I also set "iris_target_df" equal to
"pd.DataFrame()" and inside the brackets I include "iris_target" and I also
set "columns" equal to "["Species"]". Finally I use "return" with regard
to "iris_data_df", "iris_target_df", "iris_data" and "iris_target".
Moreover, I also define "plot_elbow_and_prediction" and inside the brackets
I include "range_value", "iris_data" and "iris_target". To this regard,
first I set "Error" equal to "[]" and then I create a "for loop". In the
"for loop" first I set "for i in islice(range_value, 0, None, 1):" and then
I also include "kmeans" equal to "KMeans(n_clusters=i)",
"kmeans.fit_predict(iris_data, iris_target)",
"Error.append(kmeans.inertia_)" and finally I also include
"plot_cluster_graph_step(iris_data, kmeans.labels_, i)". In addition to this,
I then set "fig, ax" equal to
"plt.subplots(constrained_layout=True, figsize=(10, 5))" and I also include
"ax.plot(range_value, Error)", "ax.set_title("Elbow plot")",
"ax.set_xlabel("Number of clusters")" and "ax.set_ylabel("Error value")".
Finally, I also include "plt.show()".
Furthermore, I also define
"plot_cluster_graph_step(iris_data, iris_target, i):". To this regard, I set
"fig, ax" equal to
"plt.subplots(1, 2, constrained_layout=True, figsize=(10, 5))" and then I
include "ax[0].scatter(iris_data[:, 0], iris_data[:, 1], c=iris_target,
cmap="gist_rainbow")", "ax[0].set_title("Petal_Width/Petal_length")",
"ax[0].set_xlabel("Petal_length")", "ax[0].set_ylabel("Petal_Width")",
"fig.suptitle("Number {} cluster ".format(i))",
"ax[1].scatter(iris_data[:, 2], iris_data[:, 3],
c=iris_target, cmap="gist_rainbow")",
"ax[1].set_title("Sepal_Width/Sepal_length")",
"ax[1].set_xlabel("Sepal_length")" and also
"ax[1].set_ylabel("Sepal_Width")".
Furthermore, I also define "plot_original_graph_(iris_data, iris_target):".
To this regard, I set "fig, ax" equal to
"plt.subplots(1, 2, constrained_layout=True, figsize=(10, 5))" and then I
also include "ax[0].scatter(iris_data[:, 0], iris_data[:, 1],
c=iris_target, cmap="gist_rainbow")",
"ax[0].set_title("Petal_Width/Petal_length")",
"ax[0].set_xlabel("Petal_length")", "ax[0].set_ylabel("Petal_Width")",
"fig.suptitle("Original")", "ax[1].scatter(iris_data[:, 2],
iris_data[:, 3], c=iris_target, cmap="gist_rainbow")",
"ax[1].set_title("Sepal_Width/Sepal_length")",
"ax[1].set_xlabel("Sepal_length")" and "ax[1].set_ylabel("Sepal_Width")".
Lastly, I also include "if __name__ == "__main__":" and I set
"iris_data_df, iris_target_df, iris_data, iris_target" equal to
"load_iris_data()", I include
"plot_original_graph_(iris_data, iris_target)", I set "range_value"
equal to "range(2, 10)" and finally I include
"plot_elbow_and_prediction(range_value, iris_data, iris_target)".
Lastly, I type "black" in the "Terminal" followed by the path of the file
in ".py format" in order to format the whole code contained in the file
taken into consideration (basically the code of the file you are reading
and so the code of the file in ".py format" named
"Assignment.4.SK.Learn.Introduction.Part.2").


Overall, several graphs are created. To this regard each graph includes
two graphs, one with "Petal_Width" on the y-axis and "Petal_length" on
the x-axis and the other one with "Sepal_Width" on the y-axis and
"Sepal_length" on the x-axis. In addition to this, also a graph regarding
the "Elbow Method" is created.
From the various graphs, it is possible to observe that 3 is the best
number of clusters. This can also be observed by taking into consideration
the graph of the "Elbow Method", which clearly displays the fact that 3 is
the best number of clusters.
In conclusion, from the various analyses and graphs it is possible to state
that 3 is the best number of clusters.

"""
