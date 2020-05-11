"""
"Assignment 4 - SK Learn Introduction" - Part 1
Boston Dataset
"""


from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_graph(boston_get_data, feature_name, target_name):

    boston_plot_data_df = pd.DataFrame(
        boston_get_data.data, columns=boston_get_data.feature_names
    )
    boston_plot_data_df[target_name] = boston_get_data.target
    sns.pairplot(
        boston_plot_data_df,
        x_vars=[feature_name],
        y_vars=target_name,
        height=7,
        aspect=0.7,
        kind="reg",
    )
    plt.title("Slope of the factor with the largest effect")
    plt.show()


def make_regression_boston_data_and_plot():

    boston_get_data = load_boston()
    x_boston_dataframe = pd.DataFrame(
        boston_get_data.data, columns=boston_get_data.feature_names
    )
    y_boston_dataframe = pd.DataFrame(boston_get_data.target, columns=["MEDV"])
    linear_regr = LinearRegression()
    linear_regr.fit(x_boston_dataframe, y_boston_dataframe)
    coeff_df = pd.DataFrame(
        linear_regr.coef_.T, x_boston_dataframe.columns, columns=["Coefficient"]
    )
    print(coeff_df)
    max_coeff_index = coeff_df.abs()["Coefficient"].idxmax()
    max_abs_coeff_value = coeff_df.abs().loc[max_coeff_index, "Coefficient"]
    print(
        "The factor which has the largest effect on the price of housing in Boston is:"
        + max_coeff_index
    )
    print(
        "The absolute value of the coefficient of the factor which has the largest effect is: {}".format(
            max_abs_coeff_value
        )
    )
    plot_graph(boston_get_data, max_coeff_index, "MEDV")


if __name__ == "__main__":

    make_regression_boston_data_and_plot()


"""

Comment

In order to solve the first part of "Assignment 4 - SK Learn
Introduction" (the part regarding the Boston dataset) first I
import "LinearRegression" from "sklearn.linear_model", "load_boston"
from "sklearn.datasets", "pandas" as "pd", "seaborn" as "sns" and
"matplotlib.pyplot" as "plt".
Then I define the first function, named "plot_graph", and I include
in the brackets "boston_get_data", "feature_name" and "target_name".
Moreover, I set "boston_plot_data_df" equal to "pd.DataFrame()" and I
also include in the brackets "boston_get_data.data" and I set
"columns" equal to "boston_get_data.feature_names". In addition to
this, I also set "boston_plot_data_df[target_name]" equal to
"boston_get_data.target".
Furthermore, I also include "sns.pairplot" and inside the brackets
I include "boston_plot_data_df", "x_vars=[feature_name]",
"y_vars=target_name" and I also set "height" equal to "7", "aspect"
equal to "0.7" and "kind" equal to "reg".
Moreover, I also include "plt.title()" and inside the brackets I also
include the title of the plot, which is "Slope of the factor with the
largest effect". Lastly, I also include "plt.show()".
Furthermore, I also define "make_regression_boston_data_and_plot()". To
this regard, first I set "boston_get_data" equal to "load_boston()" in
order to take into consideration the data of the Boston dataset. Moreover,
I set "x_boston_dataframe" equal to "pd.DataFrame()" and inside the
brackets I include "boston_get_data.data" and I also set "columns" equal
to "boston_get_data.feature_names". In addition to this, I also set
"y_boston_dataframe" equal to "pd.DataFrame()" and inside the brackets I
include "boston_get_data.target" and I also set "columns" equal to
"["MEDV"]". I then set "linear_regr" equal to "LinearRegression()" and I
also include "linear_regr.fit()" and inside the brackets I include
"x_boston_dataframe" and "y_boston_dataframe".
Moreover, I set "coeff_df" equal to "pd.DataFrame()" and inside the brackets
I include "linear_regr.coef_.T", "x_boston_dataframe.columns" and I also
set "columns" equal to "["Coefficient"]". I then use "print()" to print
the coefficients.
Furthermore, I set "max_coeff_index" equal to
"coeff_df.abs()["Coefficient"].idxmax()" and also "max_abs_coeff_value"
equal to "coeff_df.abs().loc[max_coeff_index, "Coefficient"]".
Moreover, I use "print()" to print the factor which has the largest effect
on the price of housing in Boston and then also to print the absolute
value of the coefficient of the factor which has the largest effect on the
price of housing in Boston. I then also include "plot_graph()" and inside
the brackets I include "boston_get_data", "max_coeff_index" and ""MEDV"".
Finally, I also include "if __name__ == "__main__":" and
"make_regression_boston_data_and_plot()".
Lastly, I type "black" in the "Terminal" followed by the path of the file
in ".py format" in order to format the whole code contained in the file
taken into consideration (basically the code of the file you are reading
and so the code of the file in ".py format" named
"Assignment.4.SK.Learn.Introduction.Part.1").


From the results obtained by running this code it is possible to state
that the factor that has the largest effect on the price of housing in
Boston is "NOX" and also that the absolute value of its coefficient is
17.766611228299986.

"""
