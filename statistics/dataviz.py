import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_xy_axes_manual(x, y_vals):
    for y in y_vals:
        plt.axes([...])
        plot_basic_xy(x, y)


def plot_xy_subplot(x, y_vals):
    """Plot multiple figures in different rows, stacked atop e/o"""
    for i, y in enumerate(y_vals):
        # args: nrows, ncols, which subplot to activate
        #       subplot number calculated by finding position within grid of
        #       subplots where top left subplot has number 1 and increases
        #       to the right and down
        plot.subplot(len(y_vals), 1, i)
        plot_basic_xy(x, y)


def plot_categorical_histogram():
    sns.countplot(x='foo', y='bar', data=df)


def plot_2d_rect_histogram():
    # x and y have same length
    # specify bins along each axis
    plt.hist2d(x, y, bins=(10, 20))
    plt.colorbar()

    # display a grid where cells are colored according to # observations
    # in a given (x,y) coordinate cell
    # is a pseudocolor plot similar to pcolor
    plt.show()


def plot_2d_hex_histogram():
    # x and y have same length
    # specify bins along each axis
    plt.hexbin(x, y, gridsize=(10, 20))
    plt.colorbar()

    # display a grid where cells are colored according to # observations
    # in a given (x,y) coordinate cell
    # is a pseudocolor plot similar to pcolor
    plt.show()


def plot_linear_regression():
    """Plot a simple linear regression between 2 variables
    simple linear regression b/c has 1 independent variable
    """

    # plot data points as scatterplot
    # add a line for linear regression
    # add cone around linear regression for 95% confidence interval
    sns.lmplot(x='weight', y='hp', data=auto)

    # to group by the value of the categorical variable
    # origin and plot a different regression per group
    # on the same figure specify hue='origin'
    sns.lmplot(x='weight', y='hp', hue='origin', data=auto, palette='Set1')

    # to have a separate subplot per row
    # to group by the value of the categorical variable
    # origin and plot a different regression per group
    # in a different subplot each on a separate row
    # specify row='origin'
    sns.lmplot(x='hp', y='weight', data=auto, row='origin')

    plt.show()


def plot_residuals_linear_regression():
    """Plot the residuals for a simple linear regression
    between 2 variables.
    """
    sns.residplot(x='weight', y='hp', data=auto, color='green')
    plt.show()


def plot_2nd_order_linear_reg():
    """Plots a second order linear regression

    i.e. function of the curve is of polynomial order 2

    instead of `y = slope * x + intercept`
    we get `y = slope_1 * x_1 + slope_2 + x_2 + intercept`
    """
    # use regplot with order=2
    sns.regplot(
        x='weight', y='mpg', data=auto, scatter=None, color='green', order=2)
    plt.show()

    # coefs = np.polyfit(x, y, 2)
    # y_new = np.polyval(coefs,x)
    # plt.plot(x, y_new)


##############################################################################
# Visualizing multivariate data
##############################################################################


def plot_jointplot():
    # plot scatterplot of hp against mpg
    # and histograms off the top and right sides for the individual variables
    sns.jointplot(x='hp', y='mpg', data=auto)

    # change the central plot type using kind
    sns.jointplot(x='hp', y='mpg', data=auto, kind='hex')


def plot_pairplot():
    """Pairplots plot a 2x2 matrix of subplots with scatterplots and histograms
    """
    sns.pairplot(auto)

    # to add a regression line and color observations by their origin value
    sns.pairplot(auto, hue='origin', kind='reg')


def plot_covariance_matrix():
    """Plot a heatmap of the covariances between all variables in a dataset
    Recall that covariances are a measure of how variables values change together
    with a range of values between -1 and 1 indicating anticorrelation to fully
    positive correlation
    """
    cov_matrix = df.cov()
    sns.heatmap(cov_matrix)

    # heatmap of correlation between variables
    sns.heatmap(df.corr(), square=True, cmap='RdYlGn')
