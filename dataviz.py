import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_basic_xy():
    """Plot a simple graph of x against y values"""
    # draw plot in memory
    plt.plot(x, y, color='red')

    # add axis labels`
    plt.xlabel('Date')
    plt.ylabel('Temperature')

    # add graph title
    plt.title('Dew point')

    # draw and display graph
    plt.show()


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


def plot_zoom_xy(x, y):
    """Plot a graph, zoomed in to focus on a specific range of x and y values"""
    # draw plot in memory
    plt.plot(x, y, color='red')

    # add axis labels`
    plt.xlabel('Year')
    plt.ylabel('GDP')

    # add graph title
    plt.title('US GDP')

    # only plot values with x in range (1947, 1957)
    plt.xlim((1947, 1957))
    # ...and y in range (0, 1000)
    plt.ylim((0, 1000))

    # alternatively we could have just
    plt.axis((1947, 1957, 0, 1000))

    # draw and display graph
    plt.show()


def plot_and_save(x, y, name):
    plot_basic_xy(x, y)

    # save figure to a file
    # autodetect file type from name
    plt.savefig('xlim_and_ylim.png')


def plot_with_legend():
    # the label arg is used to indicate value for figure in legend
    plt.plot(year, computer_science, color='red', label='Computer Science')
    plt.plot(year, physical_sciences, color='blue', label='Physical Sciences')

    plt.legend(loc='lower right')


def plot_with_annotation():
    plt.plot(year, computer_science, color='red', label='Computer Science')
    plt.plot(year, physical_sciences, color='blue', label='Physical Sciences')
    plt.legend(loc='lower right')

    # max cs enrollment
    cs_max = computer_science.max()
    # year of max cs enrolmment
    yr_max = year[computer_science.argmax()]

    # xy indicates coordinates of what to point to
    # xytext indicates starting coordinates of arrow (so it'll point down and to the left)
    # specify `arrowprops` to make it an arrow, and make its color black
    plt.annotate(
        'Maximum',
        xy=(yr_max, cs_max),
        xytext=(yr_max + 5, cs_max + 5),
        arrowprops=dict(facecolor='black'))

    plt.xlabel('Year')
    plt.ylabel('Enrollment (%)')
    plt.title('Undergraduate enrollment of women')
    plt.show()


def plot_stylish():
    """Change themes before plotting"""
    # use a built in theme
    plt.style.use('ggplot')

    plot_basic_xy()


def plot_meshgrid():
    """Plot a matrix of values so that each 'cell' is filled with a color
    whose density is proportional to the value in that position in the matrix
    """

    # create an array of 41 values evenly spaced between [-2, 2]
    u = np.linspace(-2, 2, 41)
    # create an array of 21 values evenly spaced between [-1, 1]
    v = np.linspace(-1, 1, 21)

    # get back 2 matrices
    # X has the values of u, repeated v times
    # Y is a list of lists, where a list at a given index i has the value
    #   u[i] repeated len(v) times
    X, Y = np.meshgrid(u, v)

    Z = np.sin(3 * np.sqrt(X**2 + Y**2))

    plt.pcolor(Z)
    plt.show()

    # recall when plotting images that element [0,0] in a matrix (i.e. in top-left)
    # is plotted in bottom left


def plot_simple_meshgrid():
    A = np.array([[1, 0, -1], [2, 0, 1], [1, 1, 1]])
    # cmap specifies a colormap
    plt.pcolor(A, cmap='Blues')
    # colorbar shows a gradient of colors and shows values per gradient
    plt.colorbar()
    plt.show()


def plot_contour():
    # as an alternative to `pcolor` you can also plot points with contour
    np.contour(Z)
    #
    # contours draw concentric shapes each of which represent an area of
    # values in the matrix with the same values
    #
    # contours can be just lines (`contour`) or filled in (`contourf`)
    np.contourf(Z)
    # contours take meshgrids as args
    # by default, the x and y axes will have ticks with values inferred
    # from the values in the meshgrid itself
    # but these can be overriden
    np.contour(X, Y, Z)

    # to increase number of contours, specify level
    np.contour(X, Y, Z, 30)


def plot_1d_histogram():
    """Plot a histogram, the fundamental tool for visualizing 1D data

    Choose intervals (bins)
    Count realizations/observations within each bin (binning)
    Histogram shows counts in each bin
    """

    counts, bins, patches = plt.hist(x, bins=25)
    plt.show()


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


def plot_stripplot():
    """Plot a strip plot, suitable for univariate data"""

    # by default repeat values will sit atop of eachother
    sns.stripplot(x='day', y='tip', data=tip)

    # add extra horizontal 'jitter' so they form more of a cloud
    # swarmplot will automatically add jitter to avoid overlapping points
    # specify smaller point size
    sns.stripplot(x='day', y='tip', data=tip, jitter=True, size=4)


def plot_swarmplot():
    # plot a swarmplot
    sns.swarmplot(x='day', y='tip', data=tip)

    # plot a swarmplot with different colors based on
    # the value of the sex column
    sns.swarmplot(x='day', y='tip', data=tip, hue='sex')

    # rotate the above swarmplot
    sns.swarmplot(x='tip', y='day', data=tip, hue='sex', orient='h')


def plot_violin_plot():
    """Plot a violin plot, suitable for univariate data

    Like a histogram, but curves in a violin plot indicate
    density of distribution, and has inner box plot
    """
    sns.violinplot(x='day', y='tip', data=tip)

    # to disable inner box plot
    sns.violinplot(x='day', y='tip', data=tip, inner=False)


def plot_boxplot():
    df.boxplot('day', 'tip')


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
