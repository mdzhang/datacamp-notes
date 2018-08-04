# Basic Descriptive Statistics

- **exploratory data analysis (EDA)**: process of organizing, processing, and summarizing a data set
    - see John Tukey

## Quantitative EDA

- **mean**
    - heavily influenced by outliers
    - `mu = np.mean(...)`

- **median**: middle value, 50th percentile
    - `np.median`
    - `np.percentile(df, [50])`

- **IQR**: between 25th and 75th percentiles

- **outlier**: no strict definition
    - > 2 IQRs from the median

- **variance**: mean squared distance of data from the mean
    - used to measure spread of data
    - `np.var`

- **standard deviation**: square root of the variance
    - `signma = np.std(...)`

- **cumulative distribution function (ECDF)**
    - y-axis ranges from 0-1
    - increasing left to right, smooth
    - `p = (x1, y1)` indicates proportion (y1) of observations with a value `<= x1`

    ```python
    def ecdf(data):
      """Compute ECDF for a one-dimensional array of measurements."""

      # Number of data points: n
      n = len(data)

      # x-data for the ECDF: x
      x = np.sort(data)

      # y-data for the ECDF: y
      y = np.arange(1, n + 1) / n

      return x, y
    ```

### Relationships in data

- **covariance**: measure of how two quantities vary together
    - variability due to codepedence
    - e.g. Obama vote share and total vote count per county
    - look at given point p = (x1, y1) distance from x mean and y mean
        - if distances both positive, positively related (positive correlation)
        - if one distance is above mean, the other below => negatively related (anticorrelation)
    - `np.cov`
        - gives a covariance matrix where `[0, 0]` is variance of data for `x`, `[1,1]` for `y`, and `[1,0] == [0,1]`

- **correlation coefficient**: measures strength/lack of relationship between 2 variables
  - what is it
      - variability due to codepedence / independent variability
      - range [-1, 1] where 1 indicates perfect correlation
      - if close to zero, assume values not correlated
      - if close to +/- 1, indicates strong positive/negative relationship
  - when relationship is linear, use  aka **Pearson R**: covariance / std(x) * std(y)
      - aka **pearson correlation coefficient**
      - use `scipy.stats.stats.pearsonr` e.g. `pearsonr(x, y)`
      - `np.corrcoef`

      ```python
      def pearson_r(x, y):
        """Compute Pearson correlation coefficient between two arrays."""
        # Compute correlation matrix: corr_mat
        corr_mat = np.corrcoef(x, y)

        # Return entry [0,1]
        return corr_mat[0,1]
      ```

      - `pd.Series.corr`
  - when relationship is non-linear, use **Spearman rank** or **Kendall Tau**
      - use `scipy.stats.stats.spearmanr`, `scipy.stats.stats.kendalltau`
  - trending datasets may seem correlated even when unrelated
    - should look at percent change: `df['prices'].pct_change()`

- **autocorrelation**: correlation between a time series and a delayed copy of itself
    - what is it
      - e.g. autocorrelation of order 3 returns correlation between `[t1, t2, t3]` and `[t4, t5, t6]`
      - aka **autocovariates** or **serial correlation**
      - normally refers to **lag-one autocorrelation** unless otherwise specified
      - negative autocorrelation aka **mean reverting**
      - positive autocorrelation aka **momentum**, **trend following**
    - why use it
      - find repetitive patterns or periodic signal in time series
      - uncorrelated means previous values in series won't tell you much about later values in the series
      - negatively autocorrelated means previous values in series likely indicate a drop in future values

- **partial autocorrelation**: like autocorrelation, but removes effects of prvious time points (???)

- **autocorrelation function (ACF)**: yields autocorrelation as a function of lag; 1 at lag 0
    - how to interpret graph
      - if values close to 0, values between consecutive observations are not correlated w/ e/o
      - values close to +/- 1, indicate strong positive/negative relationship
      - shaded region indicates confidence interval/margins of uncertainty; if values _outside_ shaded region, relationships are statistically significant
        - alpha = 0.05: 5% chance that if true autocorrelation is 0, it will fall outside of shaded area
        - alpha = 1: no bands on plot
        - smaller confidence bands if alpha is lower or have fewer observations
        - approximate confidence interval of `2 / sqrt(# observations)`

    ```python
    from statsmodels.graphics import tsaplots

    # plot the autocorrelation function (acf) of time series data (tsa)
    # alpha specifies width of confidence interval
    fig = tsaplots.plot_acf(co2_levels['co2'], lags=40, alpha=0.05)
    fig2 = tsaplots.plot_pacf(co2_levels['co2'], lags=40)

    plt.show()

    df['return'].autocorr()
    ```

- **white noise**: series w/ constant mean, constant variance, zero autocorrelations at all lags
    - if data has Gaussian/normal distribution, aka **Gaussian white noise**
    - generate w/ e.g. `noise = np.random.normal(loc=0, scale=1, size=500)` where `loc` is mean, `scale` is std dev
- **random walk**: `p_today = p_yesterday + noise`
  - **random walk with drift**: `p_today = p_yesterday + noise (epsilon) + drift (myu)`
    - drift by `myu` every period

  ```python
  # simulate a random walk
  steps = np.random.normal(loc=0, scale=1, size=500)
  steps[0] = 0
  P = 100 + np.cumsum(steps)
  ```
    - regression test for random walk: `p_today = alpha + beta * p_yesterday + epsilon`
      - null hypothesis that series is a random walk: `beta == 1`
      - alternate hypothesis: `beta < 1` (reject null hypothesis)
  - equivalent to **Dickey-Fuller test**: `p_today - p_yesterday = alpha + beta * p_yesterday + epsilon`
      - null hypothesis that series is a random walk: `beta == 0`
      - alternate hypothesis: `beta < 0` (reject null hypothesis)
      - if you add lags to right hand side (`p_yesterday`, `p_day_before_yesterday`), is aka the **augmented Dickey-Fueller test**
      - use `statsmodels.tsa.stattools.adfuller` e.g. `test_stat, p_value, *more = adfuller(series)`

- **stationarity**
  - **strong stationarity**: distribution of data is time-invariant
  - **weak stationarity**: mean, variance, autocorrelation of data are time-invariant
  - parsimonious model (few parameters) needs to have relatively stationary data
  - can transform non-stationary into stationary series by taking differences
    - can take **first differences**
    - can take differences with lag corresponding to periodicity

## Graphical EDA

- graphical EDA makes it easier to see outliers and trends that can/should influence later analysis and prevent misinterpretation of data

### Plots

- **histogram**: shows frequency that values within a specific interval occur
  - counts realizations/observations within each bin (**binning**); shows counts within each bin
  - **square root rule**: choose # bins for a histogram that is ~= sqrt(# observations)
      - help avoid **binning bias**: when same data interpreted differently when # bins is different

    ```python
    # numpy
    arr = np.array([4.7, 4.5, 4.9])
    counts, bins, patches = plt.hist(arr, bins=np.sqrt(len(arr)))
    ```

    ```python
    # pandas
    df['alcohol'].plot.hist()
    ```

    ```python
    # seaborn
    sns.distplot(df['alcohol'], kde=False, bins=10)
    ```

- **rug plot**: adds perpendicular hashmarks along an axis to indicate actual data points; often in conjunction w/ KDE or histogram

    ```python
    sns.distplot(df['alcohol'], hist=False, rug=True, bins=10)
    ```

- **scatterplot**: plot point for each observation

    ```python
    # x, y
    plt.plot(total_votes/100, dem_share, marker='.', linestyle='none')
    ```

    - with **regression line**

      ```python
      # more low-level
      sns.regplot(x='alcohol', y='pH', data=df)

      # support polynomial function fit of order 2
      sns.regplot(x='alcohol', y='pH', data=df, order=2)

      # for categorical vars, all points in same category along same vertical axis
      # reduce jitter to better see distribution
      sns.regplot(x='mnth', y='total_rentals', data=df, order=2, x_jitter=0.1)
      # instead of jitter, just squash points using estimator to summarize values
      # plot means with confidence intervals
      sns.regplot(x='mnth', y='total_rentals', data=df, order=2, x_estimator=np.mean)

      # linear model plot is more powerful
      sns.lmplot(x='alcohol', y='pH', data=df)

      # change color of scatterplot points based on value of 'type' column
      sns.lmplot(x='alcohol', y='pH', data=df, hue='type')

      # plot each different 'type' in its own subplot in its own column
      sns.lmplot(x='alcohol', y='pH', data=df, col='type')

      # plot each different 'type' in its own subplot in its own row
      sns.lmplot(x='alcohol', y='pH', data=df, row='type')
      ```

      - **facetting**: plotting multiple graphs by changing a single variable

- **boxplot**: shows min and max (at whiskers, or 1.5 IQR, whichever smaller), median at middle box line, 25th and 75th percentiles at box bottom and top, and outliers

    ```python
    # with seaborn
    sns.boxplot(x='state', y='dem_share', data=df)

    # with pandas
    df.boxplot('day', 'tip')
    ```

- **strip plot**: plots all points in a univariate dataset along a vertical access
    - if there are many observations of similar values, will wil stip atop e/o
    - can add horizontal **jitter** so points form a cloud

    ```python
    sns.stripplot(x='day', y='tip', data=tip, jitter=True, size=4)
    ```

- **bee swarm plot**
    - unaffected by **binning bias**
    - like a strip plot, but automatically adds jitter to overlapping points
        - uses algorithm to determine how to show points so they don't overlap
    - not ideal when you have many observations that start to overlap each other in the plot - doesn't sccale well

    ```python
    sns.swarmplot(x='state', y='dem_share', data=df)
    ```

- **violin plot**
    - like a box plot, but curves along the sides of a violin plot indicate density of the distribution
    - curves are **kernel density plots (KDEs)**

   ```python
    # to disable inner box plot
    sns.violinplot(x='day', y='tip', data=tip, inner=False)
   ```

- **lvplot**: letter-value plot
    - faster to render than violin plot
    - scales better than strip/swarm plot

   ```python
   sns.lvplot(x='day', y='tip', data=tip)
   ```

- **countplot**: shows # observations at each value; unlike histogram, no binning

- **barplot**: bars used to plot some estimate of the data (e.g. average); black lines extending from bars indicate confidence interval

   ```python
   sns.barplot(x='day', y='tip', data=tip)
   ```

- **pointplot**: like barplot, shows summary measure and confidence interval
    - lines drawn between points help visualize how values change across categories

   ```python
   sns.pointplot(x='day', y='tip', data=tip)
   ```

- **residplot**: for evaluating the fit of a model, graphs residuals

    ```python
    sns.residplot(data=df, x='temp', y='total_rentals')
    ```

- **matrix plots** need data to be in matrix, e.g. as returned by the below
    - must be in **tidy** format i.e. one observation per row of data, columns are variables

    ```python
    df = pd.crosstab(df['mnth'], df['weekday'], values=df['total_rentals'], aggfunc='mean')

    # to get a correlation matrix
    # can also use methods: 'spearman', 'kendalltau'
    corr_mat = df[['col1', 'col2', 'col3']].corr(method='pearson')
    ```

- **trellis plot** / **lattice plot**
    - combine many subplots (small multiples) into larger visualization useful for identifying trends in data w/ many variables
    - see interactions across different columns of data

    - **faceting**: compare multiple plots side by side by using same scale and axis
      - grid has same plot in each cell, but for different subselections of data

      ```python
      # define facets
      # each column plot corresponds to different values of 'HIGHDEG'
      g = sns.FacetGrid(df, col='HIGHDEG')

      # map plot type
      # each column plot will be a boxplot of 'Tuition' values
      # order we want plots to be displayed in
      g.map(sns.boxplot, 'Tuition', order=['1', '2', '3', '4'])

      # instead of col, order can use row, row_order
      ```

      ```python
      # equivalent to above
      sns.factorplot(x='Tuition', data=df, col='HIGHDEG', kind='box')
      ```

      - `lmplot` plots scatter + regression plots on a `FacetGrid`

      ```python
      sns.lmplot(data=df, x='Tuition', y='SAT_AVG_ALL', col='HIGHDEG', fit_reg=False)

      sns.lmplot(data=df,
              x='SAT_AVG_ALL',
              y='Tuition',
              col="Ownership",
              row='Degree_Type',
              row_order=['Graduate', 'Bachelors'],
              hue='WOMENONLY',
              col_order=inst_ord)
      ```

    - different types of plots in each cell of grid

      ```python
      g = sns.PairGrid(df, vars=['Fair_Mrkt_Rent', 'Median_Income'])

      # all plots are this type of plot
      g = g.map(plt.scatter)

      # histograms on the main diagonal (top left to bottom right)
      g = g.map_diag(plt.hist)
      # scatter plots on the anti diagonal (top right to bottom left)
      g = g.map_offdiag(plt.scatter)
      ```

      ```python
      sns.pairplot(df, vars=['Fair_Mrkt_Rent', 'Median_Income'], kind='reg',
                   diag_kind='hist')

      sns.pairplot(data=df,
          x_vars=["fatal_collisions_speeding", "fatal_collisions_alc"],
          y_vars=['premiums', 'insurance_losses'],
          # antidiagonal
          kind='scatter',
          # main diagonal
          diag_kind='kde',
          hue='Region',
          palette='husl')
      ```

    - compare distribution of data between 2 vars; has main plot and **marginal plots**

      ```python
      g = sns.JointGrid(data=df, x='Tuition', y='ADM_RATE_ALL')

      # scatter plot w/ reg line in center, distribution curve plot in margins
      g.plot(sns.regplot, sns.distplot)

      # kde plot in the center (looks like a contour)
      g = g.plot_join(sns.kdeplot)
      # kde plots in marginals
      g = g.plot_marginals(sns.kdeplot, shade=True)
      # add pearson r value in corner
      g = g.annotate(stats.pearsonr)
      ```

      ```python
      sns.jointplot(data=df, x='Tuition', y='ADM_RATE_ALL', kind='hex')
      ```

- **heatmap**
  ```python
  # df from matrix plots above
    sns.heatmap(df)

    # turn on annotations in cells, format annotations as numbers
    # specify color map to use
    # don't show colorbar; increase width of lines between cells
    sns.heatmap(df, annot=True, fmd='d', cmap='', cbar=False, linewidths=.5)

    # reorders row/columns of matrix so that more similar columns are closer to
    # e/o - makes map easier to interprete
    sns.clustermap(corr_mat)
    ```

- **meshgrid**: graph of a matrix of values where each cell if filled with a color whose density is proportional to the value in that cell
    - often used with a **colorbar** which displays a gradient of colors and the numerical values associated with colors along the gradient

    ```python
    A = np.array([[1, 0, -1], [2, 0, 1], [1, 1, 1]])
    # cmap specifies a colormap
    plt.pcolor(A, cmap='Blues')
    # colorbar shows a gradient of colors and shows values per gradient
    plt.colorbar()
    ```

- **contour**: draws concentric shapes each which represent an area of values in the matrix with the same colors; like meshgrids, values should be a matrix

    ```python
    # just graph lines of contours
    np.contour(Z)
    # more granular lines
    np.contour(X, Y, Z, 30)

    # fill in the contours with color
    np.contourf(Z)
    ```


### Annotations

- visualization's **aesthetics** include e.g. layouts, labels, colors
- visualizations should minimize extraneous markings so that the data speaks for itself

##### With `matplotlib`

  ```python
  import matplotlib.pyplot as plt
  import seaborn as sns
  import numpy as np

  # use a built in theme
  plt.style.use('ggplot')

  # or use seaborn styling
  sns.set()

  # add axis labels`
  plt.xlabel('Date')
  plt.ylabel('Temperature')

  # add graph title
  plt.title('Dew point')

  # only plot values with x in range (1947, 1957)
  plt.xlim((1947, 1957))
  # ...and y in range (0, 1000)
  plt.ylim((0, 1000))

  # alternatively we could have just
  plt.axis((1947, 1957, 0, 1000))

  # add padding so no data points overlap the edge of the graph i.e. edges are farther out
  plt.margins(0.02)

  # xy indicates coordinates of what to point to
  # xytext indicates starting coordinates of arrow (so it'll point down and to the left)
  # specify `arrowprops` to make it an arrow, and make its color black
  plt.annotate(
      'Maximum',
      xy=(yr_max, cs_max),
      xytext=(yr_max + 5, cs_max + 5),
      arrowprops=dict(facecolor='black'))

  # the label arg is used to indicate value for figure in legend
  plt.plot(year, computer_science, color='red', label='Computer Science')
  plt.legend(loc='lower right')

  # draw and display graph
  plt.show()

  # save figure to a file
  # autodetect file type from name
  plt.savefig('xlim_and_ylim.png')
  ```

##### With `seaborn`

- determing colors

  ```python
  import seaborn as sns

  # set default 'darkgrid' style
  sns.set()

  # use matplotlib color codes
  sns.set(color_codes=True)
  sns.distplot(df['Tuition'], color='g')

  # iterate through default color palettes
  for p in sns.palettes.SEABORN_PALETTES:
    sns.set_palette(p)
    sns.distplot(df['Tuition'])

  # to get current palette
  p = sns.color_palette()

  # to plot first 6 colors of a palette
  sns.palplot(p, 6)

- customizing via axes
-
  ```python
  fig, ax = plt.subplots()

  sns.distplot(df['Tuition'], ax=ax)

  # add x-axis label and limit min/max x value
  ax.set(xlabel='Tuition 2013-14', xlim=(0, 70000))
  # add a vertical dotted line to the plot
  ax.axvline(x=20000, label='My budget', linestyle='-')

  # show a legend (e.g. for axvline)
  ax.legend()

  # remove axis ticks
  sns.despine(left=True)
  ```

- types of color palettes
  - **circular / categorical**: use when data unordered
  - **sequential**: use when data has consistent range from high to low (e.g. for heatmaps)
  - **diverging**: use when data high & low values are interesting
