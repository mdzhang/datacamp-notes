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
## Graphical EDA

- graphical EDA makes it easier to see outliers and trends that can/should influence later analysis and prevent misinterpretation of data

### Plots

- **histogram**: shows frequency that values within a specific interval occur
  - counts realizations/observations within each bin (**binning**); shows counts within each bin
  - **square root rule**: choose # bins for a histogram that is ~= sqrt(# observations)
      - help avoid **binning bias**: when same data interpreted differently when # bins is different

    ```python
    arr = np.array([4.7, 4.5, 4.9])
    counts, bins, patches = plt.hist(arr, bins=np.sqrt(len(arr)))
    ```

- **scatterplot**: plot point for each observation

    ```python
    # x, y
    plt.plot(total_votes/100, dem_share, marker='.', linestyle='none')
    ```

- **boxplot**: shows min and max (at whiskers, or 1.5 IQR, whichever smaller), median at middle box line, 25th and 75th percentiles at box bottom and top, and outliers

    ```python
    # with seaborn
    sns.boxplot(x='state', y='dem_share', data=df)

    # with pandas
    df.boxplot('day', 'tip')
    ```

- **strip plot**: plots points in a univariate dataset along a vertical access
    - if there are many observations of similar values, will wil stip atop e/o
    - can add horizontal **jitter** so points form a cloud

    ```python
    sns.stripplot(x='day', y='tip', data=tip, jitter=True, size=4)
    ```

- **bee swarm plot**
    - unaffected by **binning bias**
    - like a strip plot, but automatically adds jitter to overlapping points
    - not ideal when you have many observations that start to overlap each other in the plot

    ```python
    sns.swarmplot(x='state', y='dem_share', data=df)
    ```

- **violin plot**
    - like a box plot, but curves along the sides of a violin plot indicate density of the distribution

   ```python
    # to disable inner box plot
    sns.violinplot(x='day', y='tip', data=tip, inner=False)
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
