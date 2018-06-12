# Pandas & Time Series Data

## Time Series Data

- properties
  - **seasonality**: does the data show a periodic pattern?
      - e.g. temperature has daily seasonality (colder at night than during the day) and monthly seasonality (warmer in summer than winter)
      - should be a fixed, known period
  - **trend**: does data follow consistent up/downwards slope?
  - **noise** aka **residual**: outliers & missing values inconsistent w/ rest of data

  ```python
  import statsmodels.api as sm

  decomposition = sm.tsa.seasonal_decompose(co2_levels)

  # extract decomposition attributes to dataframes
  observed = decomposition.observed
  trend = decomposition.trend
  residuals = decomposition.resid
  seasonality = decomposition.seasonal
  ```


## Reading in data

```python
# read csv and specify name to use as index column
df = pd.read_csv(filename, index_col='date', parse_dates=True)

# (or) given a dataframe, set index to the date column
time_format = '%Y-%m-%d %H:%M'
df['date'] = pd.to_datetime(df['date'], format=time_format)
df = df.set_index('date')
```

## Visualizing data

```python
import matplotlib.pyplot as plt

# Use the fivethirtyeight style on any matplot plots
plt.style.use('fivethirtyeight')

# use pandas dataframe plot method to wrap matplot
# (for timeseries data, will automatically populate e.g. x-ticks)
# returns an AxesSubplot to enable adding additional annotations etc.
ax = df.plot(color='blue')

# add annotations
ax.set_xlabel('Date')
ax.set_ylabel('Number of great df')
ax.set_title('Something')

# to add a vertical line at a specific point
ax.axvline('1945-01-01', linestyle='--')
ax.axhline(4, color='green')

# to add a shaded to area between specific points
ax.axvspan('1936-01-01', '1950-01-01', color='red' , alpha=0.5)
ax.axhspan(6, 8, color='green' , alpha=0.3)
```

## Slicing time series data

```python
# can slice using YYYY-MM-DD formatted strings
df['1960':'1970']
df['1960-01':'1970-12']
df['1960-01-01':'1960-01-15']
df.loc['2010-10-11 21:00:00':'2010-10-11 22:00:00']
```

## Cleaning time series data

```python
# finding missing values
#   return records with a null value in any column
df.isnull()
df.notnull().sum()

# imputing missing values
#   fill empty value with value of next valid observation in same column
df.fillna(method='bfill')
#   fill empty value with value of last valid observation in same column
df.fillna(method='ffill')
```

## Analyzing data

- **rolling means** aka **moving averages**
   - what is it
     - given a series of data, take average of different subsets of data
     - each subsequent subset excludes the head of the previous subset and adding the values following the tail of the previous subset
   - why use it
    - generally used to smooth data e.g. smooth short term fluctuations
    - remove outliers
    - highlight long term cycles or trends

```python
# suppose you have 100 observations, a window of 10, and your aggregation function is mean
# rolling means will look at observations 1-10, and find the mean
# then it will look at observations 2-11, and fine the mean
# etc. until it runs through all observations, returning the resulting dataframe
smoothed = unsmoothed.rolling(window=24).mean()
```

- aggregating data

```python
# can pull list of months of time series data points w/ co2_levels.index.month
# use this to summarize data at e.g. a monthly/annual/weekly/daily/hourly basis
co2_levels.groupby(co2_levels.index.month).mean()
```

- resampling data
  - **down sampling**: sample using same intervals, less frequent
      - must indicate how to aggregate values (e.g. `mean`, `std`, `var`, etc.)
  - **upsampling**: e.g. daily -> hourly
      - must indicate how to fill in missing data

```python
# (optional) integer + char to indicate sampling period

# aggregate with mean
df1 = df.resample('6h').mean()  # 1 entry per 6 hrs
df2 = df.resample('D').mean()  # 1 entry per day

# use linear interpolation
df3 = df.resample('A').first().interpolate('linear')
```

- **autocorrelation**: correlation between a time series and a delayed copy of itself
    - what is it
      - e.g. autocorrelation of order 3 returns correlation between `[t1, t2, t3]` and `[t4, t5, t6]`
      - aka **autocovariates**
    - why use it
      - find repetitive patterns or periodic signal in time series
    - how to interpret graph
      - if values close to 0, values between consecutive observations are not correlated w/ e/o
      - if close to zero, assume values not correlated
      - if close to +/- 1, indicates strong positive/negative relationship
      - shaded region indicates confidence interval/margins of uncertainty; if values _outside_ shaded region, relationships are statistically significant
- **partial autocorrelation**: like autocorrelation, but removes effects of prvious time points (???)

```python
from statsmodels.graphics import tsaplots

# plot the autocorrelation function (acf) of time series data (tsa)
fig = tsaplots.plot_acf(co2_levels['co2'], lags=40)
fig2 = tsaplots.plot_pacf(co2_levels['co2'], lags=40)

plt.show()
```
