"""
Notes from Datacamp's Pandas Foundations course
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import zscore

# get indices of rows whose values in the 'species' column of the data frame are 'setosa'
indices = iris['species'] == 'setosa'
# select rows at the given indices, select all columns for each row
setosa = iris.loc[indices, :]
# confirm all same value
setosa['species'].unique()
# delete column
del setosa['species']

# calculate error
# indicates how misleading it is to calculate statistics over entire population vs over a factor
# factor aka categorical variable
describe_all = iris.describe()
describe_setosa = setosa.describe()
error_setosa = (100 * np.abs(c - describe_all)) / describe_all

# display the box plots on 3 separate rows and 1 column
fig, axes = plt.subplots(nrows=3, ncols=1)

# box plot for each species; y axis is sepal length
iris.loc[iris['species'] == 'setosa'].plot(
    ax=axes[0], y='sepal_length', kind='box')
iris.loc[iris['species'] == 'virginica'].plot(
    ax=axes[1], y='sepal_length', kind='box')
iris.loc[iris['species'] == 'versasomething'].plot(
    ax=axes[2], y='sepal_length', kind='box')

# display the plot
plt.show()

######################################################################
# Apply built in functions to entire dataframe/series
#   - use built in dataframe/series vectorized functions
#   - else use numpy ufuncs (universal functions)
#   - else use .apply()
######################################################################

# strip whitespace from column names
df.columns = df.columns.str.strip()

# get a series of boolean values indicating whether the observation at the
# same index had the substring 'DAL' in its value
dallas = df['Destination Airport'].str.contains('DAL')

times_tz_none = pd.to_datetime(df['Date (MM/DD/YYYY)'] + ' ' + df[
    'Wheels-off Time'])
# localize the time to US/Central
times_tz_central = times_tz_none.dt.tz_localize('US/Central')
# convert the datetimes from US/Central to US/Pacific
times_tz_pacific = times_tz_central.dt.tz_convert('US/Pacific')

# else...

# apply func to each cell
df.apply(lambda n: n / 2)

######################################################################
# Visualizing data
######################################################################

# by default, has the df's index on the x axis and range of values for
# all numerical columns on the y axis
df.plot()

# can set the index in place to a column of datetimes
# having a datetime as index will automatically prettify the x-axis
df.Date = pd.to_datetime(df.Date)
df.set_index('Date', inplace=True)

######################################################################
# Reading in data
######################################################################

df = pd.read_csv('data.csv')
# for csvs with no header row
df = pd.read_csv('data.csv', header=None)

# read in data, specify column to use as index
df = pd.read_csv('data.csv', index_col='county')

# to set column names
df.columns = ['foo', 'bar', 'baz']

# to set column type
df.foo.astype(str)
# using categorical vars saves space; values stored as small ints
# separate lookup table to find actual string vals
df.bar.astype('category')  # categorical var
df.bar = pd.Categorical(values = df.bar, categories=['foo', 'bar', 'baz'], ordered=True)
df.bar = pd.as_numeric(df.bar)
df.bar = pd.as_datetime(df.baz)

######################################################################
# Filtering down data
######################################################################

# to drop columns
df2 = df.drop(col_names, axis='columns')

# use boolean filter condition to filter election df down to
# just those rows where the value of the turnout column was greater than 70
high_turnout_df = election[election.turnout > 70]

# use them to broadcast assignments
election.winner[election.margin < 1] = np.nan

# filter to companies with total # units purchased > 35
sales.groupby('Company').filter(lambda g:g['Units'].sum() > 35)


######################################################################
# Filtering down data (dropping null values)
######################################################################

# drop all rows in the data frame where any individual value is null
df.dropna(how='any')

# drop all rows in the data frame where all values across all columns are null
df.dropna(how='all')

######################################################################
# Selecting data
######################################################################

# to select column, row at index w/ value 'index_value'; get a scalar
df['col_name']['index_value']

# if row index is the ordered #/position of the observation (e.g. 0...n) then
# the following selects the given column, and the rows at 0-based indices 1, 2, and 3
df['col_name'][1:4]
df['col_name']['index_value1':'index_value2']

# select all rows for column
df['col_name'][:]

# select all columns for limited rows
df[:][1:4]

# to select column; get a series
# series is a 1d array with a labelled index
df.col_name

# to select multiple columns, get a dataframe
# dataframe is 2d array with series for columns, common row labels/indices
df[['col_name1', 'col_name2']]

# .loc uses labels
df.loc['index_value', 'col_name']
# some rows, all columns, using slices
df.loc['index_value1':'index_value2', :]
# some rows, all columns, using lists
df.loc[['index_value1', 'index_value2'], :]
# some rows (in reverse), all columns
df.loc['index_value1':'index_value2':-1, :]

# for multiindexes

idx = pd.IndexSlice

#                          Total
#         Country
#  bronze France           475.0
#         Germany          454.0
#         Soviet Union     584.0
#         United Kingdom   505.0
#         United States   1052.0
#  gold   Germany          407.0
#         Italy            460.0
#         Soviet Union     838.0
#         United Kingdom   498.0
#         United States   2088.0
#  silver France           461.0
#         Italy            394.0
#         Soviet Union     627.0
#         United Kingdom   591.0
#         United States   1195.0

# select all outermost idx values, but on inner idx value of 'United Kingdom'
medals_sorted.loc[idx[:,'United Kingdom'], :]

#                         Total
#         Country
#  bronze United Kingdom  505.0
#  silver United Kingdom  591.0
#  gold   United Kingdom  498.0


# .iloc uses indices

######################################################################
# Transforming data
######################################################################

red_vs_blue = {'Obama': 'blue', 'Romney': 'red'}

# Use the dictionary to map the 'winner' column to the new column: election['color']
election['color'] = election['winner'].map(red_vs_blue)

######################################################################
# Sorting data
######################################################################

# to sort by index in ascending (small to large) order
df.sort_index()
df.sort_index(ascending=False)

# to sort by the values in the col 'col_name'
df.sort_values('col_name')

######################################################################
# Pandas core concepts
#   - series is a 1d array with a labelled index
#   - dataframe is 2d array with series for columns each sharing common indexes
#   - index is a sequence of labels, immutable, homogeneous type; privileged column
#####################################################################

# if columns are related, can label them
df.columns.name = 'MONTHS'

# can assign index name
df.index.name = 'Zip Code'

# index can be a tuple (like a SQL compound primary key) aka MultiIndex, HierarchicalIndex
sales = df.set_index(['state', 'month'])
# sort_index() to enable more powerful slicing
sales = sales.sort_index()

# before sort_index
#                  visitors  signups
#  city   weekday
#  Austin Mon           326        3
#  Dallas Mon           456        5
#  Austin Sun           139        7
#  Dallas Sun           237       12

# after sort_index
#                  visitors  signups
#  city   weekday
#  Austin Mon           326        3
#         Sun           139        7
#  Dallas Mon           456        5
#         Sun           237       12

# limit to value NY in outer index state, and value 1 in inner index month
NY_month1 = sales.loc[('NY', 1)]

# limit to value CA or TX in outer index state, and value 2 in inner index month
CA_TX_month2 = sales.loc[(['CA', 'TX'], 2), :]

# limit to any value in outer index state, and value 2 in inner index month
# use slice(None) in place of :, which isn't compatible w/ multiindex
all_month2 = sales.loc[(slice(None), 2), :]

# like indexes, columns can be multi index
# for a df with a multi index
df.unstack(level=1)
df.unstack(level='<inner_index_name>')
# moves the inner index from the multiindex to an inner index on a single column
# so the column becomes a hierarchical column
# df.stack does the inverse, moving a hierchical column's inner index to the df's index, making it a multi index
# to reorder inner/outer indices do e.g.
df.swaplevel(0, 1)

# users
#                  visitors  signups
#  city   weekday
#  Austin Mon           326        3
#         Sun           139        7
#  Dallas Mon           456        5
#         Sun           237       12
#

# users.unstack(level='weekday')
#          visitors      signups
#  weekday      Mon  Sun     Mon Sun
#  city
#  Austin       326  139       3   7
#  Dallas       456  237       5  12

######################################################################
# Reshaping data: pivoting
######################################################################

#  suppose users defined as
#    weekday    city  visitors  signups
#  0     Sun  Austin       139        7
#  1     Sun  Dallas       237       12
#  2     Mon  Austin       326        3
#  3     Mon  Dallas       456        5

visitors_pivot = users.pivot(
    index='weekday', columns='city', values='visitors')

#  city     Austin  Dallas
#  weekday
#  Mon         326     456
#  Sun         139     237

# see that the index takes values from the weekday column,
#     that each value in the city column because a column header
#     and that the cells take their values from the visitors column

# =====================================================================

# use pivot tables when your index-column combinations have duplicate values

# pivot more_trials such that
#    the index values are the values of the treatment column
#    the column headers are the values of the gender column
#    aggregate duplicates by counting them
more_trials.pivot_table(
    index='treatment', columns='gender', values='response', aggfunc='count')

######################################################################
# Reshaping data: melting
######################################################################

# df
#  city     Austin  Dallas
#  weekday
#  Mon         326     456
#  Sun         139     237

# id_vars: columns to leave alone
# value_name: name of new column built from cell values (default: value)
# everything else gets rolled into a new column
#    that default takes its name from columns.name
#    else defaults to 'variable'
df2 = pd.melt(df, id_vars=['weekday'], value_name='visitors')

#  df2
#    weekday    city  visitors
#  0     Mon  Austin       326
#  1     Sun  Austin       139
#  2     Mon  Dallas       456
#  3     Sun  Dallas       237

# =====================================================================

#    weekday    city  visitors  signups
#  0     Sun  Austin       139        7
#  1     Sun  Dallas       237       12
#  2     Mon  Austin       326        3
#  3     Mon  Dallas       456        5

df2 = pd.melt(df, id_vars=['city', 'weekday'])

#       city weekday  variable  value
#  0  Austin     Sun  visitors    139
#  1  Dallas     Sun  visitors    237
#  2  Austin     Mon  visitors    326
#  3  Dallas     Mon  visitors    456
#  4  Austin     Sun   signups      7
#  5  Dallas     Sun   signups     12
#  6  Austin     Mon   signups      3
#  7  Dallas     Mon   signups      5

######################################################################
# Reshaping data: groupby
######################################################################

# like SQL group by
# then select fields to aggregate on
sales.groupby(customers)[['bread', 'butter']]

# follow up with an aggregation function
# aggregation/reduction functions reduce a series to a scalar
sales.groupby(customers)[['bread', 'butter']].sum()

# or multiple aggregation functions to be applied to each column
# to be stored as nested columns
sales.groupby(customers)[['bread', 'butter']].agg(['max', 'sum'])

# or specify specific agg func per column
sales.groupby(customers)[['bread', 'butter']].agg({
    'bread': max, 'butter': sum})

# can alternatively follow up with a transformation function
# transformation functions are applied element-wise i.e. to each element
# in a series
auto.groupby('yr')['mpg'].transform(zscore)

# if a df has a multiindex you want to group by, specify the levels
sales.groupby(level=['State'])

# can use one series to group another if they have a shared index
# in this case, Country
life_by_region = life.groupby(regions['region'])

# find out what fraction of children under 10 survived in each 'pclass'
# create boolean series to be used for indexing; use map to give custom index labels
under10 = (titanic['age'] < 10).map({True:'under 10', False:'over 10'})
survived_mean_2 = titanic.groupby([under10, 'pclass'])['survived'].mean()

######################################################################
# Inspecting data
######################################################################

# get cols, index and their types, # non-null entries
df.info()

df.categorical_var.unique()  # unique values
df.categorical_var.value_counts()  # frequency
df.numerical_var.{mean,median,max,min,count,first,last}()

######################################################################
# Statistics
######################################################################

# summary statistics for all numerical columns
df.describe()
df.mean()
df.median()
df.max()
df.min()
df.count()
# quartile takes any number 0-1; 0.5 == median
df.quartile(0.5)

#  z-score is the number of standard deviations by which an observation is above the mean

# print the Pearson correlation coefficient aka Person's r
# Pearson's r ranges from -1 (indicating total negative linear correlation) to 1 (indicating total positive linear correlation)
# will print a n x n matrix for a df with n columns, indicating the value of r for each cross
df.corr()

# percentage variation: min/max expressed as percentage of mean aka min/mean * 100
# percent change: (current val - previous val) / percent val
#                 just do df.pct_change()

######################################################################
# Merging data
######################################################################

# can use index of a more limited dataset to find common indices with another dataset
# 1881 had fewer names
common_names = names_1981.reindex(names_1881.index)

# concat stacks series vertically (i.e. additional rows) when axis='rows'/axis=0 provided
#        stacks series horizontally (i.e. additional cols) when axis='columns'/axis=1 provided
pd.concat([foo, bar, baz], axis='rows')

# same as
df2 = foo.append(bar).append(baz).reset_index(drop=True)

# when concatenating dataframes with same column/index names, use keys to create a multilevel index
medals = pd.concat(medals, keys=['bronze', 'silver', 'gold'])

#                          Total
#         Country
#  bronze United States   1052.0
#         Soviet Union     584.0
#  silver United States   1195.0
#         Soviet Union     627.0
#  gold   United States   2088.0
#         Soviet Union     838.0

# when appending disjoint dataframes, missing entries filled with NaN

# joining data

# => bronze
#                   Total
#  Country
#  United States   1052.0
#  Soviet Union     584.0
#  United Kingdom   505.0
#  France           475.0
#  Germany          454.0

# => gold
#                   Total
#  Country
#  United States   2088.0
#  Soviet Union     838.0
#  United Kingdom   498.0
#  Italy            460.0
#  Germany          407.0

# do a SQL-like inner join i.e. set intersection
# axis=1 for horizontal concatenation
pd.concat([bronze, gold], keys=['bronze', 'gold'], axis=1, join='inner')

#                  bronze    gold
#                   Total   Total
#  Country
#  United States   1052.0  2088.0
#  Soviet Union     584.0   838.0
#  United Kingdom   505.0   498.0
#  Germany          454.0   407.0

# merge does an inner join by default using the index
pd.merge(bronze, gold)
# all columns with shared names are used to join by default
# can override by specifying on
# this will by default create Total_x and Total_y columns
pd.merge(bronze, gold, on=['NOC', 'Country'])
# to override suffixes
pd.merge(bronze, gold, on=['NOC', 'Country'], suffixes=['_bronze', '_gold'])
# if names don't match across dfs, use left_on, right_on

# merges and orders by index, then by column value left to right
pd.merge_ordered()
