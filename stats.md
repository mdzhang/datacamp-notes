# Statistics

Notes from Probability & Statistics courses on Datacamp

- **exploratory data analysis (EDA)**: process of organizing, processing, and summarizing a data set
    - see John Tukey
- graphical EDA: visualizing data
- quantitative EDA

## Graphical EDA

- graphical EDA makes it easier to see outliers and trends that can/should influence later analysis and prevent misinterpretation of data

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# use seaborn styling by setting this before plotting
sns.set()

arr = np.array([4.7, 4.5, 4.9])
plt.hist(arr, bins=np.sqrt(len(arr)

# label axes
plt.xlabel('petal length (cm)')
plt.ylabel('count')

# add padding so no data points overlap the edge of the graph i.e. edges are farther out
plt.margins(0.02)

plt.show()
```

- **square root rule**: choose # bins for a histogram that is ~= sqrt(# observations)
    - help avoid **binning bias**: when same data interpreted differently when # bins is different
- **bee swarm plot**
    - unaffected by **binning bias**

    ```python
    sns.swarmplot(x='state', y='dem_share', data=df)
    ```

    - not ideal when you have many observations that start to overlap each other in the plot
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
- **boxplot**: shows min and max (at whiskers, or 1.5 IQR, whichever smaller), median at middle box line, 25th and 75th percentiles at box bottom and top, and outliers

    ```python
    sns.boxplot(x='state', y='dem_share', data=df)
    ```

- **scatterplot**: plot point for each observation

    ```python
    # x, y
    plt.plot(total_votes/100, dem_share, marker='.', linestyle='none')
    ```


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
- **covariance**: measure of how two quantities vary together
    - variability due to codepedence
    - e.g. Obama vote share and total vote count per county
    - look at given point p = (x1, y1) distance from x mean and y mean
        - if distances both positive, positively related (positive correlation)
        - if one distance is above mean, the other below => negatively related (anticorrelation)
    - `np.cov`
        - gives a covariance matrix where `[0, 0]` is variance of data for `x`, `[1,1]` for `y`, and `[1,0] == [0,1]`
- **pearson correlation coefficient**: covariance / std(x) * std(y)
    - aka Pearson r
    - variability due to codepedence / independent variability
    - range [-1, 1] where 1 indicates perfect correlation
    - `np.corrcoef`

    ```python
    def pearson_r(x, y):
      """Compute Pearson correlation coefficient between two arrays."""
      # Compute correlation matrix: corr_mat
      corr_mat = np.corrcoef(x, y)

      # Return entry [0,1]
      return corr_mat[0,1]
    ```

## Probability

- probabilistic reasoning allows us to describe uncertainty
- given a set of data, describe what you would expect if that data were to be acquired again
- **statistical inference**: process of going from statistical data to probabilistic conclusions

- **discrete variable**: has limited number of finitely many possible discrete values
    - aka **categorical variable** or **factor**
- **continuous variable**: can take any value

##### Discrete variable distributions

- **Bernoulli trial**: experiment that has two options success (true) and failure (false)
    - `p` probability of success
    - `1 - p` probability of failure
- `np.random.random(size=n)` for pseudo random number generation
    - takes integer seed and feeds to random number generating algorithm
    - can reproduce random number generation with `np.random.seed(42)`

- **hacker statistics**: use programatically generated data that mirrors real-world outcomes

- **probability mass function (PMF)**: set of probabilities of discrete outcomes
    - **discrete uniform PMF**: discrete set of possible outcomes, each with the same probability
- **probability distribution**: mathematical description of outcomes
    - Bernoulli trial's have a **binomial distribution**
      - r successes in n bernoulli trials with probability p of success is binomially distributed

      ```python
      # run bernoulli trial with e.g. n coin flips where p is probability of heads 10000 times
      np.random.binomial(n, p, size=10000)
      ```

- **poisson process**: timing of next event completely independent of when the previous event happened
    - e.g. # natural births in a hospital
    - `r` arrivals over some period w/ rate of arrivals `lambda`
    - is a limitation of a bernoulli trial where rate of success is low i.e. rare events and the number of Bernoulli trials is large.
    - `np.random.poisson(r)`
    - takes fewer args; easier to compute than binomial distribution
    - binomial distribution gets closer to poisson distribution as `p` decreases

      ```python
      # Draw 10,000 samples out of Poisson distribution: n_nohitters
      n_nohitters = np.random.poisson(251/115, size=10000)
      # Compute number of samples that are seven or greater: n_large
      n_large = np.sum(n_nohitters >= 7)
      # Compute probability of getting seven or more: p_large
      p_large = n_large / len(n_nohitters)
      ```

##### Continuous variable distributions

- **probability density function (PDF)**: continuous analog to the PMF
    - describes likelihood of observing value of a continuous variable
    - because there are an infinity of values between e.g. 1 and 2, instead consider area under the PDF between those values e.g. k%

- **normal distribution**: describes continuous variable whose PDF has single, symmetric peak
    - aka **gaussian distribution**
    - center of peak at the mean
    - st dev indicates width of peak
    - `np.random.normal(mean, std, size=n)`
    - has "light", flat tails i.e. probability of being > 4 stdevs away from the mean is _very_ small

- **exponential distribution**: waiting time between arrivals of a poisson process have this distribution
    - single parameter `tau`, the typical interval time between poisson process arrivals aka mean of the data
    - `np.random.exponential(tau, size=n)`
    - typical shape has maximum at 0 and decays to the right

      ```python
      def successive_poisson(tau1, tau2, size=1):
        # Draw samples out of first exponential distribution: t1
        t1 = np.random.exponential(tau1, size=size)

        # Draw samples out of second exponential distribution: t2
        t2 = np.random.exponential(tau2, size=size)

        return t1 + t2
      ```

      ```python
      # ecdf of actual data
      x, y = ecdf(nohitter_times)

      # theoretical ecdf
      tau = np.mean(nohitter_times)
      # value of tau is the optimal parameter if actual data plotted closely aligned with plot of inter_nohitter_time cdf
      inter_nohitter_time = np.random.exponential(tau, 100000)
      x_theor, y_theor = ecdf(inter_nohitter_time)

      # overlay the plots to confirm distribution is exponential
      plt.plot(x, y, marker='.', linestyle='none')
      plt.plot(x_theor, y_theor)
      ```


- data often has an underlying function giving it its shape
  - **linear function**
    - has **slope** and **intercept** (where it crosses y-axis)
    - line best fits data if collectively data points as close to line as possible
    - vertical distance between point and plotted line is the **residual** (negative if below the line, positive if above)
    - can see what line fits best by using **least squares** i.e. line for which the sum of the squares of the residual is minimal
        - i.e. optimize **residual sum of squares (RSS)**
    - `slope, intercept = np.polyfit(x_data, y_data, 1)`
        - last argument is degree of polynomial; linear functions have degree 1

      ```python
      # plot data as scatterplot
      _ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')
      plt.margins(0.02)
      _ = plt.xlabel('percent illiterate')
      _ = plt.ylabel('fertility')

      # perform a linear regression using np.polyfit(): a, b
      slope, intercept = np.polyfit(illiteracy, fertility, 1)

      # plot the theoretical line
      x = np.array([0, 100])
      y = slope * x + intercept

      # add regression line to your plot
      _ = plt.plot(x, y)

      # Draw the plot
      plt.show()
      ```

      ```python
      # demonstrate how np.polyfit finds an optimal slope and intercept
      # get slopes to consider i.e. 200 values betwee 0 and 0.2
      a_vals = np.linspace(0, 0.1, 200)

      # to store RSS values per slope
      rss = np.empty_like(a_vals)

      # compute sum of squares
      for i, a in enumerate(a_vals):
          # i.e. residual ^ 2
          # where residual = y - slope * x - b
          rss[i] = np.sum((fertility - a*illiteracy - b)**2)

      # plot it; should show a parabola whose minimum should be the slope
      # ??: how intercept is calculated?
      plt.plot(a_vals, rss, '-')
      plt.xlabel('slope (children per woman / percent illiterate)')
      plt.ylabel('sum of square of residuals')

      plt.show()
      ```

- **bootstrapping**: resampling of data to perform statistical inference
  - definition: an array of length n that was drawn from the original data with replacement
      - with replacement aka didn't delete any values from original data
  - resampling data set `o` of size n means randomly taking a sample from `o` n times - because its random each time, the data you resample is not perfectly equal to `o` (e.g. may have sampled the same value multiple times, another value not at all)
  - resampled data aka **bootstrap sample**
  - summary statistic computed from the **bootstrap sample** is a **bootstrap replicate**
  - `np.random.choice(orig_data, size=len(orig_data))`

  ```python
  def bootstrap_replicate_1d(data, func):
      return func(np.random.choice(data, size=len(data)))

  def draw_bs_reps(data, func, size=1):
      return np.array([bootstrap_replicate_1d(data, func) for _ in range(size)])


  def draw_bs_pairs_linreg(x, y, size=1):
      """Perform pairs bootstrap for linear regression."""

      # Set up array of indices to sample from: inds
      inds = np.arange(len(x))

      # Initialize replicates: bs_slope_reps, bs_intercept_reps
      bs_slope_reps = np.empty(size)
      bs_intercept_reps = np.empty(size)

      # Generate replicates
      for i in range(size):
          bs_inds = np.random.choice(inds, size=len(inds))
          bs_x, bs_y = x[bs_inds], y[bs_inds]
          bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, 1)

      return bs_slope_reps, bs_intercept_reps
  ```

- **confidence interval**: if we repeated measurements over an over again, `p%` of the observed values would lie within the `p%` confidence interval
  - when using the bootstrap method, can get a confidence interval with `np.percentile(bs_replicates, [2.5, 97.5]) # 95% confidence interval`
- **standard error of the mean (SEM)**: the standard deviation of a probabilistic distribution of the mean
  - `sem = np.std(data) / np.sqrt(len(data))`

- **nonparametric inference**: make no assumption about model/probability distribution underlying data, calculate parameter values using data alone
- **parametric inference**: does make assumptions
  - e.g. linear regression needed slope/intercept parameters we calculated directly from data sample
  - if we took another data sample, slope/intercept could change


##### Hypothesis Testing

- **hypothesis testing**: assessment of how reasonable the observed data are assuming hypothesis is true
    - steps:
      - state clear null hypothesis
      - define test statistic
      - generate many sets of simulated data assuming the null hypothesis is true
      - compute test statistic for each simulated dataset
      - p-value is fraction of simulated data sets for which the test statistic is at least as extreme as the real data
    - use permutation sampling to simulate the hypothesis that two variables have identical probability distributions
        - e.g. hypothesis that Ohio/Pennsylvania counties democratic vote % identically distributed
        - join 2 sets of data (democratic vote % for OH and PA), scramble them (permute them), reallocate to PA and OH
        - **permutation replicate** is a single value of a statistic computed from a **permutation sample**

      ```python
      def permutation_sample(data1, data2):
          """Generate a permutation sample from two data sets."""
          data = np.random.permutation(np.concatenate((data1, data2)))
          return data[:len(data1)], data[len(data1):]

      def draw_perm_reps(data_1, data_2, func, size=1):
          """Generate multiple permutation replicates."""

          # Initialize array of replicates: perm_replicates
          perm_replicates = np.empty(size)

          for i in range(size):
              # Generate permutation sample
              perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)

              # Compute the test statistic
              perm_replicates[i] = func(perm_sample_1, perm_sample_2)

          return perm_replicates
      ```
- **null hypothesis**: hypothesis we are testing
- **test statistic**: computed from observed data (actual) and data simulated under null hypothesis (predicted)
    - e.g. `mean` as test statistic for OH/PA hypothesis above
    - **true mean**: mean we would have achieved had we continued collecting more and more samples
- **p-value**: probability of obtaining value of test statistic at least as extreme as what was observed, under the assumption the null hypothesis is true
    - e.g. take a distribution of means taken from permutation samples, with original data with actual mean `mu`
    - calculate area under histogram to the right of `mu` and e.g. get `.23` or 23%
    - means that at least 23% of simulated elections had a mean _at least as extreme as_ `mu`
    - p-value in this case is `.23`
    - if p-value is small, data are considered statistically significantly different
    - **not** probability null hypothesis is true
    - low p-value suggests null hypothesis is false

    ```python
    def diff_of_means(data_1, data_2):
        """Difference in means of two arrays."""

        # The difference of means of data_1, data_2: diff
        diff = np.mean(data_1) - np.mean(data_2)

        return diff

    # compute real diff of means from observed data
    empirical_diff_means = diff_of_means(force_a, force_b)
    # get 10k replicates for the diff of means
    perm_replicates = draw_perm_reps(force_a, force_b,
                                     diff_of_means, size=10000)
    p = np.sum(perm_replicates >= empirical_diff_means) / len(perm_replicates)

    #
    # hypothesis: The mean strike force of Frog B is equal to that of Frog C
    # one sample test b/c only have sample data for frog b but have mean for frog c (0.55)
    #

    # shift force to have same mean as frog C; preserve variance, distribution, etc.
    translated_force_b = force_b - np.mean(force_b) + 0.55
    bs_replicates = draw_bs_reps(translated_force_b, np.mean, 10000)
    p = np.sum(bs_replicates <= np.mean(force_b)) / 10000
    ```

- **A/B test**: see whether a change in strategy improved an outcome

    ```python
    # hypothesis: fewer no-hitters (where batters couldn't hit a ball thrown by a pitcher)
    #             in the live ball era than the dead ball era
    #             (dead ball era when pitchers allowed to scuff/spit on ball, favoring pitchers, easier to obtain no-hitters)
    #             i.e. mean(nht_dead) < mean(nht_live) (easier to throw no-hitters => fewer games between no-hitters)
    # null hypothesis: no difference in mean # games between no-hitters in the live and dead ball era
    # test statistic: difference of means
    # data set: # games between no-hitters

    # compute observed difference
    nht_diff_obs = diff_of_means(nht_dead, nht_live)

    # acquire 10,000 permutation replicates of difference in mean no-hitter time
    perm_replicates = draw_perm_reps(nht_dead, nht_live, diff_of_means, 10000)

    p = np.sum(perm_replicates <= nht_diff_obs)/len(perm_replicates)
    print('p-val =',p)

    # small p-value indicates low probability that null hypothesis holds
    # i.e. that there is a significant difference in mean # games between no hitters in the live and dead ball era
    ```
