# Regression

Data often has an underlying function giving it its shape.

- **linear regression**: the process of finding the line/linear function that best fits a data set
    - _assumes_ data has a linear shape
    - line is defined by its **slope** and **intercept** (where it crosses y-axis)
      - numpy: `slope, intercept = np.polyfit(x, y, deg=1)`
          - last argument is degree of polynomial; linear functions have degree 1
      - pandas: `pd.ols(y, x)`
      - scipy: `scipy.stats.linregress(x, y)`
      - statsmodels: `statsmodels.api.OLS(y, x).fit()`
    - line best fits data if collectively data points as close to line as possible
      - vertical distance between point and plotted line is the **residual** (negative if below the line, positive if above)
      - can see what line fits best by using **least squares** i.e. line for which the sum of the squares of the residual is minimal
      - i.e. optimize **residual sum of squares (RSS)**





