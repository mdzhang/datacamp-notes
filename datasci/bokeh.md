Use [Bokeh][bokeh] to generate interactive figures


#### Terms

- **glyph**: Bokeh's terms for visual properties of shapes
    - **patch**: a type of glyph used to represent polygons, passed in as a list of lists, each sublist representing a coordinate
- **`ColumnDataSource`**: a table-like data object that maps string column names to sequences (columns) of data

#### Basic usage

Given a data source, bokeh generates static HTML, CSS, and JS

```python
from bokeh.plotting import figure, ColumnDataSource
from bokeh.io import output_file, show
from bokeh.layouts import row, column, gridplot
from bokeh.models import HoverTool

# create figure
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')
# draw circles on the figure at coordinates with x, y values in the given lists
# alternatively can pass pandas series or numpy arrays
# internally, Bokeh maps these data types to ColumnDataSource objercts
p.circle(female_literacy, fertility)
# output to filej
output_file('fert_lit.html')
# display
show(p)



p2 = figure(x_axis_type='datetime', x_axis_label='Date', y_axis_label='US Dollars')
# draw a line between data points, and white circles on the data points
p2.line(date, price)
p2.circle(date, price, fill_color='white', size=4)



p3 = figure(x_axis_label='longitude', y_axis_label='latitude')

x = [az_lons, co_lons, nm_lons, ut_lons]
y = [az_lats, co_lats, nm_lats, ut_lats]
p3.patches(x, y, line_color='white')


# create a bokeh ColumnDataSource from a pandas Dataframe
source = ColumnDataSource(df)
# specify that ColumnDataSource object should be used and its
# column 'color' used for the colors, 'Year' used for the x axis values
# and 'Time' column for the y axis values
p.circle(x='Year', y='Time', source=source, color='color', size=8)

# when used with a box selection tool, highlight selected circles with selection_color
# and have everything else reduced to 0.1 alpha i.e. very high transparency
p.circle(x='Year', y='Time', source=source, selection_color='red', nonselection_alpha=0.1)

# add a hover tool object to the figure so that
# when a datapoint is hovered over it shows a tooltip with 'Year' the string
# as a label and the value of 'Year' in source as a value
hover = HoverTool(tooltips=[('Year','@Year')])
p.add_tools(hover)


# arrange figures in a layout
layout = row(column(p1, p2), p3)

#equivalent to
layout = gridplot([p1, p3], [p2, None])

show(layout)
```

#### Using in an app

You can also use a bokeh server to dynamically update bokeh plots e.g. to keep in sync w/ data on a backend, to react to UI interactions

```python
from bokeh.io import curdoc
from bokeh.plotting import figure

plot = figure()

source = ColumnDataSource(data={'x': x, 'y': y})

plot.line(x='x', y='y', source=source)
slider = Slider()
layout = column(widgetbox(slider), plot)

def callback(attr, old, new):
  n = slider.value
  source.data = {'x': x, 'y': n * x}

slider.on_change('value', callback)

curdoc().add_root(plot)
```


[dc-bokeh-introcourse]: https://campus.datacamp.com/courses/interactive-data-visualization-with-bokeh
