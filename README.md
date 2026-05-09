
<!-- README.md is generated from README.Rmd. Please edit that file -->

# TheseusPlot: Visualizing Decomposition of Differences in Rate Metrics

<!-- badges: start -->

<!-- badges: end -->

## 1. Overview

In data analysis, when a metric differs between two groups, we sometimes
want to investigate whether a particular subgroup is driving that
difference. For example, when a key metric decline is detected compared
to the previous year, you may want to conduct a more detailed analysis.
In this analysis, you may focus on gender among the attributes and
examine whether the decline occurred among male, female, or both.
However, this type of analysis is challenging when the metric is a rate,
because the magnitude of each subgroup’s contribution to the rate cannot
be simply calculated, unlike in the case of volume metrics.

To address this issue, we propose an approach inspired by the story of
the *[Ship of Theseus](https://en.wikipedia.org/wiki/Ship_of_Theseus)*.
This approach involves gradually replacing the components of one group
with those of another, recalculating the metric at each step. The change
in the metric at each step can then be interpreted as the contribution
of each subgroup to the overall difference.

For instance, suppose the metric was 6.2% in 2024 and decreased to 5.2%
in 2025. Again, we focus on gender. We replace the male data within the
2024 dataset with the male data from 2025 and recalculate the metric. As
a result, the metric would drop by 0.8 percentage points, reaching 5.4%.
In this case, the contribution of the male group to the change in the
metric is -0.8 percentage points. Next, we replace the female data from
2024 with that from 2025. The dataset then consists entirely of 2025
data, and the metric drops by 0.2 percentage points, reaching 5.2%.
Thus, the contribution of the female group is -0.2 percentage points.

When visualized, the results appear as follows:

<img src="README-figures/overview-1.png" alt="" width="500" />

From this plot, we can see that the decline in the metric is primarily
driven by the male group. We call this visualization the “Theseus Plot.”

The **TheseusPlot** package is designed to make it easy to generate
Theseus Plots for various attributes.

## 2. Installation

You can install the **theseusplot** package from this repository in
editable mode.

``` bash
python -m pip install -e .
```

You can install the optional dependencies for examples and documentation
data with:

``` bash
python -m pip install -e ".[examples]"
```

## 3. Details

### 3.1 Prepare Data

To create Theseus plots, you need two data frames that share common
columns.

We use the 2013 New York City flight data from
[nycflights13](https://cran.r-project.org/package=nycflights13) as a
demo dataset. Here, we will define the rate metric as the proportion of
flights that arrived on time. In December 2013, the on-time arrival rate
dropped substantially compared to November. We investigate the cause
using a Theseus plot.

First, we create an `on_time` column in the data frame to indicate
whether each flight arrived on time. Next, we extract the flights for
November and December into separate data frames to form two comparison
groups. The on-time arrival rate was 64% in November and dropped to 47%
in December.

``` python
from nycflights13 import airlines, flights

data = (
    flights.dropna(subset=["arr_delay"])
    .assign(on_time=lambda df: df["arr_delay"] <= 15)
    .merge(airlines, on="carrier")
    .assign(carrier=lambda df: df["name"])
    .loc[
        :,
        [
            "year",
            "month",
            "day",
            "origin",
            "dest",
            "carrier",
            "dep_delay",
            "on_time",
        ],
    ]
)

print(data.head())
#>    year  month  day origin dest                 carrier  dep_delay  on_time
#> 0  2013      1    1    EWR  IAH   United Air Lines Inc.        2.0     True
#> 1  2013      1    1    LGA  IAH   United Air Lines Inc.        4.0    False
#> 2  2013      1    1    JFK  MIA  American Airlines Inc.        2.0    False
#> 3  2013      1    1    JFK  BQN         JetBlue Airways       -1.0     True
#> 4  2013      1    1    LGA  ATL    Delta Air Lines Inc.       -6.0     True

data_nov = data[data["month"] == 11]
data_dec = data[data["month"] == 12]

print(data_nov["on_time"].mean())
#> 0.8264802936487339
print(data_dec["on_time"].mean())
#> 0.6738712065136936
```

### 3.2 Basics

Using the two prepared data frames, we first create a `ship` object. The
`ship` object is an instance of the Python class `ShipOfTheseus`,
designed to create Theseus plots.

``` python
from theseusplot import create_ship

ship = create_ship(
    data_nov,
    data_dec,
    y="on_time",
    labels=("November", "December"),
)
```

You can create a Theseus plot by passing column names to the `plot`
method of a `ship` object. For example, to create a Theseus plot for the
airport of origin:

``` python
fig, ax = ship.plot("origin")
fig.show()
```

<img src="README-figures/plot_origin-3.png" alt="" width="500" />

New York City has three major airports, and Newark Liberty International
Airport (EWR) accounted for the largest share of the decline in the
on-time arrival rate.

Note that the number of flights at each airport matters, as a larger
flight volume is expected to have a greater impact. To make this clear,
the Theseus plot displays the data size for each group within each
subgroup as a bar chart. From this, we see that the number of flights is
similar across airports, allowing for direct comparison of
contributions.

In summary, a Theseus plot consists of two components:

- A waterfall plot showing how much each subgroup contributed to the
  change in the metric.
- A bar chart representing the sample size for each group within each
  subgroup.

A `ship` object also provides the `table` method to inspect the exact
values used in the Theseus plot.

``` python
ship.table("origin")
#>   origin   contrib    n1    n2    x1    x2     rate1     rate2
#> 0    EWR -0.071873  9603  9410  7995  5910  0.832552  0.628055
#> 1    JFK -0.050249  8645  8923  7290  6142  0.843262  0.688334
#> 2    LGA -0.030487  8723  8687  7006  6156  0.803164  0.708645
```

### 3.3 Flipping the Plot

When there are many subgroups, a Theseus plot can become hard to read.
In such cases, you can swap the x- and y-axes for better visualization.

``` python
fig, ax = ship.plot_flip("carrier")
fig.show()
```

<img src="README-figures/plot_carrier-5.png" alt="" width="500" />

When the number of subgroups is large, those with small contributions
are automatically grouped together. By default, this happens when there
are more than 10 subgroups, but the threshold can be adjusted with the
`n` argument.

``` python
fig, ax = ship.plot_flip("carrier", n=6)
fig.show()
```

<img src="README-figures/plot_carrier_n-7.png" alt="" width="500" />

From this plot, JetBlue Airways and United Air Lines appear to have the
largest contributions to the decline in on-time arrival rate.

### 3.4 Automatic Discretization of Continuous Values

Theseus plots do not directly support continuous variables. If a
continuous column is provided, it is automatically discretized. For
example, we can create a Theseus plot for departure delays.

``` python
fig, ax = ship.plot_flip("dep_delay")
fig.show()
```

<img src="README-figures/plot_dep_delay-9.png" alt="" width="500" />

By default, continuous variables are discretized so that each subgroup
has roughly equal sample sizes, with the number of bins set to 10. You
can modify these settings by passing the return value of
`continuous_config()` to the `continuous` argument.

``` python
from theseusplot import continuous_config

fig, ax = ship.plot_flip("dep_delay", continuous=continuous_config(n=3))
fig.show()
```

<img src="README-figures/plot_dep_delay_n-11.png" alt="" width="500" />

This result shows that both a decrease in on-time departures and an
increase in delayed departures contributed to the decline in on-time
arrival rate.

### 3.5 Ordering for Factor Columns

If a subgroup column is categorical, `table()` and `plot()` respect its
category order. This is useful when you want to keep a meaningful
predefined order, such as `"Low"`, `"Medium"`, and `"High"`, instead of
ordering categories by their contributions.

``` python
import pandas as pd
from pandas.api.types import CategoricalDtype

segment_type = CategoricalDtype(
    categories=["Low", "Medium", "High"],
    ordered=True,
)

data1 = pd.DataFrame(
    {
        "segment": pd.Series(
            ["Low", "Low", "Medium", "Medium", "High", "High"],
            dtype=segment_type,
        ),
        "y": [1, 1, 1, 0, 1, 1],
    }
)

data2 = pd.DataFrame(
    {
        "segment": pd.Series(
            ["Low", "Low", "Medium", "Medium", "High", "High"],
            dtype=segment_type,
        ),
        "y": [1, 0, 1, 1, 0, 0],
    }
)

ship = create_ship(data1, data2, y="y", labels=("Group 1", "Group 2"))

print(ship.table("segment"))
#>   segment   contrib  n1  n2  x1  x2  rate1  rate2
#> 0     Low -0.166667   2   2   2   1    1.0    0.5
#> 1  Medium  0.166667   2   2   1   2    0.5    1.0
#> 2    High -0.333333   2   2   2   0    1.0    0.0

fig, ax = ship.plot("segment")
fig.show()
```

<img src="README-figures/factor_column-13.png" alt="" width="500" />

Even if the contribution of `"High"` is larger than that of `"Low"` or
`"Medium"`, the rows and bars are shown in the order
`"Low" -> "Medium" -> "High"` because `segment` is categorical.

By contrast, if `segment` were an object column, the output would be
ordered by contribution rather than by a predefined level order.
