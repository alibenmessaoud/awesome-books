## Data Science from Scratch

![](./data-science-from-scratch-cover.png)

by Joel Grus
Copyright © 2015 O’Reilly Media. All rights reserved.

------

Data Science

Data Scientist has been called The Sexiest Job of the 21st Century. According to a Venn diagram that is somewhat famous in the industry, data science lies at the intersection of:
• Hacking skills
• Math and statistics knowledge
• Substantive expertise

If you become a data scientist, you will become intimately familiar with NumPy, with scikit-learn, with pandas, and with a panoply of other libraries without actually understanding data science.

Joel, in this book, approached data science from scratch and tried to implement algorithms by hand in order to better understand them.

### Chapter 1

data scientists are statisticians, mathematicians, PhDs, engineers, etc ... Well, let's say that a data scientist is
someone who extracts insights from messy data. Today’s world is full of people trying to turn data into insight.

> OkCupid asks its members to answer thousands of questions in order to find the most appropriate matches for them but it also analyzes to find out how likely someone is to sleep with you on the first date.
>
> Facebook asks its mambers to list their locations, ostensibly to make it easier to find and connect them. But it also analyzes to identify global migration patterns to target ads.
>
> Target tracks purchases and interactions, both online and in-store. And it uses the data to predictively model which of its customers are pregnant, to better market baby-related purchases to them.
>
> Obama 2012 re-election used data scientists to identifying voters who needed extra attention, which means that political campaigns of the future will become more and more data-driven, resulting in a never-ending arms race of data science and data collection.

Some warm up example, Finding Key Connectors in a list of user with friendships data, FOAF, Salaries and experience, and topics of interest.

### Chapter 2

Python introduction

![](./data-science-from-scratch-by-Joel-grus.ipynb)  

### Chapter 3: Visualizing Data

```python
from matplotlib import pyplot as plt
years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]
# create a line chart, years on x-axis, gdp on y-axis
plt.plot(years, gdp, color='green', marker='o', linestyle='solid')
# add a title
plt.title("Nominal GDP")
# add a label to the y-axis
plt.ylabel("Billions of $")
plt.show()

# _/
```

- Bar Charts: A bar chart is a good choice when you want to show how some quantity varies among some discrete set of items.
- Line Charts: As we saw already, we can make line charts using plt.plot() . These are a good choice for showing trends
- Scatterplots: A scatterplot is the right choice for visualizing the relationship between two paired sets of data.

For Further Exploration

- seaborn is built on top of matplotlib.
- D3.js is a JavaScript library.
- Bokeh is a newer library that brings D3-style visualizations into Python.
- ggplot is a Python port of the popular R library ggplot2 for “publication quality” charts and graphics.

### Chapter 4: Linear Algebra

