# Twitter Sentiment Analysis with Dash

This Python script utilizes Dash to create an interactive web application for performing sentiment analysis on Twitter data.

## Prerequisites

Before running the script, make sure you have the following installed:

- Python (version >= 3.6)
- Pip or Conda to install Python Libraries
- Dash (Python web framework)
- NLTK (Python library for processing textual data)
- Plotly (Python library for data visualization, i use plotly.express to build interactive dashboards for the dashboard)
  
You can install the required packages using pip or conda:

```bash
pip install dash nltk plotly
```

## Features

- Upload Data: Upload the data that will be used for the analysis.
- Preprocessing: Text preprocessing technique that utilizes urls and stopwords remover to clean the text before doing sentiment analysis.
- Sentiment Distribution and Mapping Analysis: Enter a keyword or hashtag to fetch recent tweets containing that keyword from the sample data, and visualize the sentiment analysis based on sentiment polarity distribution and mapping by Country.
- Interactive Visualization: The Dash application provides interactive graphs to display sentiment polarity and subjectivity.
- User-friendly Interface: Simple and intuitive interface designed using Dash, allowing users to easily analyze Twitter sentiment without writing any code.

## Future Development

- Write a Python script to scrap tweets using Twitter API
