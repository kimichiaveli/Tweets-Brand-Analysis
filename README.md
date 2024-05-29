# Twitter Sentiment Analysis with Dash

This Python script utilizes Dash to create an interactive web application for performing sentiment analysis on Twitter data.

## Prerequisites

Before running the script, make sure you have the following installed:

- Python (version >= 3.6)
- Pip or Conda to install Python Libraries
- Dash (Python web framework)
- NLTK (Python library for processing textual data)
- emoji (Not the actual emojis :sweat_smile:, but the Python library to detect and translate emoji into text for sentiment analysis)
- Plotly (Python library for data visualization, i use plotly.express to build interactive dashboards for the dashboard)
  
You can install the required packages using pip or conda:

```bash
pip install dash nltk plotly emoji
```

## Features

- Upload Data: Upload the data that will be used for the analysis.
- Preprocessing: Text preprocessing technique that utilizes urls and stopwords remover to clean the text before doing sentiment analysis.
- Sentiment Distribution and Mapping Analysis: Enter a keyword or hashtag to fetch recent tweets containing that keyword from the sample data, and visualize the sentiment analysis based on sentiment polarity distribution and mapping by Country.
- Interactive Visualization: The Dash application provides interactive graphs to display sentiment polarity and subjectivity.
- User-friendly Interface: Simple and intuitive interface designed using Dash, allowing users to easily analyze Twitter sentiment without writing any code.

## Gallery

<div align="center">
  <img src="https://raw.githubusercontent.com/kimichiaveli/Tweets-Brand-Analysis/master/screenshots/home_dashboard.png" alt="Dashboard Home" width="600">
  <p><span style="font-size: 12px;"><i>This is the visuals of the home dashboard.</i></span></p>
</div>

<br>
<br>

<div align="center">
  <img src="https://raw.githubusercontent.com/kimichiaveli/Tweets-Brand-Analysis/master/screenshots/preprocess_result.png" alt="Text Preprocessing" width="600">
  <p><span style="font-size: 12px;"><i>Example of text preprocessing result</i></span></p>
</div>

<br>
<br>

<div align="center">
  <img src="https://raw.githubusercontent.com/kimichiaveli/Tweets-Brand-Analysis/master/screenshots/data_visualization.png" alt="Visualization" width="600">
  <p><span style="font-size: 12px;"><i>This is the visualization generated, Sentiment Distribution and Choropleth Map</i></span></p>
</div>

## Future Development

- Write a Python script to scrap tweets using Twitter API with language detection and tweets geo tracker
