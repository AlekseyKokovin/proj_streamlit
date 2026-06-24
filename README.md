# Crypto News Sentiment and Market Reaction Analysis

An interactive Streamlit web application designed to analyze the relationship between cryptocurrency news sentiment and market returns, specifically focusing on the performance of **altcoins** versus **blue-chip** coins. 

## Project Overview

This dashboard aggregates daily news sentiment (polarity and subjectivity) and maps it against daily market returns for top cryptocurrencies. By visualizing these relationships and running statistical hypothesis tests, the app provides insights into how different types of news (Objective, Subjective, or Mixed) impact market volatility and index performance.

## Features

The application is structured into a sidebar for dynamic data filtering and five main tabs for deep-dive analysis:

* **Interactive Sidebar Filters:** Filter the dataset in real-time by News Type (Objective, Subjective, Mixed) and mean sentiment polarity ranges.
* **Tab 1: Data Overview and Quality:** Displays dataset dimensions, key fields, data sources (Kaggle CryptoNews & Yahoo Finance), and data cleaning steps.
* **Tab 2: Descriptive Statistics:** Provides summary statistics (mean, median, standard deviation, min, max) for sentiment and market indexes, alongside news type distributions.
* **Tab 3: Visualizations and Overview:** interactive Plotly charts, including:
    * BTC returns grouped by sentiment and subject.
    * Scatter plots comparing Blue-chip vs. Altcoin market reactions.
    * Violin distributions and Correlation heatmaps.
    * Timeline trends and subjectivity comparisons.
* **Tab 4: Hypothesis Testing:** * **Altcoin Reactions:** Conducts pairwise T-tests with a Bonferroni correction to see if altcoin returns differ significantly by news type.
    * **Subjectivity vs. Polarity:** Calculates the Pearson correlation to test if higher subjectivity leads to more extreme sentiment polarity.
* **Tab 5: Discussion and Insights:** Summarizes the methodology, limitations, and offers a direct download link for the filtered dataset (`full_df.csv`).

## Prerequisites and Installation

```bash
pip install streamlit pandas scipy matplotlib seaborn plotly
streamlit run app.py
```
