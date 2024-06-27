
# YouTube Sentiment Analysis on AI

## Project Overview

This project aims to conduct sentiment analysis on YouTube videos about artificial intelligence (AI) by examining user comments. Understanding public opinion on AI is crucial, as it is a rapidly growing field expected to have a significant impact on society.

## Objectives

1. **Data Extraction**: Scrape comments from YouTube videos from some of the largest countries in Europe (France, Germany, Italy, and Spain) to create a comprehensive dataset.
2. **Sentiment Analysis**: Perform a comparative study to gain insights into public opinion towards AI in these countries.
3. **Hypothesis Testing**: Investigate whether the sentiment towards AI is more positive in northern European countries due to rapid advances in AI technology.

## Findings

Our hypothesis was that the sentiment towards AI would be more positive in northern European countries. However, our findings indicate that people in these countries have a more negative view of AI.

## Features

- Web scraping of YouTube comments using Python.
- Sentiment analysis using Natural Language Processing (NLP) techniques.
- Comparative study across different European countries.

## Technologies Used

- Python
- BeautifulSoup for web scraping
- NLTK for sentiment analysis
- Pandas for data manipulation
- Matplotlib for data visualization

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/maloooon/Youtube.git
   ```
2. Navigate to the project directory:
   ```sh
   cd Youtube
   ```
3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. Run the web scraping script to collect YouTube comments:
   ```sh
   python scrape_comments.py
   ```
2. Perform sentiment analysis on the collected comments:
   ```sh
   python sentiment_analysis.py
   ```
3. Generate visualizations and comparative study results:
   ```sh
   python visualize_results.py
   ```

## Dataset

The scraped dataset is available for the community and can be accessed in the `data` directory of this repository.
