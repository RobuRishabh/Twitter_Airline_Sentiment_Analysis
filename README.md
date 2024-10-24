# Twitter US-Airline Sentiment Analysis Project

---

## Project Overview

This project involves analyzing sentiment from tweets related to U.S. airlines using the **Twitter US Airline Sentiment Dataset**. The goal is to predict the sentiment (positive, negative, or neutral) of a tweet based on its content. The dataset contains over 14,000 tweets, and the challenge is to predict the sentiment for a test set of tweets.

---

## Dataset Description

1. **Training Dataset** (`training_twitter_x_y_train.csv`):
   - This dataset contains both the features (tweets) and the labels (sentiments).
   - Important columns:
     - `text`: The tweet text that we use as input for sentiment prediction.
     - `airline_sentiment`: The target variable, which can be "positive", "negative", or "neutral".

2. **Test Dataset** (`test_twitter_x_test.csv`):
   - This dataset contains only the features (tweets) for which we need to predict the sentiment.
   - Important columns:
     - `text`: The tweet text that needs a sentiment prediction.

---

## Project Steps

### 1. Data Preprocessing

- **Text Cleaning**: The tweet texts are cleaned by converting to lowercase and removing unnecessary characters (like punctuation, stop words).
  
### 2. Feature Extraction

- **TF-IDF Vectorization**: The text data is transformed into numerical features using the **TF-IDF (Term Frequency-Inverse Document Frequency)** method. This helps convert text into a format suitable for machine learning algorithms.

### 3. Model Selection

- A **Logistic Regression** model is used to classify the tweets into positive, negative, or neutral sentiment. This model is well-suited for text classification tasks and gives good performance with TF-IDF features.

### 4. Model Training and Validation

- The training dataset is split into training and validation sets to evaluate the model's performance.
- The model achieved an accuracy of **79.5%** on the validation data.

### 5. Prediction on Test Data

- The trained model is then used to predict the sentiment of the tweets in the test dataset.
- The predictions are saved in the required format: a CSV file with one column of predictions and no headers.

---

## File Descriptions

- **twitter_train.csv**: Contains the training data with tweet text and sentiment labels.
- **twitter_test.csv**: Contains the test data with tweet text for which sentiment needs to be predicted.
- **twitter_sentiment_predictions.csv**: The output file containing sentiment predictions for the test dataset, formatted as required (no headers, one column).
- **Twitter_Airline_Sentiment_Analysis.ipynb**: The Jupyter notebook file containing the code for loading data, preprocessing, training the model, and generating predictions.

---

## Requirements

To run the project, the following Python libraries are required:
- `pandas`
- `scikit-learn`

You can install the necessary dependencies using the following command:

```bash
pip install pandas scikit-learn
```

---

## How to Run

1. Clone the repository or download the files.
2. Place the datasets in the same directory.
3. Run the Jupyter notebook to train the model and generate predictions.
4. The predictions will be saved as `twitter_sentiment_predictions.csv`.

---

## Results

The predictions are saved in a file called `twitter_sentiment_predictions.csv`, which contains a single column with the predicted sentiment for each tweet from the test dataset.

---

## Conclusion

This project successfully builds a machine learning pipeline to predict the sentiment of tweets from airline passengers. By leveraging TF-IDF vectorization and Logistic Regression, we achieved a decent accuracy and were able to make predictions on unseen test data.

---