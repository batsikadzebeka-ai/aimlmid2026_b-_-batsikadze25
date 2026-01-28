# Spam Email Detection Using Logistic Regression

## 1. Dataset Upload

The dataset used in this project is uploaded to the repository.

File name: b_batsikadze25_78912.csv

Repository link (replace with your actual GitHub URL):
https://github.com/your-username/spam-email-classifier/blob/main/b_batsikadze25_78912.csv

------------------------------------------------------------------------

## 2. Model Training and Source Code

### Data Loading and Processing

The dataset is loaded using pandas. The following features are
selected: - words - links - capital_words - spam_word_count

Target label: - is_spam (1 = Spam, 0 = Legitimate)

The dataset is split into 70 percent training data and 30 percent
testing data using train_test_split.

### Logistic Regression Model

A Logistic Regression model is used for binary classification. It is
trained on the training portion of the dataset.

### Model Coefficients

The trained model produced the following coefficients:

words: 0.0065\
links: 0.8939\
capital_words: 0.4511\
spam_word_count: 0.7658

Links and spam-related keywords have the strongest influence on spam
detection.

------------------------------------------------------------------------

## 3. Model Validation

The model is evaluated using the test dataset.

Confusion Matrix: \[\[347, 7\], \[12, 384\]\]

Accuracy: 97.47 percent

The confusion matrix and accuracy are calculated using sklearn.metrics.

------------------------------------------------------------------------

## 4. Email Text Classification

The application can classify raw email text by extracting the same
features as in the dataset. These include word count, link count,
capitalized words, and spam-related keywords. The extracted features are
passed to the trained model to determine whether an email is spam or
legitimate.

------------------------------------------------------------------------

## 5. Manually Composed Spam Email

Spam email example: WIN BIG MONEY NOW! Click this urgent link
http://spammyoffer.com to claim your FREE prize!

Explanation: This email contains spam keywords, a URL, and excessive
capitalization, all of which increase the likelihood of spam
classification.

def extract_email_features(email_text):
    word_count = len(email_text.split())
    link_count = email_text.lower().count("http") + email_text.lower().count("www")
    capital_count = sum(1 for w in email_text.split() if w.isupper())
    spam_words = ["free", "win", "money", "offer", "urgent", "click", "prize"]
    spam_word_count = sum(email_text.lower().count(word) for word in spam_words)


------------------------------------------------------------------------

## 6. Manually Composed Legitimate Email

Legitimate email example: Hello team, please review the attached project
proposal and provide your feedback by Friday.

Explanation: This email contains no spam keywords, no links, and no
excessive capitalization, resulting in a legitimate classification.

------------------------------------------------------------------------

## 7. Visualizations

Visualization 1: Spam vs Legitimate Email Distribution A bar chart
showing the number of spam and legitimate emails. This visualization
demonstrates that the dataset is relatively balanced.

Visualization 2: Feature Correlation Heatmap A heatmap showing
correlations between features and the spam label. It highlights strong
correlations between spam-related features and the target variable.

------------------------------------------------------------------------

## Conclusion

This project demonstrates a complete spam email detection system using
Logistic Regression. The model achieves high accuracy and successfully
classifies both dataset samples and manually created email text.
