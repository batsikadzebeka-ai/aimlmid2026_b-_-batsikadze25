# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Step 1: Load the dataset
# The CSV file contains email attributes and a binary label indicating spam (1) or legitimate (0)
data_file = "/Users/tchgurami/Desktop/b_batsikadze25_78912.csv"
email_data = pd.read_csv(data_file)

# Step 2: Select relevant features and target variable
# Features include: number of words, number of links, number of capitalized words, and spam-related words
features = ["words", "links", "capital_words", "spam_word_count"]
target = "is_spam"

X = email_data[features]
y = email_data[target]

# Step 3: Split data into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Step 4: Train a Logistic Regression model
# Logistic Regression is suitable for binary classification tasks
spam_model = LogisticRegression(max_iter=1200)
spam_model.fit(X_train, y_train)

# Step 5: Display model coefficients to understand feature impact
print("\nLogistic Regression Coefficients:")
for feat, coef in zip(features, spam_model.coef_[0]):
    print(f"{feat}: {coef:.4f}")

# Step 6: Evaluate model on the test set
y_pred_test = spam_model.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred_test)
accuracy = accuracy_score(y_test, y_pred_test)

print("\nConfusion Matrix:")
print(conf_mat)
print(f"\nModel Accuracy: {accuracy:.2%}")

# Step 7: Function to extract features from raw email text
def extract_email_features(email_text):
    """Extracts numeric features from an email to feed into the model."""
    word_count = len(email_text.split())
    link_count = email_text.lower().count("http") + email_text.lower().count("www")
    capital_count = sum(1 for w in email_text.split() if w.isupper())
    spam_words = ["free", "win", "money", "offer", "urgent", "click", "prize"]
    spam_word_count = sum(email_text.lower().count(word) for word in spam_words)
    return pd.DataFrame([[word_count, link_count, capital_count, spam_word_count]], columns=features)

# Step 8: Function to classify a new email
def classify_email_text(email_text):
    """Classifies a raw email text as Spam or Legitimate using the trained model."""
    feature_df = extract_email_features(email_text)
    prediction = spam_model.predict(feature_df)[0]
    label = "Spam" if prediction == 1 else "Legitimate"
    print(f"\nEmail Classification Result: {label}")
    print("Extracted Features:")
    print(feature_df.to_string(index=False))

# Test with a manually composed spam email
spam_email_text = "WIN BIG MONEY NOW! Click this urgent link http://spammyoffer.com to claim your prize!"
classify_email_text(spam_email_text)

# Test with a manually composed legitimate email
legit_email_text = "Hello team, please review the attached project proposal and provide your feedback by Friday."
classify_email_text(legit_email_text)

# Step 9: Visualization 1 - Spam vs Legitimate Email Count
plt.figure(figsize=(6, 4))
sns.countplot(x=target, data=email_data, palette="pastel")
plt.title("Distribution of Spam and Legitimate Emails")
plt.xlabel("Email Type (0 = Legitimate, 1 = Spam)")
plt.ylabel("Number of Emails")
plt.tight_layout()
plt.savefig("spam_vs_legit_count.png")
plt.show()

# Step 10: Visualization 2 - Feature Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(email_data.corr(), annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Correlation Between Email Features and Target Label")
plt.tight_layout()
plt.savefig("feature_corr_heatmap.png")
plt.show()
