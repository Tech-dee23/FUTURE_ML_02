import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Synthetic ticket data (8 samples – expandable)
data = {
    'ticket': [
        "Cannot login to my account, password reset not working",
        "Payment failed but card is valid, need refund",
        "How do I update my billing address?",
        "App crashes when I open settings on Android",
        "Request for account deletion and data export",
        "Where can I find my invoice for last month?",
        "Server down in US East region, urgent!",
        "Login page takes too long to load, 504 error"
    ],
    'category': ['Account', 'Payment', 'Billing', 'Bug', 'Account', 'Billing', 'Outage', 'Bug'],
    'priority': ['high', 'high', 'low', 'medium', 'low', 'low', 'high', 'medium']
}
df = pd.DataFrame(data)

# Text cleaning
def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

df['clean_ticket'] = df['ticket'].apply(clean)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_ticket'], df['category'], test_size=0.25, random_state=42, stratify=df['category']
)

# TF‑IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Logistic Regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test)

# Classification report
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion matrix plot
fig, ax = plt.subplots(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
plt.title('Support Ticket Classification – Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# Priority tagging logic (rule‑based)
priority_keywords = {
    'outage': 'high', 'urgent': 'high', 'error': 'medium',
    'bug': 'medium', 'crash': 'medium', 'how': 'low'
}
def assign_priority(text):
    for word, prio in priority_keywords.items():
        if word in text:
            return prio
    return 'medium'

df['predicted_priority'] = df['clean_ticket'].apply(assign_priority)
print("\nSample output with priority:")
print(df[['ticket', 'category', 'predicted_priority']].head(8).to_string(index=False))