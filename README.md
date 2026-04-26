# Task 2 – Support Ticket Classification

## Overview
An automated system that classifies customer support tickets into categories and assigns a priority level.  
Helps support teams triage issues faster.

## Approach
- **Text Preprocessing**: Lowercasing, punctuation removal.
- **Feature Extraction**: TF‑IDF vectorization.
- **Model**: Logistic Regression (multi‑class).
- **Priority Logic**: Rule‑based keyword mapping for high/medium/low tagging.

## Results
- Classification report (printed in console) shows precision/recall per category.
- Confusion matrix saved as `confusion_matrix.png`.

## Sample Output
| Ticket | Category | Priority |
|--------|----------|----------|
| Cannot login... | Account | high |
| Payment failed... | Payment | high |
| How do I update... | Billing | low |

## Visual
![Confusion Matrix](confusion_matrix.png)

## Tools
Python, Pandas, Scikit‑learn, Matplotlib