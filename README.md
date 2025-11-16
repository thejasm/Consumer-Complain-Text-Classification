# Readme

## Screenshot

![](https://github.com/thejasm/Consumer-Complain-Text-Classification/blob/main/screenshot.png)
The screenshot contains my name as a comment in the 3rd visible line in the code.
It shows the final output of the prediction algorithm using randomized data taken from the Consumer Complaints Database.
This project is built in a jupyter notebook using Google Colab.
I've used jupyter and colab for data collection and analytics for projects in the past and is the software I was most comfortable with using for this specific task.
I also made heavy use of the gemini AI which is heavily integrated into colab.

## Final Notebook and Output

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

import requests

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

sns.set_style('darkgrid')
pd.options.display.max_colwidth = 200
```

    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package wordnet to /root/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    [nltk_data] Downloading package omw-1.4 to /root/nltk_data...
    [nltk_data]   Package omw-1.4 is already up-to-date!



```python
API_BASE_URL = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
all_complaints = []
num_iterations = 5
complaints_per_request = 1000

# Define headers to mimic a web browser request
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'accept': 'application/json'
}

print(f"Attempting to fetch {num_iterations * complaints_per_request} complaints...")

for i in range(num_iterations):
    current_offset = i * complaints_per_request
    # Construct the API URL for the complaints endpoint
    # Explicitly add format=json and include 'size' and 'from' parameters
    endpoint = f"?size={complaints_per_request}&from={current_offset}"
    full_url = API_BASE_URL + endpoint

    print(f"Fetching data from: {full_url}")

    try:
        response = requests.get(full_url, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()

        if 'hits' in data and data['hits']['hits']:
            # The actual complaint records are nested under 'hits' -> 'hits'
            all_complaints.extend(data['hits']['hits'])
            print(f"Successfully fetched {len(data['hits']['hits'])} complaints (total fetched: {len(all_complaints)}).")
        else:
            print(f"No more complaints found or unexpected response for offset {current_offset}.")
            break # Stop if no more data is returned
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for offset {current_offset}: {e}")
        break # Stop on error

print(f"Finished fetching. Total complaints collected: {len(all_complaints)}")
```

    Attempting to fetch 5000 complaints...
    Fetching data from: https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/?size=1000&from=0
    Successfully fetched 1000 complaints (total fetched: 1000).
    Fetching data from: https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/?size=1000&from=1000
    Successfully fetched 1000 complaints (total fetched: 2000).
    Fetching data from: https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/?size=1000&from=2000
    Successfully fetched 1000 complaints (total fetched: 3000).
    Fetching data from: https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/?size=1000&from=3000
    Successfully fetched 1000 complaints (total fetched: 4000).
    Fetching data from: https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/?size=1000&from=4000
    Successfully fetched 1000 complaints (total fetched: 5000).
    Finished fetching. Total complaints collected: 5000


## Prepare Data from API Response

### Subtask:
Convert the `all_complaints` list, obtained from the API, into a pandas DataFrame. Extract the 'consumer_complaint_narrative' and 'product' fields for classification. Handle potential missing values in the narrative.



```python
processed_complaints = []

for complaint in all_complaints:
    source_data = complaint.get('_source', {})
    # Corrected key from 'consumer_complaint_narrative' to 'complaint_what_happened'
    narrative = source_data.get('complaint_what_happened')
    product = source_data.get('product')

    # Only add if both narrative and product are present and not empty strings
    if narrative and product:
        processed_complaints.append({
            'narrative': narrative,
            'product': product
        })

df = pd.DataFrame(processed_complaints)

# Remove rows where 'narrative' is missing
df.dropna(subset=['narrative'], inplace=True)

# Reset the index after dropping rows
df.reset_index(drop=True, inplace=True)

print(f"Original number of complaints fetched: {len(all_complaints)}")
print(f"Number of complaints after processing and dropping NaNs: {len(df)}")
print("First 5 rows of the processed DataFrame:")
print(df.head())
```

    Original number of complaints fetched: 5000
    Number of complaints after processing and dropping NaNs: 1370
    First 5 rows of the processed DataFrame:
                                                                                                                                                                                                     narrative  \
    0  Chae bank called me about a fraudulent case application. They gave me a case number XXXX, tranferred me CFPB. During the call, long silent when agent said is looking at the report, then cut off. N...   
    1  The inclusion of these unauthorized accounts and inquiries has caused considerable stress and concern regarding my financial well-being. It is imperative that my credit report accurately reflects ...   
    2  I respectfully seek your comprehension regarding the late remarks on my report, and I am fully dedicated to resolving them as per our agreement. My goal is to eliminate all the late remarks from m...   
    3                                                                          Upon reviewing my credit report, I identified multiple accounts that appear to be fraudulent and were added without my consent.   
    4  sent letters to aargon agency asking for validation of debt after the 30 day mark I didn't know the account had been sent to them the original account was with XXXXXXXX XXXX neither company never ...   
    
                                                   product  
    0                                          Credit card  
    1  Credit reporting or other personal consumer reports  
    2  Credit reporting or other personal consumer reports  
    3                                      Debt collection  
    4                                      Debt collection  


## Map Target Categories

### Subtask:
Map the 'product' column from the DataFrame to the specified target categories (0: Credit reporting, repair, or other, 1: Debt collection, 2: Consumer Loan, 3: Mortgage). Filter out categories not relevant to the specified classification task.



```python
category_mapping = {
    'Credit reporting, credit repair services, or other': 0,
    'Debt collection': 1,
    'Consumer Loan': 2,
    'Mortgage': 3
}

# Create a new column 'category_id' based on the mapping
df['category_id'] = df['product'].map(category_mapping)

# Filter out rows where 'category_id' is NaN (i.e., product not in our target categories)
df.dropna(subset=['category_id'], inplace=True)

# Convert 'category_id' to integer type
df['category_id'] = df['category_id'].astype(int)

# Display unique product categories and their counts
print("Product categories and their counts after mapping and filtering:")
print(df['product'].value_counts())
print("\nCategory IDs and their counts:")
print(df['category_id'].value_counts())

print("\nFirst 5 rows of the DataFrame after mapping and filtering:")
print(df.head())
```

    Product categories and their counts after mapping and filtering:
    product
    Debt collection    100
    Mortgage            10
    Name: count, dtype: int64
    
    Category IDs and their counts:
    category_id
    1    100
    3     10
    Name: count, dtype: int64
    
    First 5 rows of the DataFrame after mapping and filtering:
                                                                                                                                                                                                      narrative  \
    3                                                                           Upon reviewing my credit report, I identified multiple accounts that appear to be fraudulent and were added without my consent.   
    4   sent letters to aargon agency asking for validation of debt after the 30 day mark I didn't know the account had been sent to them the original account was with XXXXXXXX XXXX neither company never ...   
    47                                                      Upon carefully reviewing my recent credit report, I identified several fraudulent accounts that appear to have been added without my authorization.   
    70                                                                  Upon closely reviewing my latest credit report, I found multiple accounts that I believe to be fraudulent and added without my consent.   
    93  XX/XX/XXXX XXXX has misapplied my mortgage payments again for the 7th time in 14 months. \nXXXX ) XXXX XXXX, Assistant Vice President of Roundpoint Mortgage Servicing, has knowingly made false sta...   
    
                product  category_id  
    3   Debt collection            1  
    4   Debt collection            1  
    47  Debt collection            1  
    70  Debt collection            1  
    93         Mortgage            3  


## Exploratory Data Analysis (EDA) and Feature Engineering

### Subtask:
Perform initial EDA on the DataFrame, including checking the distribution of the target categories and the length of complaint narratives. Create a new feature for text length.



```python
print("Distribution of target categories:")
print(df['category_id'].value_counts())

# Create a new feature for text length
df['text_length'] = df['narrative'].apply(len)

print("\nDescriptive statistics for text_length:")
print(df['text_length'].describe())

# Visualize the distribution of text_length
plt.figure(figsize=(10, 6))
sns.histplot(df['text_length'], bins=50, kde=True)
plt.title('Distribution of Complaint Narrative Length')
plt.xlabel('Text Length (Characters)')
plt.ylabel('Frequency')
plt.show()

```

    Distribution of target categories:
    category_id
    1    100
    3     10
    Name: count, dtype: int64
    
    Descriptive statistics for text_length:
    count     110.000000
    mean      936.227273
    std      1182.379615
    min       127.000000
    25%       196.000000
    50%       316.500000
    75%      1347.000000
    max      4900.000000
    Name: text_length, dtype: float64



    
![png](main_files/main_10_1.png)
    


## Text Pre-Processing

### Subtask:
Clean the 'consumer_complaint_narrative' text by removing special characters, numbers, and converting to lowercase. Tokenize the text, remove stopwords, and apply lemmatization. Display a sample of the processed text.



```python
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords and lemmatize
    cleaned_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
    ]
    # Join back to string
    return ' '.join(cleaned_tokens)

# Apply preprocessing to the 'narrative' column
df['cleaned_narrative'] = df['narrative'].apply(preprocess_text)

print("First 5 entries of original and cleaned narratives:")
print(df[['narrative', 'cleaned_narrative']].head())
```

    First 5 entries of original and cleaned narratives:
                                                                                                                                                                                                      narrative  \
    3                                                                           Upon reviewing my credit report, I identified multiple accounts that appear to be fraudulent and were added without my consent.   
    4   sent letters to aargon agency asking for validation of debt after the 30 day mark I didn't know the account had been sent to them the original account was with XXXXXXXX XXXX neither company never ...   
    47                                                      Upon carefully reviewing my recent credit report, I identified several fraudulent accounts that appear to have been added without my authorization.   
    70                                                                  Upon closely reviewing my latest credit report, I found multiple accounts that I believe to be fraudulent and added without my consent.   
    93  XX/XX/XXXX XXXX has misapplied my mortgage payments again for the 7th time in 14 months. \nXXXX ) XXXX XXXX, Assistant Vice President of Roundpoint Mortgage Servicing, has knowingly made false sta...   
    
                                                                                                                                                                                              cleaned_narrative  
    3                                                                                                          upon reviewing credit report identified multiple account appear fraudulent added without consent  
    4   sent letter aargon agency asking validation debt day mark didnt know account sent original account xxxxxxxx xxxx neither company never notified see previous complaint cfpb xxxx aargon replied debt...  
    47                                                                                   upon carefully reviewing recent credit report identified several fraudulent account appear added without authorization  
    70                                                                                              upon closely reviewing latest credit report found multiple account believe fraudulent added without consent  
    93  xxxxxxxx xxxx misapplied mortgage payment th time month xxxx xxxx xxxx assistant vice president roundpoint mortgage servicing knowingly made false statement federal regulator xxxx xxxx made statem...  


## Feature Extraction using TF-IDF

### Subtask:
Convert the pre-processed text data into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.



```python
tfidf_vectorizer = TfidfVectorizer(max_features=5000) # Limiting to 5000 features for manageable size
X_tfidf = tfidf_vectorizer.fit_transform(df['cleaned_narrative'])

print("Shape of TF-IDF features (X_tfidf):")
print(X_tfidf.shape)
print("First 5 feature names:")
print(tfidf_vectorizer.get_feature_names_out()[:5])
```

    Shape of TF-IDF features (X_tfidf):
    (110, 617)
    First 5 feature names:
    ['aargon' 'aargons' 'ability' 'accept' 'according']


## Split Data into Training and Testing Sets

### Subtask:
Split the pre-processed and vectorized dataset into training and testing sets to prepare for model training and evaluation.



```python
X = X_tfidf
y = df['category_id']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")
```

    Shape of X_train: (88, 617)
    Shape of X_test: (22, 617)
    Shape of y_train: (88,)
    Shape of y_test: (22,)


## Train and Evaluate Multiple Classification Models

### Subtask:
Train and evaluate multiple multi-classification models including Logistic Regression, Multinomial Naive Bayes, Linear SVC, and RandomForest Classifier using the training data. For each model, report accuracy, precision, recall, and F1-score on the test set.



```python
model_results = {}

models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
    ('Multinomial Naive Bayes', MultinomialNB()),
    ('Linear SVC', LinearSVC(dual=False, random_state=42)), # Set dual=False when n_samples > n_features
    ('RandomForest Classifier', RandomForestClassifier(random_state=42))
]

for name, model in models:
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    model_results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }

    print(f"{name} Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

print("\nModel training and evaluation complete.")
```

    
    Training Logistic Regression...
    Logistic Regression Performance:
      Accuracy: 0.9091
      Precision: 0.8264
      Recall: 0.9091
      F1-Score: 0.8658
    
    Training Multinomial Naive Bayes...
    Multinomial Naive Bayes Performance:
      Accuracy: 1.0000
      Precision: 1.0000
      Recall: 1.0000
      F1-Score: 1.0000
    
    Training Linear SVC...
    Linear SVC Performance:
      Accuracy: 1.0000
      Precision: 1.0000
      Recall: 1.0000
      F1-Score: 1.0000
    
    Training RandomForest Classifier...
    RandomForest Classifier Performance:
      Accuracy: 1.0000
      Precision: 1.0000
      Recall: 1.0000
      F1-Score: 1.0000
    
    Model training and evaluation complete.


## Compare Model Performance

### Subtask:
Compare the performance of the trained models using metrics such as accuracy, and visualize their performance using a bar chart. Provide a clear legend for the visualization.



```python
performance_df = pd.DataFrame(model_results).T
print(performance_df)
```

                             Accuracy  Precision    Recall  F1-Score
    Logistic Regression      0.909091   0.826446  0.909091  0.865801
    Multinomial Naive Bayes  1.000000   1.000000  1.000000  1.000000
    Linear SVC               1.000000   1.000000  1.000000  1.000000
    RandomForest Classifier  1.000000   1.000000  1.000000  1.000000


```python
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
performance_df[metrics].plot(kind='bar', figsize=(12, 7), width=0.8)
plt.title('Model Performance Comparison Across Metrics')
plt.xlabel('Model')
plt.ylabel('Score')
plt.ylim(0, 1.05) # Set y-axis limit from 0 to 1.05 for scores
plt.xticks(rotation=45, ha='right')
plt.legend(title='Metrics')
plt.tight_layout()
plt.show()
```


    
![png](main_files/main_23_0.png)
    


## Detailed Model Evaluation

### Subtask:
Select the best-performing model and perform a detailed evaluation, including generating a confusion matrix and a classification report for a deeper understanding of its performance across different categories.



```python
best_model_name = 'Multinomial Naive Bayes'
best_model = MultinomialNB()
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)

print(f"Best model chosen: {best_model_name}")

print("\nClassification Report for Best Model (Multinomial Naive Bayes):")
print(classification_report(y_test, y_pred_best, target_names=[str(name) for name in sorted(y.unique())], zero_division=0))

cm = confusion_matrix(y_test, y_pred_best)
print("Confusion Matrix for Best Model (Multinomial Naive Bayes):")
print(cm)
```

    Best model chosen: Multinomial Naive Bayes
    
    Classification Report for Best Model (Multinomial Naive Bayes):
                  precision    recall  f1-score   support
    
               1       1.00      1.00      1.00        20
               3       1.00      1.00      1.00         2
    
        accuracy                           1.00        22
       macro avg       1.00      1.00      1.00        22
    weighted avg       1.00      1.00      1.00        22
    
    Confusion Matrix for Best Model (Multinomial Naive Bayes):
    [[20  0]
     [ 0  2]]



```python
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[product_names[i] for i in sorted(y.unique())],
            yticklabels=[product_names[i] for i in sorted(y.unique())])
plt.title('Confusion Matrix for Multinomial Naive Bayes')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
```


    
![png](main_files/main_26_0.png)
    


## Prediction on Sample Data

### Subtask:
Use the best-performing model to make predictions on a few sample consumer complaints from the test set or new unseen data to demonstrate its functionality.



```python
sample_complaints = [
    "I am constantly getting calls about a debt I do not owe. This is harassment.",
    "My mortgage payment increased unexpectedly without any prior notification or clear explanation from the bank.",
    "I need to dispute an inaccurate item on my credit report that is affecting my score.",
    "I took out a personal loan and now the interest rates are much higher than agreed upon.",
    "I received a letter demanding payment for a medical bill that I already paid last year."
]

# Create a reverse mapping for category_id to product name
id_to_category = {
    0: 'Credit reporting, credit repair services, or other',
    1: 'Debt collection',
    2: 'Consumer Loan',
    3: 'Mortgage'
}

# Preprocess the sample complaints
processed_sample_complaints = [preprocess_text(complaint) for complaint in sample_complaints]

print("Original sample complaints:")
for i, complaint in enumerate(sample_complaints):
    print(f"{i+1}. {complaint}")

print("\nProcessed sample complaints:")
for i, p_complaint in enumerate(processed_sample_complaints):
    print(f"{i+1}. {p_complaint}")
```

    Original sample complaints:
    1. I am constantly getting calls about a debt I do not owe. This is harassment.
    2. My mortgage payment increased unexpectedly without any prior notification or clear explanation from the bank.
    3. I need to dispute an inaccurate item on my credit report that is affecting my score.
    4. I took out a personal loan and now the interest rates are much higher than agreed upon.
    5. I received a letter demanding payment for a medical bill that I already paid last year.
    
    Processed sample complaints:
    1. constantly getting call debt owe harassment
    2. mortgage payment increased unexpectedly without prior notification clear explanation bank
    3. need dispute inaccurate item credit report affecting score
    4. took personal loan interest rate much higher agreed upon
    5. received letter demanding payment medical bill already paid last year



```python
X_sample = tfidf_vectorizer.transform(processed_sample_complaints)

# Make predictions
predicted_category_ids = best_model.predict(X_sample)

print("\nPredictions for sample complaints:")
for i, complaint in enumerate(sample_complaints):
    predicted_product_name = id_to_category[predicted_category_ids[i]]
    print(f"Original Complaint {i+1}: {complaint}")
    print(f"Predicted Category: {predicted_product_name}\n")
```

    
    Predictions for sample complaints:
    Original Complaint 1: I am constantly getting calls about a debt I do not owe. This is harassment.
    Predicted Category: Debt collection
    
    Original Complaint 2: My mortgage payment increased unexpectedly without any prior notification or clear explanation from the bank.
    Predicted Category: Debt collection
    
    Original Complaint 3: I need to dispute an inaccurate item on my credit report that is affecting my score.
    Predicted Category: Debt collection
    
    Original Complaint 4: I took out a personal loan and now the interest rates are much higher than agreed upon.
    Predicted Category: Debt collection
    
    Original Complaint 5: I received a letter demanding payment for a medical bill that I already paid last year.
    Predicted Category: Debt collection
    



```python
print("\nFetching a new random set of complaints from the API...")

new_complaints_count = 1000 # Number of new complaints to fetch
new_offset = np.random.randint(0, 10000) # Random offset to get different complaints

new_complaints = []
endpoint_new = f"?size={new_complaints_count}&from={new_offset}"
full_url_new = API_BASE_URL + endpoint_new

try:
    response_new = requests.get(full_url_new, headers=headers)
    response_new.raise_for_status()
    data_new = response_new.json()

    if 'hits' in data_new and data_new['hits']['hits']:
        new_complaints.extend(data_new['hits']['hits'])
        print(f"Successfully fetched {len(data_new['hits']['hits'])} new complaints.")
    else:
        print("No new complaints found or unexpected response.")
except requests.exceptions.RequestException as e:
    print(f"Error fetching new data: {e}")

processed_new_complaints = []
original_new_complaints_text = []

for complaint in new_complaints:
    source_data = complaint.get('_source', {})
    narrative = source_data.get('complaint_what_happened')
    product = source_data.get('product') # Keep original product for comparison if needed, but not used for prediction input

    if narrative:
        original_new_complaints_text.append(narrative)
        processed_new_complaints.append(preprocess_text(narrative))

if not processed_new_complaints:
    print("No valid new narratives found to predict.")
else:
    # Transform the processed new complaints using the fitted TF-IDF vectorizer
    X_new_sample = tfidf_vectorizer.transform(processed_new_complaints)

    # Make predictions
    predicted_new_category_ids = best_model.predict(X_new_sample)

    print("\nPredictions for new sample complaints from API:")
    for i, original_text in enumerate(original_new_complaints_text):
        predicted_product_name = id_to_category.get(predicted_new_category_ids[i], "Unknown Category")
        print(f"New Complaint {i+1}:")
        print(f"  Original Text: {original_text[:150]}...") # Print a snippet
        print(f"  Predicted Category: {predicted_product_name}\n")
```

    
    Fetching a new random set of complaints from the API...
    Successfully fetched 1000 new complaints.
    
    Predictions for new sample complaints from API:
    New Complaint 1:
      Original Text: Chae bank called me about a fraudulent case application. They gave me a case number XXXX, tranferred me CFPB. During the call, long silent when agent ...
      Predicted Category: Debt collection
    
    New Complaint 2:
      Original Text: The inclusion of these unauthorized accounts and inquiries has caused considerable stress and concern regarding my financial well-being. It is imperat...
      Predicted Category: Debt collection
    
    New Complaint 3:
      Original Text: I respectfully seek your comprehension regarding the late remarks on my report, and I am fully dedicated to resolving them as per our agreement. My go...
      Predicted Category: Debt collection
    
    New Complaint 4:
      Original Text: Upon reviewing my credit report, I identified multiple accounts that appear to be fraudulent and were added without my consent....
      Predicted Category: Debt collection
    
    New Complaint 5:
      Original Text: sent letters to aargon agency asking for validation of debt after the 30 day mark I didn't know the account had been sent to them the original account...
      Predicted Category: Debt collection
    
    New Complaint 6:
      Original Text: I am writing to you as a concerned consumer regarding the accuracy and fairness of the information contained in my credit report. As you are aware, th...
      Predicted Category: Debt collection
    
    New Complaint 7:
      Original Text: The late payments showing don't match how I regularly make my payments, so Im asking you to fix these issues and adjust my accounts as soon as possibl...
      Predicted Category: Debt collection
    
    New Complaint 8:
      Original Text: The reporting of such inaccurate information has caused severe damage to my character, my reputation, my general mode of living and my ability to obta...
      Predicted Category: Debt collection
    
    New Complaint 9:
      Original Text: Recently I looked at a copy of my credit report and noticed several inaccuracies on my account. This account is hurting my ability to obtain credit. I...
      Predicted Category: Debt collection
    
    New Complaint 10:
      Original Text: I recently received a copy of my credit report, and I noticed some accounts on my consumer report that should not be on there. Please help me to valid...
      Predicted Category: Debt collection
    
    New Complaint 11:
      Original Text: I am XXXX XXXX, and Im submitting this complaint myself and there is no third party involved. I attached letters to let you know more in detail about ...
      Predicted Category: Debt collection
    
    New Complaint 12:
      Original Text: In accordance with the Fair Credit Reporting act. The List of accounts below has violated my federally protected consumer rights to privacy and confid...
      Predicted Category: Debt collection
    
    New Complaint 13:
      Original Text: Recently i did an investigation on my credit report which caused severe XXXX upon me and found unverifiable, invalidated, inaccurate, and questionable...
      Predicted Category: Debt collection
    
    New Complaint 14:
      Original Text: These inquiries are a result of identity theft I have filed police report and ftc report when I noticed.i provided documentation to credit bureau and ...
      Predicted Category: Debt collection
    
    New Complaint 15:
      Original Text: Ive call Equifax serval times about my credit report and score when I login to Equifax it tells me to call and give the Agent the code blnk number hav...
      Predicted Category: Debt collection
    
    New Complaint 16:
      Original Text: My credit reports are inaccurate. These inaccuracies are causing creditors to deny me credit. You have the duty to report accurate information about c...
      Predicted Category: Debt collection
    
    New Complaint 17:
      Original Text: Im reaching out for your help regarding the incorrect account listed on my credit report. This situation has damaged my reputation, and I urgently nee...
      Predicted Category: Debt collection
    
    New Complaint 18:
      Original Text: In the quiet humdrum of everyday life, a sudden jolt shattered the tranquility : I had become a victim of identity theft. Like a thief in the night, u...
      Predicted Category: Debt collection
    
    New Complaint 19:
      Original Text: In accordance with the Fair Credit Reporting act. The List of accounts below has violated my federally protected consumer rights to privacy and confid...
      Predicted Category: Debt collection
    
    New Complaint 20:
      Original Text: I filed a dispute for incorrect information on my credit report and received an email from the credit bureau claiming the disputes were filed by a thi...
      Predicted Category: Debt collection
    
    New Complaint 21:
      Original Text: As per the guidance from the Consumer Financial Protection Bureau ( CFPB ) the documents needed are a picture ID, a bill, and a letter from an advocac...
      Predicted Category: Debt collection
    
    New Complaint 22:
      Original Text: I'm really not sure what happened. I have mailed off letters to the credit bureaus continuously and thus far I have not gotten a response. My name is ...
      Predicted Category: Debt collection
    
    New Complaint 23:
      Original Text: My account with Equifax was never late! I have had exceptional payment history with XXXX  XXXX and all payments were placed on XXXX. This late payment...
      Predicted Category: Debt collection
    
    New Complaint 24:
      Original Text: I'm really not sure what happened. I have mailed off letters to the credit bureaus continuously and thus far I have not gotten a response. My name is ...
      Predicted Category: Debt collection
    
    New Complaint 25:
      Original Text: Recently, I did an investigation on my credit report and found several items on there to be inaccurate. 
    
    Under 15 U.S Code 1681e ( b ) Accuracy of re...
      Predicted Category: Debt collection
    
    New Complaint 26:
      Original Text: I lodged a complaint with these Bureaus a month ago, but unfortunately, the inaccurate information on my report still remains. I haven't received any ...
      Predicted Category: Debt collection
    
    New Complaint 27:
      Original Text: In accordance with the Fair Credit Reporting act. The List of accounts below has violated my federally protected consumer rights to privacy and confid...
      Predicted Category: Debt collection
    
    New Complaint 28:
      Original Text: Recently i did an investigation on my credit report which caused XXXX XXXXXXXX upon me and found unverifiable, invalidated, inaccurate, and questionab...
      Predicted Category: Debt collection
    
    New Complaint 29:
      Original Text: I recently reviewed a copy of my credit report. I was shocked to see that I have been a victim of identity theft. I am in the process of buying a home...
      Predicted Category: Debt collection
    
    New Complaint 30:
      Original Text: When I went to check my profile I came across some inquiries that were not mine....
      Predicted Category: Debt collection
    
    New Complaint 31:
      Original Text: In accordance with the Fair Credit Reporting act. The List of accounts below has violated my federally protected consumer rights to privacy and confid...
      Predicted Category: Debt collection
    
    New Complaint 32:
      Original Text: I have already sent a letter addressing the inaccuracies and unknown items on my credit report, but unfortunately, I have not received any response ev...
      Predicted Category: Debt collection
    
    New Complaint 33:
      Original Text: I have submitted multiple disputes regarding these accounts that is reporting inaccurately on my credit report. I have also submitted multiple reports...
      Predicted Category: Debt collection
    
    New Complaint 34:
      Original Text: A month ago, I submitted a complaint to these Bureaus, but the incorrect information on my report is still there. I haven't heard back from them, asid...
      Predicted Category: Debt collection
    
    New Complaint 35:
      Original Text: I am a victim of identity theft. Please delete or remove these items on my behalf. 
    These items are not mine and this is greatly affecting me and my p...
      Predicted Category: Debt collection
    
    New Complaint 36:
      Original Text: HOW IS THE ACCOUNT OPEN CLOSED CHARGED OFF AND HAVE A PAST DUE BALANCE....
      Predicted Category: Debt collection
    
    New Complaint 37:
      Original Text: Please see unauthorized credit via identity theft
    
    XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX...
      Predicted Category: Debt collection
    
    New Complaint 38:
      Original Text: Equifax XXXX XXXX XXXX XXXX XXXX XXXX Please be advised that this is my SECOND WRITTEN REQUEST asking you to remove the unverified accounts listed bel...
      Predicted Category: Debt collection
    
    New Complaint 39:
      Original Text: I am writing to formally request the removal of several unauthorized accounts and inquiries from my credit report. These accounts and inquiries are no...
      Predicted Category: Debt collection
    
    New Complaint 40:
      Original Text: Following the Fair Credit Reporting Act, the list of accounts below has violated my federally protected consumer rights to privacy and confidentiality...
      Predicted Category: Debt collection
    
    New Complaint 41:
      Original Text: " The Fair Credit Reporting Act ( 15 U.S. Code 1681 ) says ( 1 ) The banking system is dependent upon fair and accurate credit reporting. Inaccurate c...
      Predicted Category: Debt collection
    
    New Complaint 42:
      Original Text: Dear Debt Collector, I recently received a debt collector notice from your office. I am writing to verify that you have contacted the correct individu...
      Predicted Category: Debt collection
    
    New Complaint 43:
      Original Text: I am not aware of any such inquiries, thus it looks that this one is unauthorized and false. Please act right now to resolve this problem....
      Predicted Category: Debt collection
    
    New Complaint 44:
      Original Text: The inclusion of these unauthorized accounts and inquiries has caused considerable stress and concern regarding my financial well-being. It is imperat...
      Predicted Category: Debt collection
    
    New Complaint 45:
      Original Text: I require your help to address this incorrect account on my credit report. This has negatively affected my reputation, and Im counting on your support...
      Predicted Category: Debt collection
    
    New Complaint 46:
      Original Text: I previously sent a letter to address the errors and unknown items on my credit report, but unfortunately, I have not received a response even after 3...
      Predicted Category: Debt collection
    
    New Complaint 47:
      Original Text: In accordance with the Fair Credit Reporting act. The List of accounts below has violated my federally protected consumer rights to privacy and confid...
      Predicted Category: Debt collection
    
    New Complaint 48:
      Original Text: Upon carefully reviewing my recent credit report, I identified several fraudulent accounts that appear to have been added without my authorization....
      Predicted Category: Debt collection
    
    New Complaint 49:
      Original Text: I have already sent a letter addressing the inaccuracies and unknown items on my credit report, but unfortunately, I have not received any response ev...
      Predicted Category: Debt collection
    
    New Complaint 50:
      Original Text: After discovering more fraudulent accounts on my credit report, I contacted the companies involved, and they have confirmed these accounts as fraudule...
      Predicted Category: Debt collection
    
    New Complaint 51:
      Original Text: I am bringing to your attention the presence of inaccurate information on my credit report that requires immediate correction. After thoroughly review...
      Predicted Category: Debt collection
    
    New Complaint 52:
      Original Text: I lodged a complaint with these Bureaus a month ago, but unfortunately, the inaccurate information on my report still remains. I haven't received any ...
      Predicted Category: Debt collection
    
    New Complaint 53:
      Original Text: I have previously filed a complaint regarding the negative accounts on my credit report. I have long requested for these to be removed by the three bu...
      Predicted Category: Debt collection
    
    New Complaint 54:
      Original Text: In the quiet humdrum of everyday life, a sudden jolt shattered the tranquility : I had become a victim of identity theft. Like a thief in the night, u...
      Predicted Category: Debt collection
    
    New Complaint 55:
      Original Text: A month ago, I submitted a complaint to these Bureaus, but the incorrect information on my report is still there. I haven't heard back from them, asid...
      Predicted Category: Debt collection
    
    New Complaint 56:
      Original Text: XXXX is on my credit report due to identity theft they also exposed my identity and information in a security breach...
      Predicted Category: Debt collection
    
    New Complaint 57:
      Original Text: In the quiet humdrum of everyday life, a sudden jolt shattered the tranquility : I had become a victim of identity theft. Like a thief in the night, u...
      Predicted Category: Debt collection
    
    New Complaint 58:
      Original Text: I have reviewed my credit report and found that theyre are several inaccurate data points within my personal identification that you are reporting. 
    M...
      Predicted Category: Debt collection
    
    New Complaint 59:
      Original Text: I am not associated with these inquiries, nor do I have any connection to the mentioned creditors....
      Predicted Category: Debt collection
    
    New Complaint 60:
      Original Text: My credit reports are inaccurate. These inaccuracies are causing creditors to deny me credit. You have the duty to report accurate information about c...
      Predicted Category: Debt collection
    
    New Complaint 61:
      Original Text: Subject : Dispute of Unauthorized Credit Inquiries To Whom It May Concern, I am writing to dispute several unauthorized inquiries on my credit report,...
      Predicted Category: Debt collection
    
    New Complaint 62:
      Original Text: I lodged a complaint with these Bureaus a month ago, but unfortunately, the inaccurate information on my report still remains. I haven't received any ...
      Predicted Category: Debt collection
    
    New Complaint 63:
      Original Text: My account with EQUIFAX was never late! I have had exceptional payment history with XXXX  XXXX and all payments were placed on Autopay. This late paym...
      Predicted Category: Debt collection
    
    New Complaint 64:
      Original Text: I am submitting this complaint against the credit bureaus due to their continuous failure to address my dispute letters concerning inaccurate informat...
      Predicted Category: Debt collection
    
    New Complaint 65:
      Original Text: My account with those creditors was never late! I have had exceptional payment history witH XXXX  XXXX and all payments were placed on Autopay. This l...
      Predicted Category: Debt collection
    
    New Complaint 66:
      Original Text: My personal information was used for identity theft. I am filing to request that you block the following Incorrect information from my credit report. ...
      Predicted Category: Debt collection
    
    New Complaint 67:
      Original Text: To Whom It May Concern, I am writing to express my profound disappointment with your agency 's handling of my recent dispute regarding the accuracy of...
      Predicted Category: Debt collection
    
    New Complaint 68:
      Original Text: To Whom It May Concern, As a vigilant and concerned consumer, I demand immediate and detailed information regarding the exact steps your agency has ta...
      Predicted Category: Debt collection
    
    New Complaint 69:
      Original Text: I have already sent a letter addressing the inaccuracies and unknown items on my credit report, but unfortunately, I have not received any response ev...
      Predicted Category: Debt collection
    
    New Complaint 70:
      Original Text: For the past XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX has reported Account # XXXX in error to multiple credit reporting agencies, despite my repea...
      Predicted Category: Debt collection
    
    New Complaint 71:
      Original Text: Upon closely reviewing my latest credit report, I found multiple accounts that I believe to be fraudulent and added without my consent....
      Predicted Category: Debt collection
    
    New Complaint 72:
      Original Text: My credit reports are inaccurate. These inaccuracies are causing creditors to deny me credit. You have the duty to report accurate information about c...
      Predicted Category: Debt collection
    
    New Complaint 73:
      Original Text: My credit reports are inaccurate. These inaccuracies are causing creditors to deny me credit. You have the duty to report accurate information about c...
      Predicted Category: Debt collection
    
    New Complaint 74:
      Original Text: This CFPB complaint has been filed to request pursuant of FCRA Section 605B ( 15 U.S.C. Section 1681c-2 ) that you, the XXXX credit reporting agency ,...
      Predicted Category: Debt collection
    
    New Complaint 75:
      Original Text: My credit reports are inaccurate. These inaccuracies are causing creditors to deny me credit. You have the duty to report accurate information about c...
      Predicted Category: Debt collection
    
    New Complaint 76:
      Original Text: XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX, XXXX  XXXX XXXX XXXX XX/XX/year> Consumer Financial Protection Bureau XXXX XXXX XXXX XXXX XXXX XXXX XXXX Subj...
      Predicted Category: Debt collection
    
    New Complaint 77:
      Original Text: My account with ( EQUIFAX was never late! I have had exceptional payment history with XXXX  XXXX and all payments were placed on XXXX. This late payme...
      Predicted Category: Debt collection
    
    New Complaint 78:
      Original Text: I am a victim of identity theft and some credit bureaus are refusing to removed other address related to fraud of my credit report. I am requesting to...
      Predicted Category: Debt collection
    
    New Complaint 79:
      Original Text: The Fair Credit Reporting Act ( 15 U.S. Code 1681 ) says ( 1 ) The banking system is dependent upon fair and accurate credit reporting. Inaccurate cre...
      Predicted Category: Debt collection
    
    New Complaint 80:
      Original Text: Previously filed a complaint due to in accuracy on credit report.
    
    
    Company has still failed to report accurate information regarding payment statuses...
      Predicted Category: Debt collection
    
    New Complaint 81:
      Original Text: I recently reviewed a copy of my credit report. I was shocked to see that I have been a victim of identity theft. I am in the process of buying a home...
      Predicted Category: Debt collection
    
    New Complaint 82:
      Original Text: I require your help to address this incorrect account on my credit report. This has negatively affected my reputation, and Im counting on your support...
      Predicted Category: Debt collection
    
    New Complaint 83:
      Original Text: My credit reports are inaccurate. These inaccuracies are causing creditors to deny me credit. You have the duty to report accurate information about c...
      Predicted Category: Debt collection
    
    New Complaint 84:
      Original Text: Im reaching out for your help regarding the incorrect account listed on my credit report. This situation has damaged my reputation, and I urgently nee...
      Predicted Category: Debt collection
    
    New Complaint 85:
      Original Text: My credit reports are inaccurate. These inaccuracies are causing creditors to deny me credit. You have the duty to report accurate information about c...
      Predicted Category: Debt collection
    
    New Complaint 86:
      Original Text: I recently reviewed a copy of my credit report. I was shocked to see that I have been a victim of identity theft. I am in the process of buying a home...
      Predicted Category: Debt collection
    
    New Complaint 87:
      Original Text: I am requesting the information regarding the method of verification employed by your bureau to confirm the accuracy of disputed items on my credit re...
      Predicted Category: Debt collection
    
    New Complaint 88:
      Original Text: Basis for dispute : **Inaccuracy of information. The charge off information reported is incorrect. 
    Account Name : XXXX, Account Number : XXXX Hight B...
      Predicted Category: Debt collection
    
    New Complaint 89:
      Original Text: Recently I've received a copy of my credit report in which I noticed that there is information and accounts that are unlawfully and inaccurately repor...
      Predicted Category: Debt collection
    
    New Complaint 90:
      Original Text: I was checking my report and I saw that several inquiries were made under my name but were not mine on Equifax, XXXX and XXXX. I was able to talk to e...
      Predicted Category: Debt collection
    
    New Complaint 91:
      Original Text: They pulled my credit without my consent. I went to the dealership for a purchase of a car and they pulled my credit with numerous institutions that I...
      Predicted Category: Debt collection
    
    New Complaint 92:
      Original Text: To do what is fair to me as a consumer, delete and update my credit report as they are reporting inaccurate information that I have made them aware of...
      Predicted Category: Debt collection
    
    New Complaint 93:
      Original Text: My credit reports are inaccurate. These inaccuracies are causing creditors to deny me credit. You have the duty to report accurate information about c...
      Predicted Category: Debt collection
    
    New Complaint 94:
      Original Text: XX/XX/XXXX XXXX has misapplied my mortgage payments again for the 7th time in 14 months. 
    XXXX ) XXXX XXXX, Assistant Vice President of Roundpoint Mor...
      Predicted Category: Mortgage
    
    New Complaint 95:
      Original Text: In accordance with the Fair Credit Reporting act. The List of accounts below has violated my federally protected consumer rights to privacy and confid...
      Predicted Category: Debt collection
    
    New Complaint 96:
      Original Text: In accordance with the Fair Credit Reporting act. The List of accounts below has violated my federally protected consumer rights to privacy and confid...
      Predicted Category: Debt collection
    
    New Complaint 97:
      Original Text: My credit reports are inaccurate. These inaccuracies are causing creditors to deny me credit. You have the duty to report accurate information about c...
      Predicted Category: Debt collection
    
    New Complaint 98:
      Original Text: My credit reports are inaccurate. These inaccuracies are causing creditors to deny me credit. You have the duty to report accurate information about c...
      Predicted Category: Debt collection
    
    New Complaint 99:
      Original Text: These inquiries are not related to me or any accounts I'm associated with....
      Predicted Category: Debt collection
    
    New Complaint 100:
      Original Text: I am writing to express my deep concern regarding the unavailability of my credit data from your bureau. As a diligent consumer who consistently monit...
      Predicted Category: Debt collection
    
    New Complaint 101:
      Original Text: In accordance with the Fair Credit Reporting act. The List of accounts below has violated my federally protected consumer rights to privacy and confid...
      Predicted Category: Debt collection
    
    New Complaint 102:
      Original Text: I have requested proof that I were late on the accounts reported....
      Predicted Category: Debt collection
    
    New Complaint 103:
      Original Text: I respectfully request the elimination of these incorrect accounts from my credit report, as I never authorized their inclusion. It is crucial that yo...
      Predicted Category: Debt collection
    
    New Complaint 104:
      Original Text: For the past two months, I have tried to reach out to your company regarding these items that I want removed/deleted off my report and so far, nothing...
      Predicted Category: Debt collection
    
    New Complaint 105:
      Original Text: 1. XXXX is the only credit bureau I authorized to keep my credit report, to this effect I had a personal statemen, which the company removed without c...
      Predicted Category: Debt collection
    
    New Complaint 106:
      Original Text: These late payments do not reflect my usual on-time payment habits, so I need you to correct the information and adjust my accounts promptly....
      Predicted Category: Debt collection
    
    New Complaint 107:
      Original Text: The following accounts do not belong to me, they are the product of identity theft. My name and last name are very common and this can lend itself to ...
      Predicted Category: Debt collection
    
    New Complaint 108:
      Original Text: I first contacted SANTANDER CONSUMER USA on XX/XX/XXXX regarding the insurance check I received from my insurance company, XXXX XXXX. The check is int...
      Predicted Category: Debt collection
    
    New Complaint 109:
      Original Text: XX/XX/XXXX by me paid into my escrow account XXXX to fill escrow shortage and stop house payment from increasing. Paid all house payments of XXXX regu...
      Predicted Category: Mortgage
    
    New Complaint 110:
      Original Text: There were several inaccurate personal information in my credit report. I am having difficulties in getting them removed. It might be caused by fraud ...
      Predicted Category: Debt collection
    
    New Complaint 111:
      Original Text: NOTICE TO CEASE AND DESIST ALL COLLECTION ACTIVITIES! NOTICE OF REJECTION, REVOCATION AND TERMINATION OF ANY ASSUMPTION OF CONTRACT! NO CONTRACT AND N...
      Predicted Category: Debt collection
    
    New Complaint 112:
      Original Text: The inclusion of these unauthorized accounts and inquiries has caused considerable stress and concern regarding my financial well-being. It is imperat...
      Predicted Category: Debt collection
    
    New Complaint 113:
      Original Text: Recently, I looked at a copy of my credit report and noticed inaccurate reporting on my account. This account is hurting my ability to obtain credit. ...
      Predicted Category: Debt collection
    
    New Complaint 114:
      Original Text: In XX/XX/XXXX I was Late paying my mortgage. I applied for a loan modification in XX/XX/XXXX and was told to make 3 months of payments at the new amou...
      Predicted Category: Debt collection
    
    New Complaint 115:
      Original Text: Dear Equifax Dispute Department, I am writing to formally dispute the reporting of a collection account from XXXX that is currently listed on my Equif...
      Predicted Category: Debt collection
    
    New Complaint 116:
      Original Text: On XX/XX/year> I sent XXXX, Equifax, and XXXX a dispute about some on my account that have been defaming my character and causing me financial hardshi...
      Predicted Category: Debt collection
    
    New Complaint 117:
      Original Text: My account with Equifax was never late! I have had exceptional payment history with XXXX  XXXX and all payments were placed on Autopay. This late paym...
      Predicted Category: Debt collection
    
    New Complaint 118:
      Original Text: In accordance with the Fair Credit Reporting act. The List of accounts below has violated my federally protected consumer rights to privacy and confid...
      Predicted Category: Debt collection
    
    New Complaint 119:
      Original Text: I had a Checking account with Chase Bank- Someone hacked into my account and made about 23 false claims. Once the claims were placed Chase without que...
      Predicted Category: Debt collection
    
    New Complaint 120:
      Original Text: According to 15 U.S. Code 1681c-2, this is my statement that all of the listed accounts have been reported without my consent and therefore is a resul...
      Predicted Category: Debt collection
    
    New Complaint 121:
      Original Text: Hello, the discrepancies on my credit report are seriously holding back my financial freedom. Every time I apply for new credit, I get denied because ...
      Predicted Category: Debt collection
    
    New Complaint 122:
      Original Text: On XX/XX/XXXX, I spoke to a representative of XXXX XXXX XXXX XXXX. On this call, I requested credit validation. I also stated to the representative th...
      Predicted Category: Debt collection
    
    New Complaint 123:
      Original Text: I am not associated with these inquiries, nor do I have any connection to the mentioned creditors....
      Predicted Category: Debt collection
    
    New Complaint 124:
      Original Text: I am requesting the removal of the late payment currently reflected in my credit report. This late payment record is negatively impacting my credit st...
      Predicted Category: Debt collection
    
    New Complaint 125:
      Original Text: In accordance with the Fair Credit Reporting act. The List of accounts below has violated my federally protected consumer rights to privacy and confid...
      Predicted Category: Debt collection
    
    New Complaint 126:
      Original Text: My credit reports are inaccurate. These inaccuracies are causing creditors to deny me credit. You have the duty to report accurate information about c...
      Predicted Category: Debt collection
    
    New Complaint 127:
      Original Text: In the quiet humdrum of everyday life, a sudden jolt shattered the tranquility : I had become a victim of identity theft. Like a thief in the night, u...
      Predicted Category: Debt collection
    
    New Complaint 128:
      Original Text: As of the latest information available, this collection account with XXXX XXXX XXXX has been successfully removed by both XXXX and XXXX, and it is imp...
      Predicted Category: Debt collection
    
    New Complaint 129:
      Original Text: In accordance with the Fair Credit Reporting act. The List of accounts below has violated my federally protected consumer rights to privacy and confid...
      Predicted Category: Debt collection
    
    New Complaint 130:
      Original Text: The inclusion of these unauthorized accounts and inquiries has caused considerable stress and concern regarding my financial well-being. It is imperat...
      Predicted Category: Debt collection
    
    New Complaint 131:
      Original Text: XXXX XXXX XXXX XXXX violated my rights. This account is involved in litigation. It has lingered on my credit report for over a year. I have not suppli...
      Predicted Category: Debt collection
    
    New Complaint 132:
      Original Text: Re Application # XXXX for Loan from XXXX XXXX XXXX Once again on XX/XX/year> XXXX XXXX XXXX Has apply for a loan from US Bank XXXX XXXX XXXX XXXX XXXX...
      Predicted Category: Debt collection
    
    New Complaint 133:
      Original Text: My accounts was never late! I have had exceptional payment history with XXXX  XXXX and all payments were placed on Autopay. This late payment that is ...
      Predicted Category: Debt collection
    
    New Complaint 134:
      Original Text: They took action by making daily phone calls. Here are several dates they have called me : XX/XX/XXXXXXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX...
      Predicted Category: Debt collection
    
    New Complaint 135:
      Original Text: I am writing to express my concern regarding the inaccurate BANKRUPTCIES on my credit report. I have submitted multiple complaints to the credit burea...
      Predicted Category: Debt collection
    
    New Complaint 136:
      Original Text: ( XXXX XXXX ), ( XXXX ), ( XX/XX/XXXX ), Contents of Complaint : This CFPB complaint has been filed to request pursuant to FCRA 605B ( 15 U.S.C. 1681c...
      Predicted Category: Debt collection
    
    New Complaint 137:
      Original Text: I have found an unfamiliar account on my credit report that isn't mine. I believe this might be due to identity theft or fraud....
      Predicted Category: Debt collection
    
    New Complaint 138:
      Original Text: I sent a letter certified mail to XXXX XXXX XXXX via certified mail. I have yet to get a response letter or correction of my credit report. On Experia...
      Predicted Category: Debt collection
    
    New Complaint 139:
      Original Text: Upon cheking my condumer report I received alerts of attempt inquiries made to my name. 
    XXXX. XXXX XXXX XXXX I am not liable for this inquiry please ...
      Predicted Category: Debt collection
    
    New Complaint 140:
      Original Text: My account with was never late! I have had exceptional payment history witH XXXX XXXXXXXX and all payments were placed on Autopay. This late payment t...
      Predicted Category: Debt collection
    
    New Complaint 141:
      Original Text: Hello, I am reaching out because someone tried to open a few different credit card accounts in my name. I was alerted about the fraud by XXXX ( my act...
      Predicted Category: Debt collection
    
    New Complaint 142:
      Original Text: 1. Experian is the only credit bureau I authorized to keep my credit report, to this effect I had a personal statemen, which the company removed witho...
      Predicted Category: Debt collection
    
    New Complaint 143:
      Original Text: My account with was never late! I have had exceptional payment history witH XXXX  XXXX and all payments were placed on Autopay. This late payment that...
      Predicted Category: Debt collection
    
    New Complaint 144:
      Original Text: My account with TransUnion was never late! I have had exceptional payment history with XXXX XXXX and all payments were placed on Autopay. This late pa...
      Predicted Category: Debt collection
    
    New Complaint 145:
      Original Text: For the past two months, I have tried to reach out to your company regarding these items that I want removed/deleted off my report and so far, nothing...
      Predicted Category: Debt collection
    
    New Complaint 146:
      Original Text: Hi Equifax, I just get my credit report today and I saw 3 fraud inquiries that shall be delete from my credit report. I never authorized those inaccur...
      Predicted Category: Debt collection
    
    New Complaint 147:
      Original Text: THIS IS NOT A DUPLICATE On XXXX XXXX XXXX I submitted a complaint to the Consumer Financial Protection Bureau ( CFPB ) regarding a collection account ...
      Predicted Category: Debt collection
    
    New Complaint 148:
      Original Text: These can be combined On my credit report, you have shown incorrect accounts that should not be there at all. This is not only unjust to me, but it's ...
      Predicted Category: Debt collection
    
    New Complaint 149:
      Original Text: These can be combined On my credit report, you have shown incorrect accounts that should not be there at all. This is not only unjust to me, but it's ...
      Predicted Category: Debt collection
    
    New Complaint 150:
      Original Text: My account with XXXX XXXX AND XXXX XXXX XXXX was never late! I have had exceptional payment history witH XXXX  XXXX and all payments were placed on Au...
      Predicted Category: Debt collection
    
    New Complaint 151:
      Original Text: I recently reviewed a copy of my credit report and noticed that I have fraudulent accounts. Please remove these accounts from my report, they are hurt...
      Predicted Category: Debt collection
    
    New Complaint 152:
      Original Text: On XXXX XXXX I made a deposit of {$9900.00} with a check issued by XXXX XXXX  XXXX XXXX XXXX. The check was sent from XXXX. The bank teller stated tha...
      Predicted Category: Debt collection
    
    New Complaint 153:
      Original Text: I have requested proof that I were late on the accounts reported....
      Predicted Category: Debt collection
    
    New Complaint 154:
      Original Text: They took action by making daily phone calls. Here are several dates they have called me : XX/XX/XXXXXXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX...
      Predicted Category: Debt collection
    
    New Complaint 155:
      Original Text: They took action by making daily phone calls. Here are several dates they have called me : XX/XX/XXXXXXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX...
      Predicted Category: Debt collection
    
    New Complaint 156:
      Original Text: I had Axos Bank close out one of my XXXX XXXX  and send it to another bank for deposit. It was mailed by Axos on XX/XX/year>. It has not arrived at th...
      Predicted Category: Debt collection
    
    New Complaint 157:
      Original Text: To whom it my concern : I got money taken from my account on XX/XX/year> which was XXXX from an ATM machine in XXXX and I had my card with me in South...
      Predicted Category: Debt collection
    
    New Complaint 158:
      Original Text: Inaccurate and inconsistent reporting of an account related to XXXX XXXXXXXX Account XXXX across different credit bureaus. Specifically, under 15 U.S....
      Predicted Category: Debt collection
    
    New Complaint 159:
      Original Text: I sent a letter certified mail to XXXXXXXX XXXX XXXXXXXX via certified mail. I have yet to get a response letter or correction of my credit report. On...
      Predicted Category: Debt collection
    
    New Complaint 160:
      Original Text: These can be combined On my credit report, you have shown incorrect accounts that should not be there at all. This is not only unjust to me, but it's ...
      Predicted Category: Debt collection
    
    New Complaint 161:
      Original Text: I am reaching out to address an ongoing issue with my credit report. Despite previous attempts to resolve this matter, the inaccurate information rema...
      Predicted Category: Debt collection
    
    New Complaint 162:
      Original Text: I was stunned to see incorrect accounts on my credit report tonight, and Im completely at a loss as to how they got there. I routinely check my report...
      Predicted Category: Debt collection
    
    New Complaint 163:
      Original Text: I have already reached out to the credit bureaus regarding the inaccuracies on my credit report, but unfortunately, they have not taken the necessary ...
      Predicted Category: Debt collection
    
    New Complaint 164:
      Original Text: This is an issue that I want to be addressed and this complaint is obviously not a mistake or sent in error. Ive been trying to communicate my concern...
      Predicted Category: Debt collection
    
    New Complaint 165:
      Original Text: I asked for legitimate documents to prove the appearance of these items in the credit report....
      Predicted Category: Debt collection
    
    New Complaint 166:
      Original Text: I, XXXX XXXX, on this XX/XX/XXXX, submit this CFPB complaint against Capital One Auto Finance. I allege that on or about XX/XX/XXXX, Capital One Auto ...
      Predicted Category: Debt collection
    
    New Complaint 167:
      Original Text: I made a purchase using Afterpay with XXXX through the afterpay app. My delivery was supposed to be scheduled within XXXX hours of placing the order. ...
      Predicted Category: Debt collection
    
    New Complaint 168:
      Original Text: At the end of XXXX ( unable to provide exact date, as Navy Federal changed my account numbers as part of the Fraud Alert and Account Recovery process ...
      Predicted Category: Debt collection
    
    New Complaint 169:
      Original Text: These items are totally illegal and needs to be removed from my credit report ASAP. I never authorized any of these accounts or inquiries. Please remo...
      Predicted Category: Debt collection
    
    New Complaint 170:
      Original Text: This is so distressing and you have no idea how bad it is, Im an honest man working my way off honestly, then I look at my credit report all I see are...
      Predicted Category: Debt collection
    
    New Complaint 171:
      Original Text: I received a bank statement from Marcus by Goldman Sachs. I never opened an account with them before or any knowledge of this bank before this. There ...
      Predicted Category: Debt collection
    
    New Complaint 172:
      Original Text: I am writing to formally file a complaint against PayPal for their severe neglect of customer service standards. lack of displaying human decency and ...
      Predicted Category: Debt collection
    
    New Complaint 173:
      Original Text: I am writing to formally request the removal of several accounts and inquiries from my credit report. These accounts and inquiries are unauthorized an...
      Predicted Category: Debt collection
    
    New Complaint 174:
      Original Text: I am a victim of XXXX. The information listed below, which appears on my credit report, is the result of that XXXX. 
    
    XXXX # XXXX - BALANCE {$540.00} ...
      Predicted Category: Debt collection
    
    New Complaint 175:
      Original Text: I am a victim of XXXX. The information listed below, which appears on my credit report, is the result of that XXXX. 
    
    XXXX # XXXX XXXX XXXX # XXXX - B...
      Predicted Category: Debt collection
    
    New Complaint 176:
      Original Text: I opened this account with the promise that I would be receiving XXXX  % interest no matter what my balance would be. They lowered my interest rate to...
      Predicted Category: Debt collection
    
    New Complaint 177:
      Original Text: I bought a XXXX  from a 3rd party dealership on XX/XX/XXXX and XXXX  refuses to transfer the ownership to me. I bought the car XXXX hours from where I...
      Predicted Category: Debt collection
    
    New Complaint 178:
      Original Text: I have requested proof that I were late on the accounts reported....
      Predicted Category: Debt collection
    
    New Complaint 179:
      Original Text: In accordance with the Fair Credit Reporting act. The List of accounts below has violated my federally protected consumer rights to privacy and confid...
      Predicted Category: Debt collection
    
    New Complaint 180:
      Original Text: I asked for legitimate documents to prove the appearance of these items in the credit report. 
    
    XXXX XXXX XXXX XXXXXXXX...
      Predicted Category: Debt collection
    
    New Complaint 181:
      Original Text: Good day, please take this complaint seriously as my life and my future is at stake in here, this has been stressing me out for months because of your...
      Predicted Category: Debt collection
    
    New Complaint 182:
      Original Text: I have already reached out to the credit bureaus regarding the inaccuracies on my credit report, but unfortunately, they have not taken the necessary ...
      Predicted Category: Debt collection
    
    New Complaint 183:
      Original Text: I recognize the importance of removing any incorrect information from my credit report as specified by FCRA 605B. Could you please review the attached...
      Predicted Category: Debt collection
    
    New Complaint 184:
      Original Text: Please see the attached letter; please note that I am submitting the complaint against ALL three bureaus : Equifax, XXXX and XXXX. 
    
    There are several...
      Predicted Category: Debt collection
    
    New Complaint 185:
      Original Text: Please see the attached letter; please note that I am submitting the complaint against ALL three bureaus : XXXX, Experian and XXXX. 
    
    There are severa...
      Predicted Category: Debt collection
    
    New Complaint 186:
      Original Text: I have asked proof of the appearance of the PERSONAL INFORMATION that I have no idea of....
      Predicted Category: Debt collection
    
    New Complaint 187:
      Original Text: XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX  Self Financial / SouthState Bank XXXX X...
      Predicted Category: Debt collection
    
    New Complaint 188:
      Original Text: For the past two months, I have tried to reach out to your company regarding these items that I want removed/deleted off my report and so far, nothing...
      Predicted Category: Debt collection
    
    New Complaint 189:
      Original Text: Received a letter from Transworld Systems Incorporated stating that I have a XXXX loan and have till the end of XXXX to sgin up for a program through ...
      Predicted Category: Debt collection
    
    New Complaint 190:
      Original Text: I have called Equifax several times between XX/XX/year> - XX/XX/year> asking to have a dispute from XXXX XXXX XXXX deleted from my account. I need the...
      Predicted Category: Debt collection
    
    New Complaint 191:
      Original Text: THE ACCOUNTS LISTED DOES NOT RELATE TO ANY TRANSACTIONS DONE BY ME....
      Predicted Category: Debt collection
    
    New Complaint 192:
      Original Text: Subject : DEMAND DELETE Bankruptcy Record AND ACCOUNTS WERE INCLUDED IN THE BANKRUPTCY Due to Privacy Violations and FCRA Non-Compliance AND TILA Bank...
      Predicted Category: Debt collection
    
    New Complaint 193:
      Original Text: Good day, please take this complaint seriously as my life and my future is at stake in here, this has been stressing me out for months because of your...
      Predicted Category: Debt collection
    
    New Complaint 194:
      Original Text: XXXX, Experian and XXXX XXXX are not responding to any of my requests to investigate, verify and remove accounts from my consumer credit reports that ...
      Predicted Category: Debt collection
    
    New Complaint 195:
      Original Text: I strongly recommend removing this suspicious entry from my credit report. It is not mine, and its presence is very concerning. Please take the necess...
      Predicted Category: Debt collection
    
    New Complaint 196:
      Original Text: Recently I did an investigation on my credit report which caused severe XXXX upon me and found unverifiable, invalidated, inaccurate, and questionable...
      Predicted Category: Debt collection
    
    New Complaint 197:
      Original Text: Creditor keeps calling my workplace after being told twice to not call my place of employment....
      Predicted Category: Debt collection
    
    New Complaint 198:
      Original Text: I have requested multiple times the proof the accounts appearing in my report without me knowing....
      Predicted Category: Debt collection
    
    New Complaint 199:
      Original Text: I asked for legitimate documents to prove the appearance of these items in the credit report....
      Predicted Category: Debt collection
    
    New Complaint 200:
      Original Text: Consumer Financial Protection Bureau XXXX XXXX XXXX XXXX XXXX, DC XXXX Subject : Request for Investigation of Credit Report Violations Dear Consumer F...
      Predicted Category: Debt collection
    
    New Complaint 201:
      Original Text: My account with Experian was never late! I have had exceptional payment history with XXXX XXXXXXXX and all payments were placed on Autopay. This late ...
      Predicted Category: Debt collection
    
    New Complaint 202:
      Original Text: In accordance with the Fair Credit Reporting Act XXXX XXXX XXXX has violated my rights. 15 U.S.C 1681 section 602 A. States I have the right to privac...
      Predicted Category: Debt collection
    
    New Complaint 203:
      Original Text: My Information was Exposed in a Data Breach and it is reporting incorrect information. It is showing a bankruptcy, and Reporting incorrect information...
      Predicted Category: Debt collection
    
    New Complaint 204:
      Original Text: My informaton was exposed in a data breach and there is inaccurate accounts information on my report....
      Predicted Category: Debt collection
    
    New Complaint 205:
      Original Text: My informaton was exposed in a data breach and there is inaccurate accounts information on my report....
      Predicted Category: Debt collection
    
    New Complaint 206:
      Original Text: I want to stress that I did not give written permission for these specific transactions to be included in my consumer report. At no point did I author...
      Predicted Category: Debt collection
    
    New Complaint 207:
      Original Text: THE ACCOUNTS LISTED DOES NOT RELATE TO ANY TRANSACTIONS DONE BY ME....
      Predicted Category: Debt collection
    
    New Complaint 208:
      Original Text: My credit reports are inaccurate. These inaccuracies are causing creditors to deny me credit. You have the duty to report accurate information about c...
      Predicted Category: Debt collection
    
    New Complaint 209:
      Original Text: I recently reviewed a copy of my credit report. I was shocked to see that I have been a victim of identity theft. I am in the process of buying a home...
      Predicted Category: Debt collection
    
    New Complaint 210:
      Original Text: XXXX  is reporting status on my account as open, charged off/written off and closed. All at the same time. How?...
      Predicted Category: Debt collection
    
    New Complaint 211:
      Original Text: Dear Consumer Financial Protection Bureau, I am writing to formally file a complaint against a collection agency that has placed a collection on my co...
      Predicted Category: Debt collection
    
    New Complaint 212:
      Original Text: Could you please take a moment to review and examine the attached documents? I believe your expertise will be invaluable in ensuring their accuracy an...
      Predicted Category: Debt collection
    
    New Complaint 213:
      Original Text: These accounts were not authorized or initiated by me, and it is unjust for me to be responsible for them. I implore you to promptly correct this erro...
      Predicted Category: Debt collection
    
    New Complaint 214:
      Original Text: XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXXXX/XX/XXXX ), Contents of Complaint : I am writing to request and investigation of the following accounts I hav...
      Predicted Category: Debt collection
    
    New Complaint 215:
      Original Text: I am writing to file a formal complaint and request your assistance regarding a debt that has been inaccurately reported by XXXX XXXX XXXX XXXX, accou...
      Predicted Category: Debt collection
    
    New Complaint 216:
      Original Text: I opened a Chase Freedom Unlimited Card in XXXX of 2024 as the promotion on their website says Earn a {$200.00} bonus after you spend {$500.00} on pur...
      Predicted Category: Debt collection
    
    New Complaint 217:
      Original Text: I was shocked to discover inaccurate accounts on my credit report tonight, and Im completely at a loss as to why theyre there. I check my credit repor...
      Predicted Category: Debt collection
    
    New Complaint 218:
      Original Text: I strongly recommend removing this suspicious entry from my credit report. It is not mine, and its presence is very concerning. Please take the necess...
      Predicted Category: Debt collection
    
    New Complaint 219:
      Original Text: Dear Consumer Financial Protection Bureau, I am writing to formally file a complaint against a collection agency that has placed a collection on my co...
      Predicted Category: Debt collection
    
    New Complaint 220:
      Original Text: Recently I did an investigation on my credit report which caused XXXX XXXX upon me and found unverifiable, invalidated, inaccurate, and questionable i...
      Predicted Category: Debt collection
    
    New Complaint 221:
      Original Text: I have requested multiple times the proof the accounts appearing in my report without me knowing. 
    
    XXXX XXXX XXXX XXXXXXXX,...
      Predicted Category: Debt collection
    
    New Complaint 222:
      Original Text: Recently I did an investigation on my credit report which caused XXXX XXXX upon me and found unverifiable, invalidated, inaccurate, and questionable i...
      Predicted Category: Debt collection
    
    New Complaint 223:
      Original Text: I was shocked and confused when I noticed inaccurate accounts on my credit report this evening. I have no idea why this happened. I check my credit re...
      Predicted Category: Debt collection
    
    New Complaint 224:
      Original Text: Recently I looked at a copy of my credit report and noticed several inaccuracies on my account. The account is hurting my ability to obtain credit. It...
      Predicted Category: Debt collection
    
    New Complaint 225:
      Original Text: My credit scores have really weighed on me because they're not enough to have a new loan approved. I am immensely stressed because there's so much ina...
      Predicted Category: Debt collection
    
    New Complaint 226:
      Original Text: My credit scores have really weighed on me because they're not enough to have a new loan approved. I am immensely stressed because there's so much ina...
      Predicted Category: Debt collection
    
    New Complaint 227:
      Original Text: The Fair Credit Reporting Act ( 15 U.S. Code 1681 ) says ( 1 ) The banking system is dependent upon fair and accurate credit reporting. Inaccurate cre...
      Predicted Category: Debt collection
    
    New Complaint 228:
      Original Text: have tried over XXXX times to set up account via website. I got to the end of the process and website says identity can not be verified. sends you bac...
      Predicted Category: Debt collection
    
    New Complaint 229:
      Original Text: The Fair Credit Reporting Act ( 15 U.S. Code 1681 ) says ( 1 ) The banking system is dependent upon fair and accurate credit reporting. Inaccurate cre...
      Predicted Category: Debt collection
    
    New Complaint 230:
      Original Text: I am writing to formally request the removal of several accounts and inquiries from my credit report. These accounts and inquiries are unauthorized an...
      Predicted Category: Debt collection
    
    New Complaint 231:
      Original Text: Good day, please take this complaint seriously as my life and my future is at stake in here, this has been stressing me out for months because of your...
      Predicted Category: Debt collection
    
    New Complaint 232:
      Original Text: Recently I looked at a copy of my credit report and noticed several inaccuracies on my account. This account is hurting my ability to obtain credit. I...
      Predicted Category: Debt collection
    
    New Complaint 233:
      Original Text: I am a victim of identity theft. The information listed below, which appears on my credit report, does not relate to any transactions that I have made...
      Predicted Category: Debt collection
    
    New Complaint 234:
      Original Text: I have requested proof that I were late on the accounts reported....
      Predicted Category: Debt collection
    
    New Complaint 235:
      Original Text: The inclusion of these unauthorized accounts and inquiries has caused considerable stress and concern regarding my financial well-being. It is imperat...
      Predicted Category: Debt collection
    
    New Complaint 236:
      Original Text: These accounts were not authorized or initiated by me, and it is unjust for me to be responsible for them. I implore you to promptly correct this erro...
      Predicted Category: Debt collection
    
    New Complaint 237:
      Original Text: I have requested proof that I were late on the accounts reported....
      Predicted Category: Debt collection
    
    New Complaint 238:
      Original Text: I have been paying aggressively on my student loans since the mid XXXX 's. I graduated in XXXX but began paying around XXXX. In XXXX I finished grad s...
      Predicted Category: Debt collection
    
    New Complaint 239:
      Original Text: Equifax, XXXX and XXXX XXXX are not responding to any of my requests to investigate, verify and remove accounts from my consumer credit reports that d...
      Predicted Category: Debt collection
    
    New Complaint 240:
      Original Text: My account with Experian was never late! I have had exceptional payment history with XXXX  XXXX and all payments were placed on XXXX. This late paymen...
      Predicted Category: Debt collection
    
    New Complaint 241:
      Original Text: My issue lies with XXXX XXXX being predatory and dishonest. My interest rate on my XXXX loan was 23.99 % I have attached my contract which was the onl...
      Predicted Category: Debt collection
    
    New Complaint 242:
      Original Text: I had a personal trainer agreement with XXXX, which I choose to end under the terms of the contract. I completed the online ONLY cancellation form, ta...
      Predicted Category: Debt collection
    
    New Complaint 243:
      Original Text: My account with TransUnion was never late! I have had exceptional payment history with XXXX XXXXXXXX and all payments were placed on Autopay. This lat...
      Predicted Category: Debt collection
    
    New Complaint 244:
      Original Text: My account with was never late! I have had exceptional payment history witH XXXX  XXXX and all payments were placed on Autopay. This late payment that...
      Predicted Category: Debt collection
    
    New Complaint 245:
      Original Text: I kindly demand the removal of these inaccurate accounts from my credit report, as I never authorized their use. It is essential that you swiftly eras...
      Predicted Category: Debt collection
    
    New Complaint 246:
      Original Text: It was shocking to discover inaccurate accounts on my credit report this evening. I have no idea how this happened. I check my credit report regularly...
      Predicted Category: Debt collection
    
    New Complaint 247:
      Original Text: Hi Equifax, I just get my credit report today and I saw 3 fraud inquiries that shall be delete from my credit report. I never authorized those inaccur...
      Predicted Category: Debt collection
    
    New Complaint 248:
      Original Text: Hi Equifax, I just get my credit report today and I saw 3 fraud inquiries that shall be delete from my credit report. I never authorized those inaccur...
      Predicted Category: Debt collection
    
    New Complaint 249:
      Original Text: Could you please take a moment to review and examine the attached documents? I believe your expertise will be invaluable in ensuring their accuracy an...
      Predicted Category: Debt collection
    
    New Complaint 250:
      Original Text: I strongly recommend removing this suspicious entry from my credit report. It is not mine, and its presence is very concerning. Please take the necess...
      Predicted Category: Debt collection
    
    New Complaint 251:
      Original Text: 15 U.S. code 1692c Without the prior consent of the consumer given directly to the debt collector or the express permission of a court of competent ju...
      Predicted Category: Debt collection
    
    New Complaint 252:
      Original Text: XXXX XXXX store credit card company XXXX XXXX XXXX XXXX XXXX for assistance XX/XX/year>. Was misled that the credit protection being offered at the ti...
      Predicted Category: Debt collection
    
    New Complaint 253:
      Original Text: In XXXX of 2024, I was behind in my loan agreement and called to set up some payment arrangements. After talking with a supervisor, she set me up with...
      Predicted Category: Debt collection
    
    New Complaint 254:
      Original Text: I want to stress that I did not give written permission for these specific transactions to be included in my consumer report. At no point did I author...
      Predicted Category: Debt collection
    
    New Complaint 255:
      Original Text: On XXXX submitted a direct dispute via certified mail to Equifax XXXX XXXX XXXX demanding they remove all inquiries and all fraudulent information fro...
      Predicted Category: Debt collection
    
    New Complaint 256:
      Original Text: Sent several letters to verify debt in XX/XX/year> & XX/XX/year> to confirm debt owed to company sending invoices to collect XXXX. Company repeatedly ...
      Predicted Category: Debt collection
    
    New Complaint 257:
      Original Text: I hope this letter finds you well. I am writing on behalf of a federal award recipient authorized by the XXXXXXXX XXXX XXXX  XXXX and XXXX XXXX XXXX X...
      Predicted Category: Debt collection
    
    New Complaint 258:
      Original Text: I asked for legitimate documents to prove the appearance of these items in the credit report. 
    
    XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX...
      Predicted Category: Debt collection
    
    New Complaint 259:
      Original Text: I sent a letter to Santander bank in XXXX of 2024. Santander bank settled a lawsuit involving predatory lending and my car was purchased at the time o...
      Predicted Category: Debt collection
    
    New Complaint 260:
      Original Text: On XXXX submitted a direct dispute via certified mail to XXXX XXXX XXXX Experian demanding they remove all inquiries and all fraudulent information fr...
      Predicted Category: Debt collection
    
    New Complaint 261:
      Original Text: I have a debt that has now been listed on my credit as unpaid but I never received any correspondence about this debt. I have always had medical insur...
      Predicted Category: Debt collection
    
    New Complaint 262:
      Original Text: I am writing to formally address Experian 's failure to remove specific items from my credit report, despite my prior dispute letter dated XX/XX/2024....
      Predicted Category: Debt collection
    
    New Complaint 263:
      Original Text: There is an account on my credit report that I do not recognize and believe it may be the result of fraud or identity theft....
      Predicted Category: Debt collection
    
    New Complaint 264:
      Original Text: I asked for legitimate documents to prove the appearance of these items in the credit report....
      Predicted Category: Debt collection
    
    New Complaint 265:
      Original Text: In accordance with the Fair Credit Reporting Act XXXX XXXX XXXX XXXX, has violated my rights. 15 U.S.C 1681 section 602 A. States I have the right to ...
      Predicted Category: Debt collection
    
    New Complaint 266:
      Original Text: I recently reviewed a copy of my credit report and noticed I had fraudulent account on my report. Please remove this account from my report. they are ...
      Predicted Category: Debt collection
    
    New Complaint 267:
      Original Text: On XXXX submitted a direct dispute via certified mail to XXXX TRANSUION XXXX XXXX demanding they remove all inquiries and all fraudulent information f...
      Predicted Category: Debt collection
    
    New Complaint 268:
      Original Text: I asked for legitimate documents to prove the appearance of these items in the credit report. 
    
    XXXX XXXX XXXX XXXXXXXX,...
      Predicted Category: Debt collection
    
    New Complaint 269:
      Original Text: This is so distressing and you have no idea how bad it is, Im an honest man working my way off honestly, then I look at my credit report all I see are...
      Predicted Category: Debt collection
    
    New Complaint 270:
      Original Text: I am writing to formally request the removal of several accounts and inquiries from my credit report. These accounts and inquiries are unauthorized an...
      Predicted Category: Debt collection
    
    New Complaint 271:
      Original Text: First I need to state my right to privacy is paramount. Im enforcing this right with this dispute letter. Per 15 U.S. Code 1681 ( a ) ( 4 ) you have f...
      Predicted Category: Debt collection
    
    New Complaint 272:
      Original Text: Recently I looked at a copy of my credit report and noticed several inaccuracies on my account. These account are hurting my ability to obtain credit....
      Predicted Category: Debt collection
    
    New Complaint 273:
      Original Text: On Tuesday XX/XX/year> at approximately XXXX XXXX the XXXX XXXX XXXX XXXX Department came to my place of residence and gave me paperwork on behalf of ...
      Predicted Category: Debt collection
    
    New Complaint 274:
      Original Text: I am writing to file a complaint against XXXX XXXX XXXX XXXX XXXX XXXX Collections. 
    
    In XXXX and XXXX, I was a tenant at XXXX XXXX XXXX XXXX XXXX Dur...
      Predicted Category: Debt collection
    
