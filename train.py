

import pandas as pd
import numpy as np
import re
import time
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             hamming_loss, accuracy_score, classification_report)
import os

# --- NLTK Data Download ---
try:
    stopwords.words('english')
except LookupError:
    print("NLTK stopwords not found. Downloading...")
    nltk.download('stopwords')
    print("Download complete.")

# --- Configuration ---
INPUT_CSV_FILE = 'stackoverflow_data.csv'
TAG_SEPARATOR = '|'
CSV_SEPARATOR = ','
MIN_TAG_FREQUENCY = 3 
MAX_FEATURES_TFIDF = 10000 # Max vocabulary size for TF-IDF
RANDOM_STATE = 42
TOP_N_TAGS_PLOT = 30 

# --- Plotting Setup ---
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)

def plot_top_tags(tag_counts, top_n, title):
    """Plots the frequency of the top N tags."""
    if tag_counts is None or tag_counts.empty:
        print(f"Warning: No data to plot for '{title}'.")
        return
    top_tags = tag_counts.head(top_n)
    plt.figure(figsize=(15, 8))
    sns.barplot(x=top_tags.values, y=top_tags.index, palette='viridis')
    plt.title(title, fontsize=16)
    plt.xlabel("Frequency", fontsize=12)
    plt.ylabel("Tags", fontsize=12)
    for index, value in enumerate(top_tags.values):
        plt.text(value, index, f' {value}', va='center')
    plt.tight_layout()
    plt.show()

def evaluate_model(y_true, y_pred, labels, model_name="Model"):
    """Calculates and prints standard evaluation metrics."""
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_samples = f1_score(y_true, y_pred, average='samples', zero_division=0)
    subset_accuracy = accuracy_score(y_true, y_pred)
    hamming = hamming_loss(y_true, y_pred)

    print(f"\n--- Evaluation Metrics for {model_name} ---")
    print(f"F1 Score (Micro): {f1_micro:.4f}")
    print(f"Precision (Micro): {precision_micro:.4f}")
    print(f"Recall (Micro): {recall_micro:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"F1 Score (Samples): {f1_samples:.4f}")
    print(f"Subset Accuracy: {subset_accuracy:.4f}")
    print(f"Hamming Loss: {hamming:.4f}")
    print("------------------------------------")

    print(f"\nClassification Report ({model_name}):")
    report_df = None
    if len(labels) > 0:
        try:
            y_true_np = np.asarray(y_true)
            y_pred_np = np.asarray(y_pred)
            report = classification_report(y_true_np, y_pred_np, target_names=labels, zero_division=0, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            if 'support' in report_df.columns:
                 report_df['support'] = report_df['support'].astype(int)
            print(report_df.round(4))
        except Exception as e:
            print(f"Could not generate classification report: {e}")
    else:
        print("No labels found for report.")
    return report_df

def preprocess_text(text):
    """Cleans and preprocesses text data."""
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'`<code>.*?</code>`', ' ', text)
    text = re.sub(r'```.*?```', ' ', text, flags=re.DOTALL)
    text = re.sub(r'[^a-z\s+#-]', '', text) # Keep letters, spaces, #, +, -
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 1]
    return ' '.join(words)

# --- 1. Data Loading ---
print(f"Loading data from {INPUT_CSV_FILE}...")
if not os.path.exists(INPUT_CSV_FILE):
     raise FileNotFoundError(f"Error: Input file not found: '{INPUT_CSV_FILE}'")
try:
    data = pd.read_csv(INPUT_CSV_FILE, sep=CSV_SEPARATOR)
    if not all(col in data.columns for col in ['Title', 'Body', 'Tags']):
         raise ValueError("CSV must contain 'Title', 'Body', 'Tags' columns.")
    print(f"Loaded {len(data)} entries.")
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()
data.dropna(subset=['Title', 'Body', 'Tags'], inplace=True)
print(f"Entries after dropping NA: {len(data)}")
if len(data) < 10:
    print("Not enough valid data. Exiting.")
    exit()

# --- Initial Data Exploration & Visualization ---
print("\n--- Initial Data Exploration ---")
data['TitleLength'] = data['Title'].astype(str).apply(len)
data['BodyLength'] = data['Body'].astype(str).apply(len)
data['CombinedLength'] = data['TitleLength'] + data['BodyLength']

plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1); sns.histplot(data['TitleLength'], bins=30, kde=True); plt.title('Title Length Distribution')
plt.subplot(1, 3, 2); sns.histplot(data['BodyLength'], bins=50, kde=True); plt.title('Body Length Distribution')
plt.subplot(1, 3, 3); sns.histplot(data['CombinedLength'], bins=50, kde=True); plt.title('Combined Text Length Distribution')
plt.tight_layout(); plt.show()

data['Tags'] = data['Tags'].astype(str)
data['RawTagList'] = data['Tags'].apply(lambda x: [tag.strip() for tag in x.split(TAG_SEPARATOR) if tag.strip()])
data['NumRawTags'] = data['RawTagList'].apply(len)

plt.figure(figsize=(10, 6))
sns.countplot(x='NumRawTags', data=data, palette='magma', order = data['NumRawTags'].value_counts().index)
plt.title('Tags per Question Distribution (Raw)'); plt.xlabel('Number of Tags'); plt.ylabel('Number of Questions')
ax = plt.gca();
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=9, color='black', xytext=(0, 5), textcoords='offset points')
plt.show()

all_raw_tags = [tag for tags in data['RawTagList'] for tag in tags]
raw_tag_counts = None
if all_raw_tags:
    raw_tag_counts = pd.Series(all_raw_tags).value_counts()
    print(f"\nTotal unique raw tags: {len(raw_tag_counts)}")
    plot_top_tags(raw_tag_counts, TOP_N_TAGS_PLOT, f'Top {TOP_N_TAGS_PLOT} Raw Tags')
else:
    print("No raw tags found.")

# --- 2. Text Preprocessing ---
print("\nPreprocessing text data...")
stop_words = set(stopwords.words('english'))
data['CombinedText'] = data['Title'] + ' ' + data['Body']
data['ProcessedText'] = data['CombinedText'].apply(preprocess_text)
print("Sample processed text:")
print(data[['CombinedText', 'ProcessedText']].head())

# --- Word Cloud ---
print("\nGenerating Word Cloud...")
long_string = ' '.join(data['ProcessedText'].dropna())
if long_string:
    try:
        wordcloud = WordCloud(width=1000, height=500, background_color='white', max_words=150, collocations=False)
        wordcloud.generate(long_string)
        plt.figure(figsize=(15, 8)); plt.imshow(wordcloud, interpolation='bilinear'); plt.axis("off"); plt.title("Word Cloud of Processed Text", fontsize=16); plt.show()
    except Exception as e:
        print(f"Could not generate word cloud: {e}")
else:
    print("Not enough text for word cloud.")

# --- 3. Tag Processing & Filtering ---
print("\nProcessing and filtering tags...")
data['TagList'] = data['RawTagList']
all_tags = [tag for tags in data['TagList'] for tag in tags]
if not all_tags: print("Error: No tags found."); exit()

tag_counts = pd.Series(all_tags).value_counts()
frequent_tags = tag_counts[tag_counts >= MIN_TAG_FREQUENCY].index.tolist()
print(f"\nTotal unique tags: {len(tag_counts)}")
print(f"Using {len(frequent_tags)} tags with frequency >= {MIN_TAG_FREQUENCY}")
if not frequent_tags: print(f"Error: No tags meet min frequency {MIN_TAG_FREQUENCY}. Exiting."); exit()

def filter_tags(tags): return [tag for tag in tags if tag in frequent_tags]
data['FilteredTagList'] = data['TagList'].apply(filter_tags)
data['NumFilteredTags'] = data['FilteredTagList'].apply(len)
initial_count = len(data)
data = data[data['NumFilteredTags'] > 0].copy()
print(f"Data shape after tag filtering: {data.shape} (Removed {initial_count - len(data)})")
if len(data) == 0: print("Error: No data remaining after filtering tags."); exit()

# --- Filtered Tag Visualization ---
print("\n--- Filtered Tag Analysis ---")
plt.figure(figsize=(10, 6))
sns.countplot(x='NumFilteredTags', data=data, palette='viridis', order = data['NumFilteredTags'].value_counts().index)
plt.title('Tags per Question Distribution (Filtered)'); plt.xlabel(f'Number of Tags (Freq >= {MIN_TAG_FREQUENCY})'); plt.ylabel('Number of Questions')
ax = plt.gca();
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=9, color='black', xytext=(0, 5), textcoords='offset points')
plt.show()

all_filtered_tags = [tag for tags in data['FilteredTagList'] for tag in tags]
filtered_tag_counts = None
if all_filtered_tags:
    filtered_tag_counts = pd.Series(all_filtered_tags).value_counts()
    plot_top_tags(filtered_tag_counts, TOP_N_TAGS_PLOT, f'Top {TOP_N_TAGS_PLOT} Filtered Tags (Labels)')
else:
    print("No filtered tags found.")

# --- 4. Label Encoding ---
print("\nEncoding labels...")
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(data['FilteredTagList'])
labels = mlb.classes_
print(f"Shape of label matrix (y): {y.shape}")
print(f"Number of unique classes (tags): {len(labels)}")

# --- Stratification Check ---
min_samples_per_tag = 0
if y.shape[0] > 0 and y.shape[1] > 0: min_samples_per_tag = np.min(y.sum(axis=0))
else: print("Warning: Label matrix 'y' is empty."); exit()
use_stratify = True
if min_samples_per_tag < 2: print(f"\nWarning: Min samples per tag is {min_samples_per_tag}. Disabling stratification."); use_stratify = False
elif y.shape[0] < 4: print(f"\nWarning: Only {y.shape[0]} samples. Disabling stratification."); use_stratify = False

# --- Tag Co-occurrence Heatmap ---
if len(labels) > 1 and len(labels) <= 50:
    print("\nPlotting tag co-occurrence heatmap...")
    co_occurrence = np.dot(y.T, y); np.fill_diagonal(co_occurrence, 0)
    co_occurrence_df = pd.DataFrame(co_occurrence, index=labels, columns=labels)
    plt.figure(figsize=(15, 15)); sns.heatmap(co_occurrence_df, cmap="viridis", linewidths=.5, annot=False)
    plt.title('Tag Co-occurrence Heatmap', fontsize=16); plt.xticks(rotation=90, fontsize=8); plt.yticks(rotation=0, fontsize=8); plt.tight_layout(); plt.show()
elif len(labels) > 50: print("\nSkipping tag co-occurrence heatmap (>50 tags).")
else: print("\nSkipping tag co-occurrence heatmap (<2 tags).")

# --- 5. Train/Test Split ---
print("\nSplitting data into train/test sets...")
X = data['ProcessedText']; y_filtered = y
if len(X) < 2: print("Error: Fewer than 2 samples."); exit()
stratify_option = y_filtered if use_stratify else None
split_info = f"(Stratify: {'Yes' if use_stratify else 'No'})"
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y_filtered, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify_option)
except ValueError as e:
    print(f"\nError during split {split_info}: {e}. Trying without stratification...")
    try: X_train, X_test, y_train, y_test = train_test_split(X, y_filtered, test_size=0.2, random_state=RANDOM_STATE, stratify=None)
    except Exception as final_e: print(f"Fatal Error: Could not split data: {final_e}"); exit()
print(f"Train size: {len(X_train)}, Test size: {len(X_test)} {split_info}")

# --- 6. Feature Extraction (TF-IDF) ---
print("\nExtracting TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(max_features=MAX_FEATURES_TFIDF, ngram_range=(1, 2), min_df=3, max_df=0.9)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
print(f"TF-IDF matrix shape (Train): {X_train_tfidf.shape}, (Test): {X_test_tfidf.shape}")
try: print(f"Vocabulary size: {len(tfidf_vectorizer.get_feature_names_out())}")
except Exception: pass

# --- 7. Model Definition ---
print("\nDefining models...")
classifiers = {
    "Logistic Regression": OneVsRestClassifier(
        LogisticRegression(solver='liblinear', C=1.0, random_state=RANDOM_STATE, class_weight='balanced'), n_jobs=-1
    ),
    "Linear SVC": OneVsRestClassifier(
        LinearSVC(C=1.0, class_weight='balanced', random_state=RANDOM_STATE, dual=False, max_iter=2000), n_jobs=-1
    )
}
results = {}; per_tag_reports = {}

# --- 8. Model Training & Evaluation Loop ---
print("\n--- Training and Evaluating Models ---")
for name, classifier in classifiers.items():
    print(f"\nTraining {name}...")
    start_time = time.time()
    classifier.fit(X_train_tfidf, y_train)
    train_time = time.time() - start_time
    print(f"Training {name} completed in {train_time:.2f}s.")

    print(f"Evaluating {name}...")
    start_time = time.time()
    y_pred = classifier.predict(X_test_tfidf)
    predict_time = time.time() - start_time
    print(f"Prediction with {name} completed in {predict_time:.2f}s.")

    results[name] = {'predictions': y_pred, 'train_time': train_time, 'predict_time': predict_time}
    report_df = evaluate_model(y_test, y_pred, labels, model_name=name)
    per_tag_reports[name] = report_df

# --- 9. Comparison Visualization ---
print("\n--- Model Comparison Summary ---")
print("\nMicro F1 Scores & Timings:")
for name, result_data in results.items():
     f1_micro = f1_score(y_test, result_data['predictions'], average='micro', zero_division=0)
     print(f"  {name}: {f1_micro:.4f} (Train: {result_data['train_time']:.2f}s, Predict: {result_data['predict_time']:.2f}s)")

metric_names = ['f1-score', 'precision', 'recall']
average_types = ['micro avg', 'weighted avg']
comparison_data = []
for avg_type in average_types:
    for name, report_df in per_tag_reports.items():
        if report_df is not None and avg_type in report_df.index:
            metrics = report_df.loc[avg_type, metric_names].to_dict()
            metrics['model'] = name; metrics['average_type'] = avg_type.replace(' avg','')
            comparison_data.append(metrics)

if comparison_data:
    comparison_df = pd.DataFrame(comparison_data)
    plt.figure(figsize=(15, 7)); sns.barplot(data=comparison_df, x='average_type', y='f1-score', hue='model', palette='Spectral')
    plt.title('Comparison of F1 Scores (Micro vs Weighted)', fontsize=16); plt.ylabel('F1 Score'); plt.xlabel('Averaging Method'); plt.ylim(0, 1); plt.legend(title='Model', loc='lower right'); plt.tight_layout(); plt.show()

if filtered_tag_counts is not None and len(per_tag_reports) > 1:
     top_tags_list = filtered_tag_counts.head(TOP_N_TAGS_PLOT).index.tolist()
     tag_comparison_data = []
     for tag in top_tags_list:
         for name, report_df in per_tag_reports.items():
             if report_df is not None and tag in report_df.index:
                 f1 = report_df.loc[tag, 'f1-score']; tag_comparison_data.append({'tag': tag, 'model': name, 'f1-score': f1})
     if tag_comparison_data:
         tag_comparison_df = pd.DataFrame(tag_comparison_data)
         avg_f1_per_tag = tag_comparison_df.groupby('tag')['f1-score'].mean().sort_values(ascending=False)
         tags_to_plot = avg_f1_per_tag.head(15).index
         plot_df = tag_comparison_df[tag_comparison_df['tag'].isin(tags_to_plot)]
         plt.figure(figsize=(15, 10)); sns.barplot(data=plot_df, y='tag', x='f1-score', hue='model', palette='Spectral', order=tags_to_plot)
         plt.title(f'Comparison of F1 Scores for Top {len(tags_to_plot)} Tags', fontsize=16); plt.xlabel('F1 Score'); plt.ylabel('Tag'); plt.xlim(0, 1); plt.legend(title='Model'); plt.tight_layout(); plt.show()

# --- 10. Prediction Example ---
best_model_name = "Linear SVC" #default
if best_model_name not in classifiers: best_model_name = list(classifiers.keys())[0]
print(f"\n--- Predicting Tags for New Question using {best_model_name} ---")
best_classifier = classifiers[best_model_name]

new_question_title = "Reading specific columns from csv using pandas"
new_question_body = "How do I load only column 'A' and column 'C' from a large csv file into a pandas dataframe to save memory?"
new_question_text = new_question_title + " " + new_question_body

processed_new_question = preprocess_text(new_question_text)
print(f"Processed new question: '{processed_new_question}'")
new_question_tfidf = tfidf_vectorizer.transform([processed_new_question])
print(f"Shape of new question TF-IDF: {new_question_tfidf.shape}")

new_question_pred = best_classifier.predict(new_question_tfidf)
predicted_tags_new = mlb.inverse_transform(new_question_pred)

print(f"\nPredicted Tags (using {best_model_name}):")
if predicted_tags_new and predicted_tags_new[0]: print(list(predicted_tags_new[0]))
else: print("No tags predicted.")

