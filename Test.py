import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import joblib

# Load the dataset
file_path = r'C:\Users\shez8\Desktop\NYTimes\2minMovie\Final_Normalized_Movie_Data_with_Mood_Scores.csv'
df = pd.read_csv(file_path)

# Features (input) and target (output)
X = df['overview'].fillna('')  # Handle missing values
y = df[['Feel-good', 'Heart-breaking', 'Inspiring', 'Curious', 'Nostalgic',
        'Guilty pleasure', 'Hilarious', 'Scared', 'Geeky', 'Loving']]

# Text vectorization using TF-IDF
tfidf = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train XGBoost models for each target
models = {}
for mood in y.columns:
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train[mood])
    models[mood] = model

# Save models and vectorizer
for mood, model in models.items():
    joblib.dump(model, f'{mood}_xgb_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

# Define a function to predict mood scores
def predict_mood_scores(overview):
    overview_tfidf = tfidf.transform([overview])
    mood_scores = {}
    for mood, model in models.items():
        mood_scores[mood] = model.predict(overview_tfidf)[0]
    return mood_scores

# Example usage
example_overview = "Gru and Lucy and their girls \"Margo, Edith and Agnes\" welcome a new member to the Gru family, Gru Jr., who is intent on tormenting his dad. Gru also faces a new nemesis in Maxime Le Mal and his femme fatale girlfriend Valentina, forcing the family to go on the run."
predicted_scores = predict_mood_scores(example_overview)

# Normalize scores to sum to 100
total = sum(predicted_scores.values())
normalized_scores = {mood: (score / total) * 100 for mood, score in predicted_scores.items()}

# Display the normalized mood scores
print("\nPredicted Mood Scores:")
for mood, score in normalized_scores.items():
    print(f"{mood}: {score:.2f}")
