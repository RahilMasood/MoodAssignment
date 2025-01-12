import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
import scipy
import joblib

file_path = r'C:\Users\shez8\Desktop\NYTimes\2minMovie\Updated_Movie_Mood_Scores_Fixed.csv'
df = pd.read_csv(file_path)

# Preprocess the genres column to handle multiple genres per movie
df['genres'] = df['genres'].fillna('Unknown').str.lower().str.split(',')

# Use MultiLabelBinarizer to create genre features
mlb = MultiLabelBinarizer()
genre_features = mlb.fit_transform(df['genres'])
genre_columns = mlb.classes_

# Text vectorization for the overview column using TF-IDF
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
overview_features = tfidf.fit_transform(df['overview'].fillna(''))

# Weight the genre features more heavily
genre_weight = 15
weighted_genre_features = genre_features * genre_weight

# Combine genre features and TF-IDF features
X_combined = scipy.sparse.hstack((overview_features, weighted_genre_features))

y = df[['Feel-good', 'Heart-breaking', 'Inspiring', 'Curious', 'Nostalgic',
        'Guilty pleasure', 'Hilarious', 'Scared', 'Geeky', 'Loving']]

X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

model = MultiOutputRegressor(Ridge(random_state=42))
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
#mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
#print("Mean Squared Error for each mood:", mse)

model_file = 'mood_prediction_model_genre_weighted.pkl'
tfidf_file = 'tfidf_vectorizer_weighted.pkl'
mlb_file = 'genre_binarizer_weighted.pkl'

joblib.dump(model, model_file)
joblib.dump(tfidf, tfidf_file)
joblib.dump(mlb, mlb_file)

def predict_mood_scores(genre_list, overview):
    overview_tfidf = tfidf.transform([overview])
    genre_features = mlb.transform([genre_list])
    weighted_genre_features = genre_features * genre_weight
    combined_features = scipy.sparse.hstack((overview_tfidf, weighted_genre_features))
    predicted_scores = model.predict(combined_features)[0]
    total = sum(predicted_scores)
    normalized_scores = {mood: (score / total) * 100 for mood, score in zip(y.columns, predicted_scores)}

    return normalized_scores

example_genre = ['comedy', 'animation', 'adventure']
example_overview = '''Gru and Lucy and their girls \"Margo, Edith and Agnes\" welcome a new member to 
                    the Gru family, Gru Jr., who is intent on tormenting his dad. Gru also faces a new 
                    nemesis in Maxime Le Mal and his femme fatale girlfriend Valentina, forcing the family 
                    to go on the run.'''
predicted_scores = predict_mood_scores(example_genre, example_overview)

print("\nPredicted Mood Scores:")
for mood, score in predicted_scores.items():
    print(f"{mood}: {score:.2f}")
