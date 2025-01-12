import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
file_path = r'C:\Users\shez8\Desktop\NYTimes\2minMovie\Final_Normalized_Movie_Data_with_Mood_Scores.csv'  
df = pd.read_csv(file_path)

# Features (input) and target (output)
X = df['overview'].fillna('')
y = df[['Feel-good', 'Heart-breaking', 'Inspiring', 'Curious', 'Nostalgic',
        'Guilty pleasure', 'Hilarious', 'Scared', 'Geeky', 'Loving']]

# Text vectorization using TF-IDF
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = tfidf.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train the model using Ridge regression for faster performance
model = MultiOutputRegressor(Ridge(random_state=42))
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
#mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
#print("Mean Squared Error for each mood:", mse)

# Save the model and vectorizer for future use
model_file = 'mood_prediction_model_ridge.pkl'
vectorizer_file = 'tfidf_vectorizer.pkl'
joblib.dump(model, model_file)
joblib.dump(tfidf, vectorizer_file)

model = joblib.load(model_file)
tfidf = joblib.load(vectorizer_file)

def predict_mood_scores(overview):
    overview_tfidf = tfidf.transform([overview])
    predicted_scores = model.predict(overview_tfidf)[0]
    mood_scores = {
        'Feel-good': predicted_scores[0],
        'Heart-breaking': predicted_scores[1],
        'Inspiring': predicted_scores[2],
        'Curious': predicted_scores[3],
        'Nostalgic': predicted_scores[4],
        'Guilty pleasure': predicted_scores[5],
        'Hilarious': predicted_scores[6],
        'Scared': predicted_scores[7],
        'Geeky': predicted_scores[8],
        'Loving': predicted_scores[9],
    }
    return mood_scores

# Take input from the user
movie_overview = input("Enter a movie overview: ")

# Predict mood scores
mood_scores = predict_mood_scores(movie_overview)

# Display the mood scores
print("\nPredicted Mood Scores:")
for mood, score in mood_scores.items():
    print(f"{mood}: {score:.2f}")
