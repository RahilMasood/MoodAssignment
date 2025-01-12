# Movie Mood Predictor

## Project Overview
The **Movie Mood Predictor** is a machine learning-based project designed to predict mood scores for movies based on their genres and plot overviews. The model assigns scores to different moods, such as *Feel-good*, *Heart-breaking*, *Inspiring* and more based on the input description of the movie. This project enhances movie recommendations by classifying movies based on emotions rather than conventional metrics like ratings or genres alone.

---

## Features
1. **Multi-Mood Prediction**: Predicts scores for 10 moods:
   - Feel-good
   - Heart-breaking
   - Inspiring
   - Curious
   - Nostalgic
   - Guilty pleasure
   - Hilarious
   - Scared
   - Geeky
   - Loving

2. **Weighted Genre Influence**: The genres of a movie are assigned higher weights to improve prediction accuracy by emphasizing their importance.

3. **Text Analysis with TF-IDF**: The movie overviews are vectorized using TF-IDF to extract meaningful features from text descriptions.

4. **Customizable Input**: Users can input a list of genres and a movie overview to get real-time mood predictions.

5. **Scalable Architecture**: The model combines genre and text data using a sparse matrix representation, making it memory-efficient and scalable for large datasets.

---

## Workflow
1. **Data Preprocessing**:
   - Genres are split into a list format and processed using a `MultiLabelBinarizer`.
   - Overviews are vectorized using a `TfidfVectorizer`.

2. **Feature Engineering**:
   - Genres are weighted to emphasize their impact on mood prediction.
   - TF-IDF features and weighted genre features are combined into a single feature matrix.

3. **Model Training**:
   - A `MultiOutputRegressor` with a `Ridge` model is used to predict multiple mood scores simultaneously.
   - The model is trained on a custom-built dataset of movies with pre-labeled mood scores.

4. **Prediction**:
   - Inputs: List of genres and a textual overview.
   - Outputs: Normalized mood scores in percentages for each mood.

---

## Dataset
The dataset used for this project was built manually using web scraping bots. It includes:
- Movies with pre-assigned mood scores.
- Genres for each movie.
- Movie plot overviews.

The dataset was cleaned, preprocessed, and labeled to ensure high-quality training data for the model.

---

## Model and Files
- `mood_prediction_model_genre_weighted.pkl`: Trained Ridge Regression model for multi-output mood prediction.
- `tfidf_vectorizer_weighted.pkl`: TF-IDF vectorizer trained on movie overviews.
- `genre_binarizer_weighted.pkl`: MultiLabelBinarizer for encoding movie genres.

---

## Contributors
- **Rahil Masood** ([@RahilMasood](https://github.com/RahilMasood))

Feel free to contribute, report issues, or suggest improvements!
