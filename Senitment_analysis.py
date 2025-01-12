import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Ensure you have the necessary resources for VADER
nltk.download("vader_lexicon")

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Adjust pandas settings to display full content
pd.set_option("display.max_colwidth", None)  # Show full column content
pd.set_option("display.expand_frame_repr", False)  # Avoid wrapping rows
pd.set_option("display.max_rows", None)  # Show all rows

# Define mood categories and their sentiment mapping
mood_mapping = {
    "Feel-good": "pos",
    "Heart-breaking": "neg",
    "Inspiring": "pos",
    "Curious": "neu",
    "Nostalgic": "neu",
    "Guilty pleasure": "pos",
    "Hilarious": "pos",
    "Scared": "neg",
    "Geeky": "neu",
    "Romantic": "pos",
}

# Example dataset of movie overviews
data = {
    "Overview": [
        "A listless Wade Wilson toils away in civilian life with his days as the morally flexible mercenary, Deadpool, behind him. But when his homeworld faces an existential threat, Wade must reluctantly suit-up again with an even more reluctant Wolverine.",
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Function to calculate mood scores
def calculate_mood_scores(overview):
    sentiment = sia.polarity_scores(overview)
    scores = {}
    
    for mood, sentiment_type in mood_mapping.items():
        scores[mood] = sentiment[sentiment_type]
    
    # Normalize scores to sum to 100
    total = sum(scores.values())
    normalized_scores = {mood: round((score / total) * 100, 2) if total > 0 else 0 for mood, score in scores.items()}
    
    return normalized_scores

# Apply sentiment analysis to each overview
df["Mood Scores"] = df["Overview"].apply(calculate_mood_scores)

# Display the full DataFrame
print(df)
