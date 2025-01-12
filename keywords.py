import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample mood keywords for each category
mood_keywords = {
    "Feel-good": ["heartwarming", "joy", "uplifting", "family", "bond"],
    "Heart-breaking": ["tragic", "loss", "emotional", "sad"],
    "Inspiring": ["heroic", "motivational", "courage", "overcome"],
    "Curious": ["mystery", "unexpected", "intriguing", "twist"],
    "Nostalgic": ["classic", "memories", "childhood", "past"],
    "Guilty pleasure": ["fun", "entertaining", "over-the-top", "wild"],
    "Hilarious": ["comedy", "funny", "laugh", "humor"],
    "Scared": ["fear", "threat", "tense", "danger"],
    "Geeky": ["sci-fi", "fantasy", "nerd", "tech"],
    "Romantic": ["love", "relationship", "affection", "romance"],
}

def calculate_mood_scores(overviews):
    vectorizer = TfidfVectorizer()
    overview_vectors = vectorizer.fit_transform(overviews)
    mood_scores = []
    
    for overview in overviews:
        scores = {}
        overview_vector = vectorizer.transform([overview])
        total_score = 0
        
        for mood, keywords in mood_keywords.items():
            keyword_vector = vectorizer.transform([" ".join(keywords)])
            score = cosine_similarity(overview_vector, keyword_vector).flatten()[0]
            scores[mood] = score
            total_score += score
        
        # Normalize scores to sum up to 100
        normalized_scores = {mood: round((score / total_score) * 100, 2) if total_score > 0 else 0 for mood, score in scores.items()}
        mood_scores.append(normalized_scores)
    
    return mood_scores

# Example dataset
data = {
    "Overview": [
        "A listless Wade Wilson toils away in civilian life with his days as the morally flexible mercenary, Deadpool, behind him. But when his homeworld faces an existential threat, Wade must reluctantly suit-up again with an even more reluctant Wolverine.",
        "Gru and Lucy and their girls \"Margo, Edith and Agnes\" welcome a new member to the Gru family, Gru Jr., who is intent on tormenting his dad. Gru also faces a new nemesis in Maxime Le Mal and his femme fatale girlfriend Valentina, forcing the family to go on the run.",
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Calculate mood scores
df["Mood Scores"] = calculate_mood_scores(df["Overview"])

# Adjust pandas display settings
pd.set_option('display.max_colwidth', None)  # Do not truncate column contents
pd.set_option('display.max_rows', None)     # Display all rows

# Display results
print(df)
