from transformers import pipeline

# Load a zero-shot classification model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# List of mood categories
moods = [
    "Feel-good", "Heart-breaking", "Inspiring", "Curious", 
    "Nostalgic", "Guilty pleasure", "Hilarious", 
    "Scared", "Geeky", "Romantic"
]

# Example overviews
overviews = [
"A listless Wade Wilson toils away in civilian life with his days as the morally flexible mercenary, Deadpool, behind him. But when his homeworld faces an existential threat, Wade must reluctantly suit-up again with an even more reluctant Wolverine."    
]

# Predict mood scores
results = []
for overview in overviews:
    prediction = classifier(overview, moods, multi_label=True)
    raw_scores = {label: score for label, score in zip(prediction["labels"], prediction["scores"])}
    total_score = sum(raw_scores.values())

    # Normalize scores to add up to 100
    normalized_scores = {label: round((score / total_score) * 100, 2) for label, score in raw_scores.items()}
    results.append(normalized_scores)

# Display results
for i, result in enumerate(results):
    print(f"Overview {i+1} Mood Scores:")
    print(result)
