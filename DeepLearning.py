import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=10)

# Define mood categories (mapped to integer labels for classification)
moods = [
    "Feel-good", "Heart-breaking", "Inspiring", "Curious", 
    "Nostalgic", "Guilty pleasure", "Hilarious", 
    "Scared", "Geeky", "Romantic"
]

# Example dataset of movie overviews with mood labels (adjust your data format accordingly)
data = {
    "Overview": [
        "A listless Wade Wilson toils away in civilian life with his days as the morally flexible mercenary, Deadpool, behind him. But when his homeworld faces an existential threat, Wade must reluctantly suit-up again with an even more reluctant Wolverine.",
        "Gru and Lucy and their girls \"Margo, Edith and Agnes\" welcome a new member to the Gru family, Gru Jr., who is intent on tormenting his dad. Gru also faces a new nemesis in Maxime Le Mal and his femme fatale girlfriend Valentina, forcing the family to go on the run.",
    ],
    "Feel-good": [1, 0],
    "Heart-breaking": [0, 1],
    "Inspiring": [1, 0],
    "Curious": [1, 0],
    "Nostalgic": [0, 1],
    "Guilty pleasure": [1, 0],
    "Hilarious": [1, 1],
    "Scared": [0, 1],
    "Geeky": [0, 0],
    "Romantic": [0, 1]
}

# Convert data into a DataFrame
df = pd.DataFrame(data)

# Split dataset into training and validation
train_data, val_data = train_test_split(df, test_size=0.2)

# Prepare the data for BERT
def tokenize_function(examples):
    return tokenizer(examples['Overview'], padding="max_length", truncation=True)

train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)

# Tokenize the data
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Convert labels into a list for multi-label classification
def convert_labels(example):
    example['labels'] = np.array([example[mood] for mood in moods])
    return example

train_dataset = train_dataset.map(convert_labels)
val_dataset = val_dataset.map(convert_labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Create a Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)

# Predict mood scores for new overviews
def predict_mood(overview):
    inputs = tokenizer(overview, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.sigmoid(logits).squeeze().numpy()  # Sigmoid to get probabilities for each label
    return {mood: prob for mood, prob in zip(moods, probs)}

# Example overview to predict mood scores
overview = "Gru and his family face a new adventure and unexpected challenges together."
mood_scores = predict_mood(overview)
print(mood_scores)
