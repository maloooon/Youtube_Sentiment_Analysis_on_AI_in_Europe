import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification



# Step 1: Load the model and tokenizer
model_path = '/Users/marlon/VS-Code-Projects/Youtube/sentiment_model_fine_tuned_distilbert_english'
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)


# Step 2: Prepare the input data
# Example input text
texts = ["maybe ai will do a better job of taking care of the planet, we're not a hard act follow", "AI will destroy this world, horrible.", "AI will be neither beneficial nor bad. I do not care for its effect really.", "AI... was soll ich dazu sagen? Ich weiß nicht, ob es gut oder schlecht ist."]

# Step 3: Tokenize the input data (same as in training)
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)

# Leverage GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs = {key: value.to(device) for key, value in inputs.items()}

# Step 4: Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits



# Step 5: Interpret the results
predictions = torch.argmax(logits, dim=-1)


# Label mapping for three classes
label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
predicted_labels = [label_map[prediction.item()] for prediction in predictions]

# Print the results
for text, label in zip(texts, predicted_labels):
    print(f"Text: {text}\nSentiment: {label}\n")



