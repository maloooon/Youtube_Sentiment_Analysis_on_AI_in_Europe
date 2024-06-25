import torch
from transformers import BertTokenizer, BertForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer, pipeline
# TODO :

# possibly padding true 

LANGUAGE = 'italian'



if LANGUAGE == 'german':
    model_path = '/Users/marlon/VS-Code-Projects/Youtube/sentiment_model_finetuned_german'
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    texts = ["Erneuter Streik in der S-Bahn", "Ich liebe es, wenn die Sonne scheint", "Ich hasse es, wenn es regnet"]
    inputs = tokenizer(texts, return_tensors="pt", padding='max_length', truncation=True, max_length=64)

    # Predict sentiment
    with torch.no_grad():
        outputs = model(**inputs)

    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
  

    # Define sentiment classes
    sentiment_classes = ['negative', 'neutral', 'positive']

    # Get predictions for each input text
    predicted_classes = probabilities.argmax(dim=1)

    # Print the predicted sentiment for each input text
    for i, predicted_class in enumerate(predicted_classes):
        print(f"Text: {texts[i]}")
        print(f"Predicted Sentiment: {sentiment_classes[predicted_class]}")
        print(f"Probabilities: {probabilities[i]}")


if LANGUAGE == 'spanish':
    
  #  threshold = 0.5 # for classification of negative or positive # TODO : check what it actually does
    model_path = '/Users/marlon/VS-Code-Projects/Youtube/sentiment_model_finetuned_spanish'
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    # Examplary spanish youtube comments
    texts = ["Me encanta este video, es muy interesante", "No me gusta este video, es aburrido", "No estoy seguro de si me gusta este video o no"]
    inputs = tokenizer(texts, return_tensors="pt", padding='max_length', truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    probabilities = torch.softmax(logits, dim=1)
    print(probabilities)
    predicted_classes = torch.argmax(logits, dim=1)

    sentiment_classes = ['negative', 'positive']

    # Print the predicted sentiment for each input text
    for i, predicted_class in enumerate(predicted_classes):
        print(f"Text: {texts[i]}")
        print(f"Predicted Sentiment: {sentiment_classes[predicted_class]}")
        print(f"Probabilities: {probabilities[i]}")

    
  #  if probabilities[predicted_class] <= threshold and predicted_class == 1:
  #      predicted_class = 0

   # return bool(predicted_class), probabilities


if LANGUAGE == 'french':
    model_path = '/Users/marlon/VS-Code-Projects/Youtube/sentiment_model_finetuned_french'
    model=AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer=AutoTokenizer.from_pretrained(model_path)
    analyzer = pipeline(
    task='text-classification', model=model, tokenizer=tokenizer)

    texts = ["J'adore cette vidéo, elle est très intéressante", "Je n'aime pas cette vidéo, elle est ennuyeuse", "Je ne suis pas sûr si j'aime cette vidéo ou non"]

    
    for text in texts:
        result = analyzer(text, return_all_scores=True)
        print(result)

 



if LANGUAGE == 'italian':
    model_path = '/Users/marlon/VS-Code-Projects/Youtube/sentiment_model_finetuned_italian'
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    analyzer = pipeline("text-classification", model = model, tokenizer = tokenizer)

    texts = ["Adoro questo video, è molto interessante", "Non mi piace questo video, è noioso", "Non sono sicuro se mi piace questo video o no"]

    for text in texts:
        result = analyzer(text, return_all_scores=True)
        print(result)











