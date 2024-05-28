from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset


# Instantiate the tokenizer for multilingual data
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')


def tokenize_function(examples, tokenizer=tokenizer):
    """
    Function to tokenize the data.
    examples : data to tokenize ; dict
    tokenizer : tokenizer to use ; DistilBertTokenizer
    """
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)



def DistilBertModel(train_comments, train_labels, 
                    val_comments, val_labels,
                    batch_size_train, batch_size_val,
                    epochs, num_labels, tokenizer=tokenizer):
    """
    Function to train a DistilBert model on the data.
    train_comments : comments for training ; lst of str
    train_labels : labels for training ; lst of int
    val_comments : comments for validation ; lst of str
    val_labels : labels for validation ; lst of int
    batch_size_train : batch size for training ; int
    batch_size_val : batch size for validation ; int
    epochs : number of epochs ; int
    num_labels : number of labels (for denoiser 2, for classification 3) ; int
    tokenizer : tokenizer to use ; DistilBertTokenizer
    """

    # Instantiate multilingual model
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased', num_labels=num_labels)
    

    # Setup the Hugging Face Dataset Class
    train_dataset_dict = {"text": train_comments, "label": train_labels}
    val_dataset_dict = {"text": val_comments, "label": val_labels}

    train_dataset = Dataset.from_dict(train_dataset_dict)
    val_dataset = Dataset.from_dict(val_dataset_dict)

    # Apply the tokenizer to the datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    # Remove columns we do not need for training
    train_dataset = train_dataset.remove_columns(["text"])
    val_dataset = val_dataset.remove_columns(["text"])

    # Set the format of the datasets to PyTorch tensors
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")


    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=epochs,              # total number of training epochs
        per_device_train_batch_size=batch_size_train,  # batch size for training
        per_device_eval_batch_size=batch_size_val,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        evaluation_strategy="steps",     # Evaluate every `eval_steps`
        eval_steps=10,                   # Number of steps between evaluations
        save_steps=10,                   # Save the model every `save_steps`
        load_best_model_at_end=True,     # Load the best model at the end of training
    )

    # Trainer
    trainer = Trainer(
        model=model,                         # model
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,            # evaluation dataset
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()



    return model, tokenizer


def save_model(model, tokenizer, path):
    """
    Function to save the model
    model : model to save ; DistilBertForSequenceClassification
    tokenizer : tokenizer to save ; DistilBertTokenizer
    path : path to save the model ; str
    """

    model_save_path =  path
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

