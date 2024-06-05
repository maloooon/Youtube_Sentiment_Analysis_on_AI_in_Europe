import pandas as pd
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Carica il modello e il tokenizer
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# Funzione per tradurre il testo
def translate_text(text, source_lang, target_lang):
    tokenizer.src_lang = source_lang
    encoded_text = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded_text,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang]
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# Leggi il CSV
input_csv_path = '/Users/giopa/Onedrive/Desktop/Uni II/Youtube/6_NEGATIVE_ENGLISH_cleaned_and_filtered_comments_for_labeling_LABEL_HERE.csv'    # Percorso del file CSV di input
output_csv_path = "/Users/giopa/Onedrive/Desktop/Uni II/Youtube/translated_comments.csv"  # Percorso del file CSV di output
comments_df = pd.read_csv(input_csv_path)

# Colonna dei commenti e lingua di origine
comments_column = "Comment"  # Nome della colonna dei commenti
source_lang = "en_XX"  # Codice della lingua di origine
target_lang = "fr_XX"  # Codice della lingua di destinazione

# Traduci i commenti
comments_df["translated_comment"] = comments_df[comments_column].apply(lambda x: translate_text(x, source_lang, target_lang))

# Scrivi il risultato in un nuovo CSV
comments_df.to_csv(output_csv_path, index=False)