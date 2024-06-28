How to reproduce our results (using Google Colab):

1. For the inference results :
- Open the Inference.ipynb
- Select your LANGUAGE 
- Load the .csv file for inference, found at Comments DB/LANGUAGE/Inference/results 
- Load the respective model of the LANGUAGE, that is sentiment_model_finetuned_LANGUAGE
 (Note that we loaded it via google Drive, because the models can exceed 1 GB ;
  therefore, you would have to upload to your google drive account and then mount your account in Google Colab using the code,
  then you can access the model directly)
- Run the correct cell, depending on the language, which will then automatically download your inferenced file 


2. The general metholodgy :
- To use the scraper, one needs to generate a YouTube API Key here (https://console.cloud.google.com/apis/api/youtube.googleapis.com/metrics?project=fair-app-421615)
- Then, one needs to put videos in a public playlist on YouTube, extract the ID of that Playlist, add it into the code (see the code for details)
- Select which language the comments are in 
- Then you can scrape the comments 

- This process is already done and we provide the files in Comments DB/english/Scraped/processed for the english comments and
  in Comments DB/LANGUAGE/Inference/processed for the other languages 

- Next, you would have to do secondary cleaning on the english comments using SecondaryCleaningEnglishComments.ipynb 
- Then, you label these comments using EnglishModelForLabelling.ipynb
- Next, you augment the labelled data in Augmentation.ipynb
- And translate it using Translator.ipynb



Note that especially the translator took us multiple hours for each language. The translated versions can be found in 
Comments DB/LANGUAGE/TranslatedFromEnglish

- Using these translated files, you first have to process them again (since we translated on the original comments), using 
  ProcessingTranslatedComments.ipynb ; also, we need to process the scraped comments for inference. See the .ipynb file for more details.

- Finally, run the fine-tuning.ipynb's for the respective languages. We prepared the files for that in 
  Comments DB/LANGUAGE/Finetuning

Note that we bought Google Colab Pro for this, since we ran into memory issues using the free version and our Laptops couldn't handle it.
Thus, we utilized GPU L4 of Google Colab Pro for this process.

- The models will be saved in your drive account, where you then can access them for the inference part. 





