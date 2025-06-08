## a stripped down version of the original code that doesn't involve dashboards
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from gliner_spacy.pipeline import GlinerSpacy
import warnings
import os
import sys
import gc
import logging

import requests
import json
import pandas as pd

## Load Google's content categories
def load_google_categories(url='https://www.google.com/basepages/producttype/taxonomy-with-ids.en-US.txt'):
    f = requests.get(url).text
    google_categories = f.split('\n')
    google_categories = [x.split(' - ') for x in google_categories]
    google_categories = {x[1].strip():int(x[0].strip()) for x in google_categories if len(x)==2}
    return google_categories



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", message="The sentencepiece tokenizer")

# Configuration for GLiNER integration
custom_spacy_config = {
    "gliner_model": "urchade/gliner_small-v2.1",
    "chunk_size": 128,
    "labels": ["person", "organization", "location", "event", "work_of_art", "product", "service", "date", "number", "price", "address", "phone_number", "misc"],
    "threshold": 0.5
}

# Model variables for lazy loading
nlp = None
sentence_model = None
google_categories = []

# Function to lazy load NLP model
def get_nlp():
    global nlp
    if nlp is None:
        try:
            logger.info("Loading spaCy model")
            nlp = spacy.blank("en")
            nlp.add_pipe("gliner_spacy", config=custom_spacy_config)
            logger.info("spaCy model loaded successfully")
        except Exception as e:
            logger.exception("Error loading spaCy model")
            raise
    return nlp

# Function to lazy load sentence transformer model
def get_sentence_model():
    global sentence_model
    if sentence_model is None:
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    return sentence_model

# Function to precompute category embeddings
def compute_category_embeddings():
    try:
        categories = list(load_google_categories().keys())
        return get_sentence_model().encode(categories)
    except Exception as e:
        return []

# Function to perform topic modeling using sentence transformers
def perform_topic_modeling_from_similarities(similarities):
    try:
        categories = list(load_google_categories().keys())
        top_indices = similarities.argsort()[-3:][::-1]
        
        best_match = categories[top_indices[0]]
        second_best = categories[top_indices[1]]
        
        if similarities[top_indices[0]] > similarities[top_indices[1]] * 1.1:
            return best_match
        else:
            return f"{best_match} , {second_best}"
    except Exception as e:
        return "Error in topic modeling"

## Optimized batch processing of keywords
def batch_process_keywords(keywords, batch_size=8):
    processed_data = {'Keywords': [], 'Google Content Topics': [], 'Google Content Topic IDs': []}
    
    try:
        sentence_model = get_sentence_model()
        category_embeddings = compute_category_embeddings()
        
        for i in range(0, len(keywords), batch_size):
            logger.info(f"Processing {len(keywords)} keywords")
            batch = keywords[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}")
            batch_embeddings = sentence_model.encode(batch, batch_size=batch_size, show_progress_bar=False)
                        
            similarities = cosine_similarity(batch_embeddings, category_embeddings)
            Google_Content_Topics = [perform_topic_modeling_from_similarities(sim) for sim in similarities]
            cats = [x.split(' , ') for x in Google_Content_Topics]
            cats = [[category_with_ids[x] for x in item] for item in cats]
            
            processed_data['Keywords'].extend(batch)
            processed_data['Google Content Topics'].extend(Google_Content_Topics)
            processed_data['Google Content Topic IDs'].extend(cats)
            
            # Force garbage collection
            gc.collect()
        logger.info("Keyword processing completed successfully")
    except Exception as e:
        logger.exception("An error occurred in batch_process_keywords")
    
    return processed_data


## Read file containing product titles and save output to file
if __name__ == "__main__":
    inputf = sys.argv[1]
    outputf = sys.argv[2]
    category_with_ids = load_google_categories()
    products = []
    with open(inputf,'r') as f:
        for line in f:
            products.append(line.strip())
    results = batch_process_keywords(products)
    pd.DataFrame(results).to_csv(outputf, index=False)

