# Movie Recommendation System

This project implements three different recommendation algorithms for movie recommendations based on text data (descriptions). The three solutions include:

1. **Conventional Information Retrieval (IR) System**: A traditional method using TF-IDF scores, creating a linked list to store the postings list, and merging the posting lists for information retrieval.
   
2. **TF-IDF with Cosine Similarity**: A modern approach where we compute TF-IDF vectors for the corpus and the query, then use cosine similarity to find the top 5 most relevant documents.
   
3. **Sentence Transformers for Semantic Search**: The corpus is embedded using Sentence Transformers (binary encoders). The system compares the query embedding to the corpus embeddings and returns the top 5 most relevant results based on cosine similarity.

The working of this library requires updated huggingface libraries. Make sure to follow the video for running the jupyter notebook.

Video Link: 

## Dataset

The dataset used for the movie recommendation system is the netflix-movies_shows dataset from HuggingFace. You can easily import the dataset using Hugging Face Datasetslibrary for easy loading.

To load the dataset using Hugging Face, you can use the following code:

```python
from datasets import load_dataset

data = load_dataset("harshi321/netflix-movies_shows")

The working of this library requires updated huggingface libraries. Make sure to follow the video for running the jupyter notebook.
