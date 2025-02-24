# Movie Recommendation System

This project implements three different recommendation algorithms for movie recommendations based on text data (descriptions). The three solutions include:

1. **Conventional Information Retrieval (IR) System**: A traditional method using TF-IDF scores, creating a linked list to store the postings list, and merging the posting lists for information retrieval.
   
2. **TF-IDF with Cosine Similarity**: A modern approach where we compute TF-IDF vectors for the corpus and the query, then use cosine similarity to find the top 5 most relevant documents.
   
3. **Sentence Transformers for Semantic Search**: The corpus is embedded using Sentence Transformers (binary encoders). The system compares the query embedding to the corpus embeddings and returns the top 5 most relevant results based on cosine similarity.

The working of this library requires updated huggingface libraries. Make sure to follow the video for running the jupyter notebook.

Video Link: https://drive.google.com/file/d/1EPTimmj1g2Ffacr_66TCis_GL1eczdy8/view?usp=sharing

## Dataset

The dataset used for the movie recommendation system is the netflix-movies_shows dataset from HuggingFace. You can easily import the dataset using Hugging Face Datasetslibrary for easy loading.

To load the dataset using Hugging Face, you can use the following code:

```python
from datasets import load_dataset

data = load_dataset("harshi321/netflix-movies_shows")

```

## Conventional Information Retrieval (IR) System

Make sure to have these .py files in the same directory.

---> preprocessor.py
---> indexer.py
---> linkedlist.py

```python

runner = Run_query()
runner.run_indexer(data['description'])


queries = ['romance'] 

```
**Sample Output**

{'romance': {'results': [798, 402, 492, 1038, 1269], 'num_docs': 42, 'num_comparisons': 0}}
Query: romance
Retrieved Documents with skip pointers:
Love Jones
The Last Letter From Your Lover
Midnight Sun
Dancing Angels
Geez & Ann


 ## TF-IDF with Cosine Similarity

 No other dependencies required, just run the class and run this code.

 ```python

 titles = data['title'] 
descriptions = data['description']

query_processor = QueryProcessor(titles=titles, descriptions=descriptions)

query = "I need thriller movies"
query_processor.process_query(query, top_n=5)

```
**Sample Output**

Top 5 Results for the Query 'I need thriller movies':
Document 201: Title: Krishna Cottage | Description: True love is put to the test when another woman comes between a pair of star-crossed young lovers in this thriller. (Cosine Similarity: 0.2157)
Document 766: Title: Xtreme | Description: In this fast-paced and action-packed thriller, a retired hitman — along with his sister and a troubled teen — takes revenge on his lethal stepbrother. (Cosine Similarity: 0.1880)
Document 1679: Title: Raman Raghav 2.0 | Description: A corrupt cop and a serial killer obsessed with a psychopath from the '60s get caught up in a ruthless cat-and-mouse game in this Indian thriller. (Cosine Similarity: 0.1865)
Document 802: Title: Never Back Down 2: The Beatdown | Description: A group of mixed martial arts fighters stars in this action thriller that follows a quartet of brawlers as they prepare for a major underground event. (Cosine Similarity: 0.1844)
Document 902: Title: Deadly Switch | Description: In this indie thriller, a foreign exchange student moves in with her roommate's family who grieves over the daughter they would do anything to get back. (Cosine Similarity: 0.1818)


## Sentence Transformers for Semantic Search

 No other dependencies required, just run the class and run this code.

 ```python

titles = data['title'][:100] # List of titles
descriptions = data['description'][:100]  # List of descriptions
binary_model = SentenceTransformer('all-MiniLM-L6-v2')  # Replace with your specific model

searcher = EncoderSearch(titles=titles, descriptions=descriptions, model=binary_model)

query = "I need crime movies"
searcher.search(query, top_n=5)

```

**Sample Output**


Top Results:
1. Title: The Women and the Murderer | Description: This documentary traces the capture of serial killer Guy Georges through the tireless work of two women: a police chief and a victim's mother. (Score: 0.4125)
2. Title: Crime Stories: India Detectives | Description: Cameras following Bengaluru police on the job offer a rare glimpse into the complex and challenging inner workings of four major crime investigations. (Score: 0.3945)
3. Title: Omo Ghetto: the Saga | Description: Twins are reunited as a good-hearted female gangster and her uptight rich sister take on family, crime, cops and all of the trouble that follows them. (Score: 0.3187)
4. Title: Show Dogs | Description: A rough and tough police dog must go undercover with an FBI agent as a prim and proper pet at a dog show to save a baby panda from an illegal sale. (Score: 0.3049)
5. Title: Vendetta: Truth, Lies and The Mafia | Description: Sicily boasts a bold "Anti-Mafia" coalition. But what happens when those trying to bring down organized crime are accused of being criminals themselves? (Score: 0.3033)









