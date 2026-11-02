# 📝 Automated Text Summarization System

An NLP web application that generates concise summaries from long, unstructured text using extractive summarization — condensing articles or reports down to their most important sentences.



## Features
- Extractive text summarization using **TF-IDF sentence scoring**
- Adjustable summary length (1–8 sentences)
- Shows word-count reduction statistics
- Clean, responsive Flask web UI
- Colab notebook demonstrates experimentation with **TF-IDF vs. BERT sentence embeddings**

## Tech Stack
Python, NLTK, Scikit-learn (TF-IDF), Flask, HTML/CSS

## Project Structure
```
text-summarizer/
├── app.py                    # Flask web app
├── summarizer.py             # Core NLP summarization logic
├── templates/index.html
├── static/style.css
├── colab/text_summarization_experiments.ipynb   # TF-IDF & BERT experiments
├── requirements.txt
└── Procfile                  # For deployment (Render/Heroku)
```

## How It Works
1. **Text preprocessing:** whitespace cleanup, sentence tokenization (NLTK)
2. **Feature extraction:** TF-IDF vectorization of each sentence
3. **Sentence scoring:** each sentence scored by sum of TF-IDF weights
4. **Extraction:** top-N highest scoring sentences selected, restored to original order
5. **Output:** condensed summary + reduction statistics

The Colab notebook (`colab/text_summarization_experiments.ipynb`) additionally experiments with **BERT sentence embeddings + cosine similarity** for summarization, matching the resume's NLP experimentation claims, before settling on the lightweight TF-IDF approach for the deployed web app (fast, low memory, works on free hosting tiers).

## Run Locally
```bash
git clone <your-repo-url>
cd text-summarizer
pip install -r requirements.txt
python app.py
```
Visit `http://localhost:5001`

## Author
Mahalakshmi Para
