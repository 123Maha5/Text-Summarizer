import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

def summarize_bert(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    sentences = sent_tokenize(text)

    freq = {}
    for word in words:
        word = word.lower()
        if word.isalpha() and word not in stop_words:
            freq[word] = freq.get(word, 0) + 1

    scores = {}
    for sentence in sentences:
        for word in freq:
            if word in sentence.lower():
                scores[sentence] = scores.get(sentence, 0) + freq[word]

    avg = sum(scores.values()) / len(scores) if scores else 0

    summary = ' '.join([s for s in sentences if scores.get(s, 0) > 1.2 * avg])
    return summary


