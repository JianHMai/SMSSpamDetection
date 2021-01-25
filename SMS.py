from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string

# Remove stopwords and punctuation
def preprocess(line):
    # Used to stem the words
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    words = ""
    for word in line:
        # If word is not a stop word, stem it and add to string
        if word not in stop_words:
            words += ps.stem(word) + " "
    # Remove extra spaces at the end of string        
    return " ".join(words.split())