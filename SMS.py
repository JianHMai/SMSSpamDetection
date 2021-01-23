from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string

# Remove stopwords and punctuation
def preprocess(line):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    words = ""
    for word in line:
        if word not in stop_words:
            words += ps.stem(word) + " "
    for c in string.punctuation:
        words = words.replace(c, "")
    return " ".join(words.split())