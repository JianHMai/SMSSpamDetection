import csv
import string
import io
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#Remove stopwords and punctuation
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

with open('dataset.csv') as f:
    reader = csv.reader(f, delimiter = ',')
    #Skip first row
    next(reader)
    with open('processed.csv','w',encoding='utf-8') as new_file:
        for row in reader:
            for words in row[1:-3]:
                text = preprocess(word_tokenize(words))
            for classification in row[:1]:
                classify = classification
            new_file.write(classify + "," + text + "\n")
    new_file.close()
f.close()