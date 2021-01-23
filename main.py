from SMS import preprocess
import re
import string
import io
from nltk.tokenize import word_tokenize
import csv

with open('dataset.csv') as f:
    reader = csv.reader(f, delimiter = ',')
    #Skip first row
    next(reader)
    with open('processed.csv','w',encoding='UTF-8') as new_file:
        for row in reader:
            for messages in row[1:-3]:
                re.sub('[^a-zA-Z0-9]+', '', messages)
                text = preprocess(word_tokenize(messages))
            for classification in row[:1]:
                classify = classification
            new_file.write(classify + "," + text + "\n")
    new_file.close()
f.close()
    
    #x_train, x_test, y_train, y_test = get_data()