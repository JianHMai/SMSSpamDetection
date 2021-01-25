from SMS import preprocess
from train import get_data
from train import train
from train import validate_model
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
                # Remove all non alphabetic letters 
                encoded_string = messages.encode("ascii", "ignore")
                messages = encoded_string.decode()
                # Call word tokenize for each row of messages
                text = preprocess(word_tokenize(messages))
            for classification in row[:1]:
                classify = classification
            # Write each classfication to new proceesed file    
            new_file.write(classify + "," + text + "\n")
    new_file.close()
f.close()

X_train, X_test, y_train, y_test = get_data()
model = train(X_train, y_train)
validate_model(model, X_test, y_test)