import csv
import sklearn.metrics
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import sklearn.metrics

# Retrieve data
def get_data():
    # Corpus
    x = []
    y = []

    # Open CSV file
    with open('processed.csv') as csvfile:
        csvreader = csv.reader(csvfile, delimiter = ",")
        # Goes through data in each row
        for row in csvreader:
            # Add message to list
            x.append(row[1])
            # Add classification to list spam = 1, ham = 0
            if row[0] == 'spam':
                y.append(1)
            else: y.append(0)
    # Seperate training data with 80% and test data with 20% of dataset
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.20)
    return X_train, X_test, y_train, y_test

# Train model
def train(X_train, y_train):
    # Used to vectorize words
    vectorizer = CountVectorizer()
    kn = KNeighborsClassifier(n_neighbors=1)

    # Used to build a composite estimator using vectorizer and SVM 
    model = Pipeline([('vectorizer', vectorizer), ('kn', kn)])
    model.fit(X_train,y_train)

    # Save model 
    pickle.dump(model, open('model.sav', 'wb'))
    return model

# Used to validate model accuracy
def validate_model(model, X_test, y_test):
    # Use X training data to predict using the model
    y_predict = model.predict(X_test)
    # Used to measure accuracy of model
    print(sklearn.metrics.classification_report(y_test, y_predict))
