import csv
import sklearn.metrics
import pickle
from sklearn.model_selection import train_test_split

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
#def train():
    # Used to vectorize words

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = get_data()