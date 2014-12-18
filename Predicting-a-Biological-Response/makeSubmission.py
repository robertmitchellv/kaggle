"""
    //kaggle submission
    //Biological Response
    --> random forest classifier
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def main():
    # create the training + test sets
    try:
        dataset = pd.read_csv('Data/train.csv')
    except IOError:
        print("io ERROR-->Could not locate file.")

    target = dataset.Activity.values

    train = dataset.drop('Activity', axis = 1).values

    test = pd.read_csv('Data/test.csv').values

    # create and train the random forest and call it 'rf'
    # --> n_estimators = the number of trees in this forest, viz.
    #     100 trees of forest
    # --> n_jobs set to -1 will use the number of cores present on your system.
    rf = RandomForestClassifier(n_estimators = 100, n_jobs = -1)
    # fit(X, y[, sample_weight]) = build a forest of tress from the
    # training set (X, y)
    rf.fit(train, target)

    # predict_proba(X) predict class probabilities for X as list
    predicted_probs = [x[1] for x in rf.predict_proba(test)]

    # pandas series = a one dimentional ndarray with axis labels on the
    # previously predicted
    # class probabilities for the test
    predicted_probs = pd.Series(predicted_probs)

    # set index to start at 1 to use for MoleculeID when writing to file
    predicted_probs.index += 1
    print(type(predicted_probs))

    print(predicted_probs)
    """
    # write predicted_probs to file with pandas method .to_csv()--add header
    # for submission
    try:
        predicted_probs.to_csv('Data/submission.csv', header = True, \
                                index_label = ['MoleculeId', \
                                'PredictedProbability'], cols = 2)
        print("File successfully written; check 'Data' folder")
    except IOError:
        print("io ERROR-->Could not write data to file.")
    """
# call the main function
main()
