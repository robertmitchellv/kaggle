"""
    //kaggle submission
    //Biological Response
    --> cross validation
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
import numpy as np
import pandas as pd
import logloss

def main():
    #read data from csv; use nparray to create the training + target sets
    try:
        train = pd.read_csv('Data/train.csv')
    except IOError:
        print("io ERROR-->Could not locate file.")

    target = np.array([x[0] for x in train])
    train = np.array([x[1:] for x in train])

    # in this case we'll use a random forest, but this could be any classifier
    model = RandomForestClassifier(n_estimators = 100, n_jobs = -1)

    # simple K-Fold cross validation. 10 folds.
    cv = KFold(n = len(train), n_folds = 10, indices = False)

    #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list    
    results = []
    for traincv, testcv in cv:
        prob = model.fit(train[traincv], target[traincv]).predict_proba(train[testcv])
        results.append(logloss.llfun(target[testcv], [x[1] for x in prob]))

    #print out the mean of the cross-validated results
    print('Results: ', str(np.array(results).mean()))

# call main function
main()
