"""
    //kaggle submission
    //Titanic: Machine Learning from Disaster
    --> Predict survival on the Titanic

    Author: Robertmitchellv
    Date: Dec 22, 2104
    Revised: Dec 22, 2014
"""

import pandas as pd
import numpy as np


def main():
    # comment
    try:
        train = pd.read_csv('Data/train.csv')
    except IOError:
        print("io ERROR-->Could not locate file.")
    
    # comment
    try:
        test = pd.read_csv('Data/test.csv')
    except IOError:
        print("io ERROR-->Could not locate file.")


# call main function
main()
