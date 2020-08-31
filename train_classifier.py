# import packages
from workspace_utils import active_session
import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report


from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.datasets import make_multilabel_classification
from sklearn.svm import SVC

from sklearn.multioutput import MultiOutputClassifier

from sqlalchemy import create_engine
import pickle
from sqlalchemy import create_engine

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

def load_data(db_filepath):
    """
    Load the data

    Inputs:
    db_filepath: String. Filepath of db file that contains cleaned data.

    Output:
    X: pd df. Contains the feature data.
    y: pd df. Contains the catagories data.
    category_names: List of strings. Contains the category names.
    """

    #LOAD FROM DATABASE
    engine = create_engine('sqlite:///{}'.format(db_filepath))
    df = pd.read_sql_table('messages', engine, index_col=None)

    #Replace '2'-Values by '1' (according tho the documentation of factor8)
    df.related.replace(2,1,inplace=True)

    #Define feature and target variables X and Y
    X = df['message'].values
    Y = df.drop(['id','message','original','genre'], axis = 1)
    category_names = Y.columns.tolist()

    return X, Y, category_names


def tokenize2(text):
    """
    Normalize, tokenize and stems messages.

    Input:
    text: string. text of the messages.

    Output:
    tok_list: list of strings. A list of strings that contain normalized and stemmed tokens.
    """

    # tokenize text
    tokens = word_tokenize(text)

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    tok_list = []
    for tok in tokens:

        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        tok_list.append(clean_tok)

    return tok_list



def build_model():
    """
    Builds a ML pipeline and uses a gridsearch.
    Args:
    Nothing
    Returns:
    model: gridsearchcv object.
    """

    # Create the pipeline
    pipeline1 = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize2)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(AdaBoostClassifier(random_state=42)))
    ])

    # Parameters
    parameters = {'tfidf__use_idf': (True, False),
            'clf__estimator__n_estimators': [50, 100], 
            'clf__estimator__random_state': [42],
            'clf__estimator__learning_rate': [0.5]}

    # Do a gridsearch
    model = GridSearchCV(pipeline1, param_grid = parameters, cv = 10,
                          refit = True, verbose = 1, return_train_score = True, n_jobs = -1)

    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Returns test accuracy, number of 1s and 0s, recall, precision and F1 Score.

    Inputs:
    model: model object. Instanciated model.
    X_test: pd df that contains test features.
    Y_test: pd df that contains test categories.
    category_names: list of strings containing category names.

    Returns:
    Nothing
    """
    # Predict the categories 
    Y_pred = model.predict(X_test)

    # Display results
    print(classification_report(Y_test, Y_pred, target_names = category_names))


def save_model(model, model_filepath):

    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


def main():
    with active_session():
        if len(sys.argv) == 3:
            database_filepath, model_filepath = sys.argv[1:]
            print('Loading data...\n    DATABASE: {}'.format(database_filepath))
            X, Y, category_names = load_data(database_filepath)

            # stratify multilabel data (Train/Test split)
            mlss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.33, random_state=42)

            for train_index, test_index in mlss.split(X, Y):
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y.values[train_index], Y.values[test_index]
            Y_train = pd.DataFrame(Y_train,columns=category_names)
            Y_test = pd.DataFrame(Y_test,columns=category_names)

            print('Building model...')
            model = build_model()

            print('Training model...')
            model.fit(X_train, Y_train)

            print('Evaluating model...')
            evaluate_model(model, X_test, Y_test, category_names)

            print('Saving model...\n    MODEL: {}'.format(model_filepath))
            save_model(model, model_filepath)

            print('Trained model saved!')

        else:
            print('Please provide the filepath of the disaster messages database '\
                  'as the first argument and the filepath of the pickle file to '\
                  'save the model to as the second argument. \n\nExample: python '\
                  'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
