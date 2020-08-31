import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine, index_col=None)
#Replace '2'-Values by '1' (according tho the documentation of factor8)
df.related.replace(2,1,inplace=True)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    #quantity of categories
    categories=df.iloc[:,4:].sum().sort_values(ascending=False).reset_index()
    categories.columns=['category','quantity']
    categories_quantity=categories['quantity'].values.tolist()
    categories_values=categories['category'].values.tolist()

    # select top 5 categories
    category_quantity = df.iloc[:,4:].sum(axis = 0).sort_values(ascending = False)
    category_top_5 = category_quantity.head(5)
    category_top_5_values = list(category_top_5.index)

    #top words
    word_srs = pd.Series(' '.join(df['message']).lower().split())
    words_top_5 = word_srs[~word_srs.isin(stopwords.words("english"))].value_counts()[:5]
    word_top_5_values = list(words_top_5.index)

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=categories_values,
                    y=categories_quantity,
                )
            ],

            'layout': {
                'title': "Messages categories Quantity",
                'yaxis': {
                    'title':"Message Category Quantity"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_top_5_values,
                    y=category_top_5
                )
            ],

            'layout': {
                'title': 'Top 5 Message Categories',
                'yaxis': {
                    'title': "Quantity"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=word_top_5_values,
                    y=words_top_5
                )
            ],

            'layout': {
                'title': 'Top 5 Words',
                'yaxis': {
                    'title': "Quantity"
                },
                'xaxis': {
                    'title': "Words"
                }
            }
        }





    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
