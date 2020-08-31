# import packages
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Loads and merges the two datasets
    Args:
    messages_filepath: String. Filepath of the csv-file that contains the messages.
    categories_filepath: String. Filepath of the csv-file that contains the categories.
    Returns:
    df: pd df. df that contains messages and categories.
    '''
    # read in file
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    print('Messages df')
    messages.head()
    print('Categories df')
    categories.head()

    df = messages.merge(categories, left_on='id', right_on='id')

    return df

    ################################################
    # clean data
def clean_data(df):

    '''
    Clean df from unneeded columns, redundant data and text fragments
    Args:
    df: pd df. df that contains messages and categories.
    Returns:
    df: pd df. df that contains cleaned messages and categories.
    '''

    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand=True)

    # select the first row of the categories dataframe
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing

    category_colnames = []
    category_colnames2 = []

    categories_one = categories.iloc[ 0 , : ]
    category_colnames = categories_one.values.tolist()

    for i in category_colnames:
        category_colnames2.append(i[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames2

    #Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    # concatenate the original dataframe with the new `categories` dataframe
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(subset=['id'], keep='first', inplace=True)

    return df

def save_data(df, db_filename):
    """
    Save the cleaned data.

    Input:
    df: pd df. df that caontains cleaned messages and categories.
    db_filename: String. Filename of the output database.

    Output:
    Nothing.
    """
    # load to database
    engine = create_engine('sqlite:///'+ db_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, db_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(db_filepath))
        save_data(df, db_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'df as the first and second argument, as '\
              'well as the filepath of the db to store the cleaned data '\
              'as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()