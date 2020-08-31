# UDACITY Nanodegree Become a Data Scientist 

## Project #5 ETL_Disaster_Response_Project
Building my first web app. Data provided by Figure Eight

### Content
1. Python Libraries
2. Overview
3. Files
4. ETL Skript
5. ML Skript
6. Putting it all together
7. Plot Example
8. Classification Example
9. Licensing, Authors, and Acknowledgements

### 1. Python Libraries
Mainly the Libraries were used which are included in a Standard Anaconda Installation. Additionally 
- sqlalchemy
- NLTK <br>

were used

### 2. Overview

Figure Eight is specialized in annotating and attributing all kind of data. In Case of a Disaster Machine Learning Algorithms can be beneficial by filtering relevant information. This project is about using such annotated data to build a machine learning pipeline in order to classify e.g. SMS send during a disaster. I therefore make use of the NLTK Library that helps to extract certain words. Specifically, I will use an annotated Figure Eight Data set (Messages send during disasters) to classify each message (30 Categories such as “Flood”) By doing so the messages can be filtered by relevance. (e.g. “Medical Help” seems to be more relevant than “Shops”) <br>
<br>
Pipelines were used one the hand to improve the model further in an efficient way (e.g. if there is more data available in the future) and on the other hand to prevent data leakage (clear distinction between training and tes data). Pipelines were used both for the ETL (e.g. Data cleaning) and the ML part. In the end only the best ML model is chosen and deployed. User can access the Model via a Web app. There they can explore the data used to build the model (various charts are provided). And they can as well test random messages to see how these are classified. 

### 3. Files

Input:<br>
The Data can be downloaded from the Udacity Website. Specifically there is a "disaster_categories.csv"-File and a "disaster_messages.csv"-File which need to be joined in the first place. The Data is provied by Factor Eight<br>
<br>

Output:<br>
- SQLite database (containing Cleaned Data)
- pickle file (Containing ML Model)

### 4. ETL Skript

Contains a pipeline that
- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

### 5. ML Skript

Contains a pipeline that 
- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

### 6. Putting it all together

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### 7. Plot Example

![alt text](https://github.com/pascal-bartl/ETL_Disaster_Response_Project/blob/master/1.png?raw=true)

### 8. Classification Example

![alt text](https://github.com/pascal-bartl/ETL_Disaster_Response_Project/blob/master/2.png?raw=true)

### 9. Licensing, Authors, and Acknowledgements

There are no references to other websites, books, and other resources. I did enjoy working on this project. Thanks UDACITY!
