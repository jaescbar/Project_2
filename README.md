# Disaster Response Pipeline Project 2

## 1. Proyect Motivation

  Project_2 is the second project of the data cience Udacity Course. The idea is to create a disaster response pipeline that helps categorize different real life messages.
  

## 2. File Description

The repository has 3 folders and this README file. The data folder contains csv files,  process_data.py and a Jupyter Notebook that contains the same code that process_data.py. The models folder contains train_classifier.py, which builds and trains the model that will categorize messages. Finally the app folder contains run.py which is used to deploy the flask app 

## 3. Instructions

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## 4. Author and Acknowledgements

  The author of this project is Jorge Andrés Escobar, Universidad de los Andes economist with a masters degree in economics, currently working in Banco Davivienda.
  Special acknowledgments to Gustavo Venegas, Gustavo Torres and Udacity for the help they gave in this project.
