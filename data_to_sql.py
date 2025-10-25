import sqlite3 as sq 
import numpy as np
import pandas as pd
import datetime as dt

# specifically for saving dmlstm array output
class CreateSQLiteDatabase():
    def __init__(self, tickers: list[str], name_database: str, name_data: str):
        # define initial class attributes
        self.tickers = tickers 
        self.name_db = name_database
        self.name = name_data

    def create_access_database_file(self):
        self.con = sq.connect(self.name_db) # will create database in this folder / or access it if already exists
        self.cur = self.con.cursor() # will allow us to exacute sql query's

    def create_data(self):
        '''Create new data frame if one exists with the same name
        '''
        data = pd.DataFrame(columns = self.tickers)
        data.to_sql(name = self.name, con = self.con, if_exists = 'replace', index = True) 

    def list_to_data(self, weights: list):
        if isinstance(weights, list): 
            data_pre_process = dict(zip(self.tickers, weights))
            self.weights = pd.DataFrame(data = data_pre_process, index = [dt.datetime.today().isoformat()])
        
        # update database
        self.weights.to_sql(name = self.name, con = self.con, if_exists = 'append', index = True)
        
    def query_database(self): 
        '''Intended to be used seperatley from class
        '''
        return pd.read_sql_query(f'SELECT * FROM {self.name}', con = self.con) # return all columns of the dataframe

