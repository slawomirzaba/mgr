import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # to remove tensorflow backend text on beginning
stderr = sys.stderr # to remove tensorflow backend text on beginning
sys.stderr = open(os.devnull, 'w') # to remove tensorflow backend text on beginning
import pandas
import numpy as np
from sklearn import preprocessing
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

sys.stderr = stderr # to remove tensorflow backend text on beginning

class DataProvider:
    def __init__(self, csv_url, features_column, class_column, test_size=0.3, random_state = None):
        encoder = preprocessing.LabelEncoder()
        self.data_frame = pandas.read_csv(csv_url, header=None)
        self.__remove_lack_values()
        self.__parse_feature_values_to_number(features_column)
        instances_number = self.data_frame.shape[0]
        x = self.data_frame.iloc[0:instances_number, features_column].values
        y = self.data_frame.iloc[0:instances_number, class_column].values
        encoded_y = encoder.fit_transform(y)
        hot_encoded_y = np_utils.to_categorical(encoded_y)
        normalized_x = np_utils.normalize(x)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(normalized_x, hot_encoded_y, test_size=test_size, random_state=random_state)

    def __remove_lack_values(self):
        data_frame_to_create_mask = self.data_frame.select_dtypes(['object'])
        mask = (data_frame_to_create_mask.apply(lambda x: x.str.strip()) == '?').any(axis=1)
        self.data_frame = self.data_frame[~mask]

    def __parse_feature_values_to_number(self, features_column):
        self.data_frame[features_column] = self.data_frame[features_column].apply(pandas.to_numeric)
    
    def get_training_data(self):
        return self.x_train, self.y_train

    def get_test_data(self):
        return self.x_test, self.y_test

    def get_whole_data(self):
        return self.x_train, self.y_train, self.x_test, self.y_test
