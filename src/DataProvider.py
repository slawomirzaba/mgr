import pandas as pandas
from sklearn import preprocessing
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

class DataProvider:
    def __init__(self, csv_url, column_of_features, column_of_class):
        encoder = preprocessing.LabelEncoder()
        data_frame = pandas.read_csv(csv_url, header=None)
        instances_number = data_frame.shape[0]
        x = data_frame.iloc[0:instances_number, column_of_features].values
        y = data_frame.iloc[0:instances_number, column_of_class].values
        encoded_y = encoder.fit_transform(y)
        hot_encoded_y = np_utils.to_categorical(encoded_y)
        normalized_x = np_utils.normalize(x)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(normalized_x, hot_encoded_y, test_size=0.2, random_state=42)

    def get_training_data(self):
        return self.x_train, self.y_train

    def get_test_data(self):
        return self.x_test, self.y_test

    def get_whole_data(self):
        return self.x_train, self.y_train, self.x_test, self.y_test
