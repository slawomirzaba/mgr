import pandas
csv_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
data_frame = pandas.read_csv(csv_url) # wczytanie pliku csv

instances_number = data_frame.shape[0] # pobranie liczby wszystkich instancji

columns_number = data_frame.shape[1] # pobranie liczby kolumn

data_frame.select_dtypes(['object']) # pobranie wszystkich wierszy, w ktorych co najmniej jedna wartosc jest typu object

data_frame.apply(pandas.to_numeric) # przekonwertowanie wszystkich wartosci w zbiorze na liczby

data_frame['nazwa_kolumny'].unique() # pobranie unikalnych wartosci spod wskazanej kolumny

data_frame.iloc[0:instances_number, [1, 2, 3]].values # pobranie wybranych kolumn (1, 2, 3) sposrod wszystkich instacji