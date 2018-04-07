import numpy

array = numpy.arange(6) # stworzenie tablicy z wartościami od 0 do 5
print(array) # [0 1 2 3 4 5]

reshaped_array = numpy.reshape(array, (2, 3)) # zmiana kształtu macierzy bez zmiany jej danych
print(reshaped_array) # [[0 1 2]
                      #  [3 4 5]]

dim_reshaed_array = numpy.ndim(reshaped_array) # liczba wymiarów macierzy
print(dim_reshaed_array) # 2

shape = reshaped_array.shape # kształt macierzy
print(shape) # (2, 3)

data_type = reshaped_array.dtype # typ danych w macierzy
print(data_type) # int64

new_array = reshaped_array + 3 # dodanie do każdego elementu macierzy liczby 3
print(new_array) # [[3 4 5]
                 #  [6 7 8]]

new_array = reshaped_array * 3 # przemnozenie wszystkich elementow macierzy przez 3
print(new_array) # [[ 0  3  6]
                 #  [ 9 12 15]]

new_array = reshaped_array ** 2 # potegowanie wszystkich elementow macierzy 
print(new_array) #  [[ 0  1  4]
                 # [ 9 16 25]]

condition_result = reshaped_array > 2 # sprawdzenie, ktore elementy macierzy sa wieksze od 2
print(condition_result) # [[False False False]
                        # [ True  True  True]]

array[array > 2] = 5 # podmiana elementow wiekszych od 2 na 5
print(array) # [0 1 2 5 5 5]

new_arr = [1, 2, 3]
result = numpy.exp(new_arr) # exponenta wsyztskich elementów
print(result) # [ 2.71828183  7.3890561  20.08553692]

new_arr = [1, 2, 3]
result = numpy.sqrt(new_arr) # pierwiastek wsyztskich elementów
print(result) # [1. 1.41421356 1.73205081]

first_array = [1, 2, 3]
second_array = [4, 5, 6]
result = numpy.dot(first_array, second_array) # iloczyn skalarny dwóch wektorów
print(result) # 32

first_array = [[1, 2], [3, 4]]
second_array = [[5, 6], [7, 8]]
result = numpy.add(first_array, second_array) # dodawanie macierzy
print(result) # [[ 6  8]
              #  [10 12]]
