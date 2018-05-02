import pandas as pd

paths = [{
    'original': '/home/szaba/Documents/studia/mgr/tables/QuasiNewton-all.csv',
    'targetName': 'QuasiNewton',
    'to_remove': ['Lp.', 'iteracja', 'Nazwa algorytmu', 'Wspolczynnik uczenia', 'Liczba warstw ukrytych']
}, {
    'original': '/home/szaba/Documents/studia/mgr/tables/ConjugateGradient-all.csv',
    'targetName': 'ConjugateGradient',
    'to_remove': ['Lp.', 'iteracja', 'Nazwa algorytmu', 'Liczba warstw ukrytych']
}, {
    'original': '/home/szaba/Documents/studia/mgr/tables/GradientDescent-all.csv',
    'targetName': 'GradientDescent',
    'to_remove': ['Lp.', 'iteracja', 'Nazwa algorytmu', 'Liczba warstw ukrytych']
}]
hidden_layers = [1, 3, 5]

for data_read in paths:
    df = pd.read_csv(data_read['original'])
    for hidden_layer in hidden_layers:
        hidden_layers_filter = df["Liczba warstw ukrytych"] == hidden_layer
        mask = hidden_layers_filter
        df_filtered = df.loc[mask]
        df_filtered = df_filtered.sort_values(by=['iteracja', 'Lp.'])
        df_filtered = df_filtered.drop(data_read['to_remove'], axis=1)
        
        file_name = '/home/szaba/Documents/studia/mgr/tables/{0}/{0}-layers{1}.csv'.format(data_read['targetName'], hidden_layer)
        df_filtered.to_csv(file_name, index=False)
