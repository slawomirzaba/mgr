import pandas as pd

df = pd.read_csv("/home/szaba/Documents/studia/mgr/experiment_results/testy.csv", index_col=0)
algorithm_names = ['ConjugateGradient', 'QuasiNewton', 'GradientDescent']

for algorithm_name in algorithm_names:
    algorithm_name_filter = df["Nazwa algorytmu"] == algorithm_name
    mask = algorithm_name_filter
    df_filtered = df.loc[mask]
    fileName = '/home/szaba/Documents/studia/mgr/tables/{0}-all.csv'.format(algorithm_name)
    df_filtered.to_csv(fileName)