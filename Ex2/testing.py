import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('s2.csv')
    size = len(df)
    training = df[:size // 2]
    testing = df[size // 2:]
    dataframes = [df, training, testing]
    for dataframe in dataframes:
        print(dataframe.name)
