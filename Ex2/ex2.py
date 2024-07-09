import pandas as pd

def analyse_s2_stats_ex2():
    df = pd.read_csv('s2.csv')

    requireedCols = ['PWM_ref', 'MES_ref', 'NNSplice_ref', 'HSF_ref',
                           'GeneSplicer_ref', 'GENSCAN_ref', 'NetGene2_ref', 'SplicePredictor_ref']
    missing_report = df[requireedCols].isna().sum() / len(df)
    print("Missing values report (rate):")
    print(missing_report)


if __name__ == '__main__':
    analyse_s2_stats_ex2