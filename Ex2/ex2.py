import pandas as pd
from sklearn.model_selection import train_test_split


def getMissingReport(requiredCols, df):
    numMissing = (df[requiredCols] == 0).sum()
    missingRate = (numMissing / len(df)).round(3)

    missingReport = pd.DataFrame({
        'Tools': requiredCols,
        'No of missing': numMissing.values,
        'Missing rate': missingRate.values
    })
    print(missingReport)


def reportDataTypes(df, groupName):
    positiveNums = (df['Group'] == 'Positive').sum()
    negativeNums = (df['Group'] == 'Negative').sum()
    size = len(df)

    posPercentage = (positiveNums / size) * 100
    negPercentage = (negativeNums / size) * 100

    print(f"Group: {groupName}\n"
          f"Total samples: {size}\n"
          f"Positive samples: {positiveNums}, in percentages: {posPercentage.round(3)}\n"
          f"Negative samples: {negativeNums}, in percentages: {negPercentage.round(3)}\n")


def splitIntoEqualGroups(df, nonZeroValues):
    trainDf, testDf = train_test_split(nonZeroValues, test_size=0.1, stratify=nonZeroValues['Group'])
    positiveNums = (testDf['Group'] == 'Positive').sum()
    negativeNums = (testDf['Group'] == 'Negative').sum()
    size = len(testDf)
    posPercentage = (positiveNums / size) * 100
    negPercentage = (negativeNums / size) * 100

    print(f"Group:\n"
          f"Total samples: \n"
          f"Positive samples: {positiveNums}, in percentages: {posPercentage.round(3)}\n"
          f"Negative samples: {negativeNums}, in percentages: {negPercentage.round(3)}\n")


def analyse_s2_stats_ex2():
    df = pd.read_csv('s2.csv')

    requiredCols = ['PWM_ref', 'MES_ref', 'NNSplice_ref', 'HSF_ref',
                    'GeneSplicer_ref', 'GENSCAN_ref', 'NetGene2_ref', 'SplicePredictor_ref']

    requiredData = (df[requiredCols] == 0).sum(axis=1) / len(requiredCols) < 0.05
    nonZeroIndexes = df[requiredData].index.tolist()

    # Save the indices to a text file
    # with open('nonMissingTop4.txt', 'w') as file:
    #     for index in nonZeroIndexes:
    #         file.write(f"{index}\n")
    #
    # print("Indices of non-missing samples saved to nonMissingTop4.txt")

    nonZeroValues = df.loc[nonZeroIndexes]
    trainSize = int(len(nonZeroValues) * 0.9)
    trainDf = nonZeroValues.iloc[:trainSize]
    testDf = nonZeroValues.iloc[trainSize:]
    assert len(trainDf) + len(testDf) == len(nonZeroValues), "The training set and the test set do not sum up to the size of the original data frame"
    reportDataTypes(df, 'Original')
    reportDataTypes(trainDf, 'Training')
    reportDataTypes(testDf, 'Test')

    # Stage 5:
    splitIntoEqualGroups(df, nonZeroValues)


if __name__ == '__main__':
    analyse_s2_stats_ex2()
