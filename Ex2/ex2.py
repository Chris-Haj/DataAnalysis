import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, roc_curve
import numpy as np


def getMissingReport(requiredCols, df):
    numMissing = (df[requiredCols] == 0).sum()
    missingRate = (numMissing / len(df)).round(3)

    missingReport = pd.DataFrame({
        'Tools': requiredCols,
        'No of missing': numMissing.values,
        'Missing rate': missingRate.values
    })
    print(missingReport, "\n\n")


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


def confusionMatrixReport(actuals, predictions, groupName):
    cm = confusion_matrix(actuals, predictions, labels=['Positive', 'Negative'])
    print(f"\nConfusion Matrix for {groupName} group:\n{cm}")


def findOptimalThreshold(df, column, targetColumn):
    thresholds = np.linspace(0, 2, 200)
    bestThreshold = 0
    bestTPR = 0

    for threshold in thresholds:
        predictions = np.where(df[column] > threshold, 'Positive', 'Negative')
        cm = confusion_matrix(df[targetColumn], predictions, labels=['Positive', 'Negative'])
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)

        if fpr <= 0.1 and tpr > bestTPR:
            bestTPR = tpr
            bestThreshold = threshold

    return bestThreshold


def analyse_s2_stats_ex2():
    df = pd.read_csv('s2.csv')

    requiredCols = ['PWM_ref', 'MES_ref', 'NNSplice_ref', 'HSF_ref',
                    'GeneSplicer_ref', 'GENSCAN_ref', 'NetGene2_ref', 'SplicePredictor_ref']

    getMissingReport(requiredCols, df)

    requiredData = (df[requiredCols] == 0).sum(axis=1) / len(requiredCols) < 0.05
    nonZeroIndexes = df[requiredData].index.tolist()

    # Save the indices to a text file
    with open('NonZeroIndexes.txt', 'w') as file:
        for index in nonZeroIndexes:
            file.write(f"{index}\n")

    print("Indices of non-missing samples saved to NonZeroIndexes.txt\n")

    nonZeroValues = df.loc[nonZeroIndexes]

    trainSize = int(len(nonZeroValues) * 0.9)
    trainDf = nonZeroValues[:trainSize].copy()
    testDf = nonZeroValues[trainSize:].copy()
    assert len(trainDf) + len(testDf) == len(
        nonZeroValues), "The training set and the test set do not sum up to the size of the original data frame"
    reportDataTypes(nonZeroValues, 'Original')
    reportDataTypes(trainDf, 'Training')
    reportDataTypes(testDf, 'Test')

    # Stage 5:
    trainDfEqualRatios, testDfEqualRatios = train_test_split(nonZeroValues, test_size=0.1,
                                                             stratify=nonZeroValues['Group'])
    assert len(trainDfEqualRatios) + len(testDfEqualRatios) == len(
        nonZeroValues), "The training set and the test set do not sum up to the size of the original data frame"
    reportDataTypes(nonZeroValues, 'Original')
    reportDataTypes(trainDfEqualRatios, 'TrainingEqualRatios')
    reportDataTypes(testDfEqualRatios, 'TestEqualRatios')

    ratioNames = ['PWM', 'MES', 'NNSplice', 'HSF']

    def createMutationLabelsConfusionMatrix(name, labelRef, labelAlt):
        dataFrames = [trainDf, testDf, trainDfEqualRatios, testDfEqualRatios]
        dataFrameNames = ['trainDf', 'testDf', 'trainDfEqualRatios', 'testDfEqualRatios']

        for df, dfName in zip(dataFrames, dataFrameNames):
            df[f'ratio_{name}'] = df[labelRef] / df[labelAlt]
            df[f'Mutation_{name}'] = np.where(df[f'ratio_{name}'] > 1, 'Positive', 'Negative')
            confusionMatrixReport(df['Group'], df[f'Mutation_{name}'], f'{name} {dfName}')

    for name in ratioNames:
        createMutationLabelsConfusionMatrix(name, f'{name}_ref', f'{name}_alt')

    # Stage 6:
    pass
    confusionMatrixReport(trainDf['Group'], trainDf['Mutation'], 'Training')
    confusionMatrixReport(testDf['Group'], testDf['Mutation'], 'Test')
    confusionMatrixReport(trainDfEqualRatios['Group'], trainDfEqualRatios['Mutation'], 'TrainingEqualRatios')
    confusionMatrixReport(testDfEqualRatios['Group'], testDfEqualRatios['Mutation'], 'TestEqualRatios')

    # Stage 7: 10-fold cross-validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cm_train_total = np.zeros((2, 2))
    cm_test_total = np.zeros((2, 2))

    for train_index, test_index in skf.split(nonZeroValues, nonZeroValues['Group']):
        fold_train = nonZeroValues.iloc[train_index].copy()
        fold_test = nonZeroValues.iloc[test_index].copy()

        fold_train['ratio_PWM'] = fold_train['PWM_ref'] / fold_train['PWM_alt']
        fold_test['ratio_PWM'] = fold_test['PWM_ref'] / fold_test['PWM_alt']

        fold_train['Mutation'] = np.where(fold_train['ratio_PWM'] > 1, 'Positive', 'Negative')
        fold_test['Mutation'] = np.where(fold_test['ratio_PWM'] > 1, 'Positive', 'Negative')

        y_train_true = fold_train['Group']
        y_train_pred = fold_train['Mutation']
        y_test_true = fold_test['Group']
        y_test_pred = fold_test['Mutation']

        cm_train = confusion_matrix(y_train_true, y_train_pred, labels=['Positive', 'Negative'])
        cm_test = confusion_matrix(y_test_true, y_test_pred, labels=['Positive', 'Negative'])

        cm_train_total += cm_train
        cm_test_total += cm_test

    avg_cm_train = cm_train_total / 10
    avg_cm_test = cm_test_total / 10

    print(f"\nAverage Confusion Matrix for Training (10-fold CV):\n{avg_cm_train}")
    print(f"\nAverage Confusion Matrix for Test (10-fold CV):\n{avg_cm_test}")


if __name__ == '__main__':
    analyse_s2_stats_ex2()
