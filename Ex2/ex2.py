import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

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

    with open('NonZeroIndexes.txt', 'w') as file:
        for index in nonZeroIndexes:
            file.write(f"{index}\n")

    print("Indices of non-missing samples saved to NonZeroIndexes.txt\n")

    nonZeroValues = df.loc[nonZeroIndexes]

    trainSize = int(len(nonZeroValues) * 0.9)
    trainDf = nonZeroValues[:trainSize].copy()
    testDf = nonZeroValues[trainSize:].copy()
    assert len(trainDf) + len(testDf) == len(nonZeroValues), "The training set and the test set do not sum up to the size of the original data frame"
    reportDataTypes(nonZeroValues, 'Original')
    reportDataTypes(trainDf, 'Training')
    reportDataTypes(testDf, 'Test')

    trainDfEqualRatios, testDfEqualRatios = train_test_split(nonZeroValues, test_size=0.1, stratify=nonZeroValues['Group'])
    assert len(trainDfEqualRatios) + len(testDfEqualRatios) == len(nonZeroValues), "The training set and the test set do not sum up to the size of the original data frame"
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

    def stratify(name, labelRef, labelAlt):
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cm_train_total = np.zeros((2, 2))
        cm_test_total = np.zeros((2, 2))
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        plt.figure(figsize=(10, 6))

        for i, (train_index, test_index) in enumerate(skf.split(nonZeroValues, nonZeroValues['Group'])):
            fold_train = nonZeroValues.iloc[train_index].copy()
            fold_test = nonZeroValues.iloc[test_index].copy()

            fold_train[f'ratio_{name}'] = fold_train[labelRef] / fold_train[labelAlt]
            fold_test[f'ratio_{name}'] = fold_test[labelRef] / fold_test[labelAlt]

            fold_train[f'Mutation_{name}'] = np.where(fold_train[f'ratio_{name}'] > 1, 'Positive', 'Negative')
            fold_test[f'Mutation_{name}'] = np.where(fold_test[f'ratio_{name}'] > 1, 'Positive', 'Negative')

            confusionMatrixReport(fold_train['Group'], fold_train[f'Mutation_{name}'], f'{name} train')
            confusionMatrixReport(fold_test['Group'], fold_test[f'Mutation_{name}'], f'{name} test')

            cm_train = confusion_matrix(fold_train['Group'], fold_train[f'Mutation_{name}'], labels=['Positive', 'Negative'])
            cm_test = confusion_matrix(fold_test['Group'], fold_test[f'Mutation_{name}'], labels=['Positive', 'Negative'])

            cm_train_total += cm_train
            cm_test_total += cm_test

            y_test = fold_test['Group'].apply(lambda x: 1 if x == 'Positive' else 0)
            y_score = fold_test[f'ratio_{name}']

            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {i} AUC: {roc_auc:.3f}')

            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0

        avg_cm_train = cm_train_total / 10
        avg_cm_test = cm_test_total / 10
        print(f"\nAverage Confusion Matrix for Training (10-fold CV):\n{avg_cm_train}")
        print(f"\nAverage Confusion Matrix for Test (10-fold CV):\n{avg_cm_test}")

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        plt.plot(mean_fpr, mean_tpr, color='b', lw=2, alpha=0.8, label=f'Mean ROC (AUC: {mean_auc:.3f} ± {std_auc:.3f})')
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'± 1 std. dev.')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic for {name}')
        plt.legend(loc='lower right')
        plt.show()

    for name in ratioNames:
        stratify(name, f'{name}_ref', f'{name}_alt')

if __name__ == '__main__':
    analyse_s2_stats_ex2()
