# Avoiding warning
import warnings


def warn(*args, **kwargs): pass


warnings.warn = warn
# _______________________________

# Essential Library
import lightgbm as lgb
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# _____________________________

# np.random.seed(seed=111)

# scikit-learn :
from sklearn.linear_model import LogisticRegression, SGDClassifier

#from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
#from imblearn.combine import SMOTEENN
#from imblearn.over_sampling import SMOTE
#from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, \
    RandomForestClassifier, \
    AdaBoostClassifier, \
    GradientBoostingClassifier, \
    ExtraTreesClassifier
#from catboost import CatBoostClassifier, Pool, cv
import lightgbm as lgb



Names = ['LXGB','XGB', 'ERT']
Classifiers = [

    #lgb.LGBMClassifier(n_estimators=500),
    XGBClassifier(n_estimators=500),
    #ExtraTreesClassifier(n_estimators=400),
    #CatBoostClassifier(n_estimators=500),

    #lgb.LGBMClassifier(),

    # SVC(kernel='rbf', C=1.03, gamma=6.123, probability=True),

]


def runClassifiers(args):
    args.dataset = 'K-S-Bigrm-PSSM-VEGF-train-884-922.csv'

    D = pd.read_csv(args.dataset, header=None)
    X = D.iloc[:, :-1].values
    y = D.iloc[:, -1].values


    Results = []  # compare algorithms

    from sklearn.metrics import accuracy_score, \
        log_loss, \
        classification_report, \
        confusion_matrix, \
        roc_auc_score, \
        average_precision_score, \
        auc, \
        roc_curve, f1_score, recall_score, matthews_corrcoef, auc

    # Step 05 : Spliting with 10-FCV :
    from sklearn.model_selection import KFold
    Seed = 1579

    cv = KFold(n_splits=10, shuffle=True, random_state=Seed)

    for classifier, name in zip(Classifiers, Names):
        accuray = []
        auROC = []
        avePrecision = []
        F1_Score = []
        AUC = []
        MCC = []
        Recall = []
        LogLoss = []

        mean_TPR = 0.0
        mean_FPR = np.linspace(0, 1, 100)

        CM = np.array([
            [0, 0],
            [0, 0],
        ], dtype=int)

        print('{} is done.'.format(classifier.__class__.__name__))
        # print(classifier.__class__.__name__+'\n\n')

        model = classifier
        counterPre = 0
        for (train_index, test_index) in cv.split(X, y):
            X_train = X[train_index]
            X_test = X[test_index]

            y_train = y[train_index]
            y_test = y[test_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Save to file in the current working directory
            import pickle
            pkl_filename = "pickle_model2.pkl"

            with open(pkl_filename, 'wb') as file:
                pickle.dump(model, file)

            # Calculate ROC Curve and Area the Curve
            y_proba = model.predict_proba(X_test)[:, 1]
            y_probaR = y_proba.round(3)

            # df = pd.DataFrame(y_proba)
            # df.to_csv('ROC results.csv')

            FPR, TPR, _ = roc_curve(y_test, y_proba)
            mean_TPR += np.interp(mean_FPR, FPR, TPR)
            mean_TPR[0] = 0.0
            roc_auc = auc(FPR, TPR)
            ##########################################
            # print(FPR)
            # print(TPR)
            ##########################################

            y_artificial = model.predict(X_test)

            auROC.append(roc_auc_score(y_true=y_test, y_score=y_proba))

            accuray.append(accuracy_score(y_pred=y_artificial, y_true=y_test))
            avePrecision.append(average_precision_score(y_true=y_test, y_score=y_proba))  # auPR
            F1_Score.append(f1_score(y_true=y_test, y_pred=y_artificial))
            MCC.append(matthews_corrcoef(y_true=y_test, y_pred=y_artificial))
            Recall.append(recall_score(y_true=y_test, y_pred=y_artificial))
            AUC.append(roc_auc)

            CM += confusion_matrix(y_pred=y_artificial, y_true=y_test)

           # np.savetxt(str(counterPre) + '.csv', np.asarray(y_probaR.round(2)))
           # counterPre = counterPre + 1
        accuray = [_ * 1.0 for _ in accuray]
        Results.append(accuray)

        mean_TPR /= cv.get_n_splits(X, y)
        mean_TPR[-1] = 1.0
        mean_auc = auc(mean_FPR, mean_TPR)
        plt.plot(
            mean_FPR,
            mean_TPR,
            linestyle='-',
            label='{} ({:0.3f})'.format(name, mean_auc), lw=2.0)

        TN, FP, FN, TP = CM.ravel()
        print('Accuracy: {0:.2f}%\n'.format(np.mean(accuray) * 100.0))
        print('Sensitivity: {0:.2f}%\n'.format(((TP) / (TP + FN)) * 100.0))
        print('Specificity: {0:.2f}%\n'.format(((TN) / (TN + FP)) * 100.0))
        print('F1_Score: {0:.2f}%\n'.format(np.mean(F1_Score) * 100.0))
        print('MCC: {0:.4f}\n'.format(np.mean(MCC)))
        print('Confusion Matrix:\n')
        print(str(CM) + '\n')
        # np.savetxt('prediction.txt',np.asarray(predictions_))
        # df = pd.DataFrame(y_proba)
        # df.to_csv('ROC results.csv')
        # F.write('_______________________________________'+'\n')

    '''### auROC Curve ###
    if args.auROC == 1:
        auROCplot()
    ### boxplot algorithm comparison ###
    if args.boxPlot == 1:
        boxPlot(Results, Names)
        ### --- ###

    print('\nPlease, eyes on evaluationResults.txt')


def boxPlot(Results, Names):
    ### Algoritms Comparison ###
    # boxplot algorithm comparison
    fig = plt.figure()
    # fig.suptitle('Classifier Comparison')
    ax = fig.add_subplot(111)
    ax.yaxis.grid(True)
    plt.boxplot(Results, patch_artist=True, vert=True, whis=True, showbox=True)
    ax.set_xticklabels(Names, fontsize=12)
    plt.xlabel('Classifiers', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)

    #plt.savefig('Accuracy_boxPlot.png', dpi=300)
    #plt.show()
    ### --- ###


def auROCplot():
    ### auROC ###
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Random')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('ROC curve', fontweight='bold')
    plt.legend(loc='lower right')
    #plt.savefig('auROC.png', dpi=300)
    #plt.show()
    ### --- ###

    #  --- ###'''



if __name__ == '__main__':
    # print('Please, enter number of cross validation:')
    import argparse

    p = argparse.ArgumentParser(description='Run Machine Learning Classifiers.')

    p.add_argument('-cv', '--nFCV', type=int, help='Number of crossValidation', default=10)
    p.add_argument('-data', '--dataset', type=str, help='~/dataset.csv', default='optimumDataset.csv')
    p.add_argument('-roc', '--auROC', type=int, help='Print ROC Curve', default=1, choices=[0, 1])
    p.add_argument('-box', '--boxPlot', type=int, help='Print Accuracy Box Plaot', default=1, choices=[0, 1])

    args = p.parse_args()

    runClassifiers(args)