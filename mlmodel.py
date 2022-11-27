# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import datetime
import pandas as pd
import numpy as np
import pickle
from collections import Counter

from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing
from sklearn import svm

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

DATA_FOLDER = 'data'


def load_data(filename):
    _data_df = pd.read_csv(DATA_FOLDER + '\\' + filename)
    return _data_df


def test_train(data_df):
    print('test_train....')

    # Labels are the values we want to predict
    labels = np.array(data_df['flag'])
    # Remove the labels from the features
    # axis 1 refers to the columns
    features = data_df.drop(['flag'], axis=1)

    pickle.dump(features.columns, open('sim_df_model.features', 'wb'))

    # Saving feature names for later use
    feature_list = list(features.columns)
    print(len(feature_list))

    # perform a robust scaler transform of the dataset
    trans = StandardScaler()
    features = trans.fit_transform(features)

    # feature extraction
    model = ExtraTreesClassifier(n_estimators=10)
    model.fit(features, labels)
    print(model.feature_importances_)

    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                                random_state=1, stratify=labels)
    print(Counter(train_labels))
    print(Counter(test_labels))

    log_reg = LogisticRegression(random_state=0, solver='saga', max_iter=1000, multi_class='auto').fit(train_features,
                                                                                                       train_labels)
    print(train_features.shape)
    filename = 'log_reg_model.sav'
    pickle.dump(log_reg, open(filename, 'wb'))
    y_pred1 = log_reg.predict(test_features)
    print("Log Reg Accuracy:", metrics.accuracy_score(test_labels, y_pred1))
    confusion_matrix = pd.crosstab(test_labels, y_pred1, rownames=['Actual'], colnames=['Predicted'])
    print(confusion_matrix)

    # Instantiate model with 1000 decision trees
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(train_features, train_labels)

    filename = 'rf_class_model.sav'
    pickle.dump(rf_model, open(filename, 'wb'))
    y_pred2 = rf_model.predict(test_features)
    print("RF Accuracy:", metrics.accuracy_score(test_labels, y_pred2))
    confusion_matrix = pd.crosstab(test_labels, y_pred2, rownames=['Actual'], colnames=['Predicted'])
    print(confusion_matrix)

    knn = DecisionTreeClassifier()
    knn.fit(train_features, train_labels)
    filename = 'dtree_class_model.sav'
    pickle.dump(knn, open(filename, 'wb'))
    y_pred3 = knn.predict(test_features)
    print("DecisionTreeClassifier Accuracy:", metrics.accuracy_score(test_labels, y_pred3))
    confusion_matrix = pd.crosstab(test_labels, y_pred3, rownames=['Actual'], colnames=['Predicted'])
    print(confusion_matrix)

    # predict probabilities
    pred_prob1 = log_reg.predict_proba(test_features)
    pred_prob2 = rf_model.predict_proba(test_features)
    pred_prob3 = knn.predict_proba(test_features)

    # roc curve for models
    fpr1, tpr1, thresh1 = roc_curve(test_labels, pred_prob1[:, 1], pos_label=1)
    fpr2, tpr2, thresh2 = roc_curve(test_labels, pred_prob2[:, 1], pos_label=1)
    fpr3, tpr3, thresh3 = roc_curve(test_labels, pred_prob3[:, 1], pos_label=1)

    # roc curve for tpr = fpr
    random_probs = [0 for i in range(len(test_labels))]
    p_fpr, p_tpr, _ = roc_curve(test_labels, random_probs, pos_label=1)

    # auc scores
    auc_score1 = roc_auc_score(test_labels, pred_prob1[:, 1])
    auc_score2 = roc_auc_score(test_labels, pred_prob2[:, 1])
    auc_score3 = roc_auc_score(test_labels, pred_prob3[:, 1])

    print(auc_score1, auc_score2, auc_score3)

    # plot roc curves
    plt.plot(fpr1, tpr1, linestyle='--', color='orange', label='Logistic Regression')
    plt.plot(fpr2, tpr2, linestyle='--', color='green', label='Random Forest Classifier')
    plt.plot(fpr3, tpr3, linestyle='--', color='blue', label='DecisionTree Classifier')
    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
    # title
    plt.title('ROC curve')
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')

    plt.legend(loc='best')
    plt.savefig('ROC', dpi=300)
    plt.show();


def build_features(data_df):
    arr_index = data_df.idxname.unique()
    le = preprocessing.LabelEncoder()
    le.fit(arr_index)
    data_df['idxname'] = le.transform(data_df['idxname'])
    return data_df


def pred_simulations(data_df):
    print('pred_simulations....')
    _data_df = build_features(data_df)
    labels = np.array(_data_df['flag'])

    # Remove the labels from the features
    # axis 1 refers to the columns
    _data_df = _data_df.replace(np.nan, 0)
    features = _data_df.drop(['flag'], axis=1)
    print(features.shape)

    filename = 'rf_class_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    y_pred = loaded_model.predict(features)
    print("rf Predicted:", metrics.accuracy_score(labels, y_pred))
    confusion_matrix = pd.crosstab(labels, y_pred, rownames=['Actual'], colnames=['Predicted'])
    print(confusion_matrix)
    return y_pred


if __name__ == '__main__':

    # filename = 'r_feature.csv'
    # _data_df = load_data(filename)
    # print(_data_df.shape)
    # _data_df = build_features(_data_df)
    # test_train(_data_df)

    filename = 'r_test.csv'
    _data_df = load_data(filename)
    print(_data_df.shape)
    pred_simulations(_data_df)
