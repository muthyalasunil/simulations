# This is a sample Python script.

import pickle

import matplotlib.pyplot as plt
import numpy as np
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.saving.save import load_model
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn import metrics, preprocessing
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

    # perform a robust scaler transform of the dataset
    trans = StandardScaler()
    features = trans.fit_transform(features)

    # feature extraction
    model = ExtraTreesClassifier(n_estimators=10)
    model.fit(features, labels)
    print(model.feature_importances_)

    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                                random_state=1)

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
    print(y_pred)
    confusion_matrix = pd.crosstab(labels, y_pred, rownames=['Actual'], colnames=['Predicted'])
    print(confusion_matrix)
    return y_pred


# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(32, input_dim=32, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def larger_model():
    # create model
    model = Sequential()
    model.add(Dense(32, input_shape=(32,), activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # filename = 'r_feature.csv'
    # _data_df = load_data(filename)
    # print(_data_df.shape)
    # _data_df = build_features(_data_df)
    # test_train(_data_df)
    #
    # filename = 'r_test.csv'
    # _data_df = load_data(filename)
    # print(_data_df.shape)
    # pred_simulations(_data_df)

    data_df = load_data('r_feature_l.csv')
    arr_index = data_df.idxname.unique()
    le = preprocessing.LabelEncoder()
    le.fit(arr_index)
    data_df['idxname'] = le.transform(data_df['idxname'])

    dataset = data_df.values
    X = dataset[:, 0:32].astype(float)
    Y = dataset[:, 32]
    print(X.shape)
    print(Y.shape)


def predict_nn(data_df):
    # load dataset
    # data_df = load_data('r_feature.csv')
    arr_index = data_df.idxname.unique()
    le = preprocessing.LabelEncoder()
    le.fit(arr_index)
    data_df['idxname'] = le.transform(data_df['idxname'])

    dataset = data_df.values
    X = dataset[:, 0:32].astype(float)
    Y = dataset[:, 32]
    print(X.shape)
    print(Y.shape)

    # load model
    model = load_model('model.h5')
    # summarize model.
    model.summary()

    # make class predictions with the model
    predictions = (model.predict(X) > 0.5).astype(int)
    print("rf Predicted:", metrics.accuracy_score(Y, predictions))
    return predictions

    # summarize the first 5 cases
    #for i in range(len(Y)):
        #print('=> %d (expected %d)' % (predictions[i], Y[i]))

    # estimator = KerasClassifier(build_fn=larger_model, epochs=200, batch_size=5, verbose=0)
    # kfold = StratifiedKFold(n_splits=10, shuffle=True)
    # results = cross_val_score(estimator, X, Y, cv=kfold)
    # print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    # estimators = []
    # estimators.append(('standardize', StandardScaler()))
    # estimators.append(('mlp', KerasClassifier(model=baseline_model, epochs=100, batch_size=5, verbose=0)))
    # pipeline = Pipeline(estimators)
    # kfold = KFold(n_splits=10, shuffle=True) #StratifiedKFold(n_splits=10, shuffle=True)
    # results = cross_val_score(pipeline, X, Y, cv=kfold)
    # print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


def build_save_nn(data_df):
    # data_df = load_data('r_feature_l.csv')
    arr_index = data_df.idxname.unique()
    le = preprocessing.LabelEncoder()
    le.fit(arr_index)
    data_df['idxname'] = le.transform(data_df['idxname'])

    dataset = data_df.values
    X = dataset[:, 0:32].astype(float)
    Y = dataset[:, 32]
    print(X.shape)
    print(Y.shape)
    # define model
    model = Sequential()
    model.add(Dense(96, input_dim=32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model -acc 84, 63
    model.fit(X, Y, epochs=200, batch_size=10)
    # evaluate the model
    scores = model.evaluate(X, Y)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    # save model and architecture to single file
    model.save("model.h5")
    print("Saved model to disk")


def evaluate_nn(data_df):
    # load model
    model = load_model('model.h5')
    # summarize model.
    model.summary()
    # load dataset
    # data_df = load_data('r_feature.csv')
    arr_index = data_df.idxname.unique()
    le = preprocessing.LabelEncoder()
    le.fit(arr_index)
    data_df['idxname'] = le.transform(data_df['idxname'])

    dataset = data_df.values
    X = dataset[:, 0:32].astype(float)
    Y = dataset[:, 32]
    print(X.shape)
    print(Y.shape)

    # evaluate the model
    score = model.evaluate(X, Y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
