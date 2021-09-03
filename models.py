from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.pipeline import Pipeline
from warnings import simplefilter
import warnings


# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action="ignore")


"""
This function trains the model on the data set, using the selected features.
:param: X_train represents the training feature or independent data
:param: y_train represents the target data
"""


def svm_model(X_train, y_train, penalty, depth, state, gamma_):
    # generate the best and optimal parameters for the svm models
    # get the best maximum depth, criterion and random state
    c = penalty
    kernel = 'rbf'
    gamma = gamma_
    class_weight = "balanced"
    state = 12

    pipe = Pipeline([('simple', SimpleImputer(missing_values=np.nan, strategy='median')),
                     ('model', SelectFromModel(DecisionTreeClassifier(max_depth=depth, random_state=5, criterion='entropy'))),
                     ('scalar', StandardScaler()),
                     ('svm', SVC(C=c, class_weight=class_weight,gamma=gamma, random_state=state,kernel=kernel, probability=True))
                     ])

    return pipe.fit(X_train, y_train)


"""
This function generates a logistic model that learns from the training dataset
:param X_train and y_train represents the X and y training datasets
:param penalty represents a penalty added to the model
:param: class_weight represent the weight assigned to the majority and minority classes,
for this data set, balanced has been used to assign more weight to minority class 
to reduce bias toward classifying them.

:return the model
"""


def logistic_model(X_train, y_train, penalty, class_weight, depth):
    pipe = Pipeline([('simple', SimpleImputer(missing_values=np.nan, strategy='median')),
                     ('model',
                      SelectFromModel(DecisionTreeClassifier(max_depth=depth, random_state=5, criterion='entropy'))),
                     ('scalar', StandardScaler()),
                     ('log_model', LogisticRegression(penalty=penalty, class_weight=class_weight, max_iter=10000))
                     ])

    return pipe.fit(X_train, y_train)

"""
Advanced Model, using the KNN Classifier.
"""

"""
This function generates a KNN model that memorizes the training data in a bid to understand new data
:param X_train and y_train represents the X and y training datasets
:param penalty represents a penalty added to the model
:param: class_weight represent the weight assigned to the majority and minority classes,
for this data set, balanced has been used to assign more weight to minority class 
to reduce bias toward classifying them.

:return the model
"""

def nearest_neigh(X_train, y_train, k_value, distance, depth):
    pipe = Pipeline([('simple', SimpleImputer(missing_values=np.nan, strategy='median')),
                     ('model',
                      SelectFromModel(DecisionTreeClassifier(max_depth=depth, random_state=5, criterion='entropy'))),
                     ('scalar', StandardScaler()),
                     ('knn_model', KNeighborsClassifier(n_neighbors=k_value, metric=distance))
                     ])

    return pipe.fit(X_train, y_train)