# importing all libraries to be used in the project
import pandas as pd
import data_preprocessing as dp
import show_diagrams as sd
import os
from sklearn.model_selection import train_test_split
import models as md
import evaluation as ev
import warnings

# ignore warnings from popping up. only show errors
warnings.filterwarnings(action='ignore')

'''
This function displays the data which has been loaded.
In this function, the data type for each data was check as well.
:param filename represents the name of the data.
'''


def check_data(filename):
    data_info = filename.count()
    print("_____________________________\n Data information \n",data_info)
    print("\n __________________________ \n Observations \n", filename.info())


# get the current working directory
working_direc = os.getcwd()

# create variables that holds the data for both the train and test data set
train_file = "data_train.csv"
test_file = "data_test.csv"
# Combining the train and test files with the current working directory
train_data_ = os.path.join(working_direc, train_file)
test_data_ = os.path.join(working_direc, test_file)

# load and store both the train and test data file
train_data = pd.read_csv(train_data_)
test_data = pd.read_csv(test_data_)

# check the data, to ascertain the number of features present, the data-type of each feature
# the total number of observation present in each feature.
check_data(train_data)

'''
Check for the total number of Missing values for each column and number of missing values each column has.
 '''
dp.check_missing_value(train_data)

# extracting our features.
# remove the target columns from the features
X= train_data.drop(['texture','color'], axis=1)
# generating the color target from the data set
y_color = train_data['color']
# generate the texture target from the data set
y_texture = train_data['texture']

'''
Visualise and generate the count of each label in the target or class to understand the nature of the data.
By nature, it implies either being balanced or imbalanced.
'''
dp.check_data_nature(y_color,"Color")
dp.check_data_nature(y_texture, "texture")

""" ------------------------------------------------------------------------------------------
Classification model for color target
---------------------------------------------------------------------------------------------- """
# splitting the data using train test split for color
X_train_c, X_valid_c, y_train_c, y_valid_c = train_test_split(X, y_color,
                                                            random_state=123, stratify=y_color)
# performing missing value imputation and feature selection.
# returns the model for featuring scaling, best_parameters, and some values for parameters (depth, random
# state and criterion -> f
feature_model_result_c = dp.feature_select_scaling(X_train_c, y_train_c, 15, 'entropy')

"""
get the selected features.
 modified from
 https://stackoverflow.com/questions/39839112/the-easiest-way-for-getting-feature-names-after-running-selectkbest-in-scikit-le
"""
selected_features_c = X_train_c[X_train_c.columns[feature_model_result_c.get_support()]]
# get the feature importances of the features
feature_importances = pd.Series(feature_model_result_c.estimator_.feature_importances_,
                                index = X_train_c.columns)
# show the important features for color.,
sd.plot_features(feature_importances, "color")

# show the distribution of the data, if its linearly seperable or not.
# combine the data along the column axis. Here we checked using the best features.
# 3 columns were selected, since we had several important features.
combined_data_c = pd.concat([selected_features_c.iloc[:, 1:4], y_train_c], axis=1)
# show a plot of the data
sd.show_distribution_plot(combined_data_c, "color")


# perform scaling and model training
# for modelling training, the use of Support Vector machine was used.
svm_model = md.svm_model(X_train_c, y_train_c, 1.0, 15, 12, 0.01)

# performing modelling and scaling using the logistic regression algorithm
# using the l2 regularization parameter.
log_model = md.logistic_model(X_train_c,y_train_c,'l2','balanced',15)

# k nearest neighbour on train data set -> advanced part
knn_color = md.nearest_neigh(X_train_c, y_train_c, 5, 'manhattan', 2)

# Evaluating both model performance for the support vector machine and the logistic model and knn
evaluation_report = ev.model_evaluation(svm_model, log_model, knn_color, X_valid_c, y_valid_c, "color")

print ("\n\n\n\n----------------------------------------------"
       "\nEvaluation report for color using different model")
print ("_________________________________________________")
print(evaluation_report)
print("_________________________________________________\n\n\n\n\n")

"""
-------------------------------------------------------------------------------------------
Test Data Analysis 
-------------------------------------------------------------------------------------------
"""
# analyzing the test data.
# check if it contains missing data
dp.check_missing_value(test_data)

# check the data, to ascertain the number of features present, the data-type of each feature
# the total number of observation present in each feature.
check_data(test_data)
color_pred = svm_model.predict(test_data)
color_test = pd.DataFrame(color_pred)
color_test.to_csv('colour_test.csv', index=False, index_label=False, header=None)


"""--------------------------------------------------------------------------------------------
Texture Classification problem.

-----------------------------------------------------------------------------------------------"""
X_train_t, X_valid_t, y_train_t, y_valid_t = train_test_split(X, y_texture, random_state = 123, stratify=y_texture)
# performing missing value imputation and feature selection.
# returns the model for featuring scaling, best_parameters, and some values for parameters (depth, random
# state and criterion -> f
feature_model_result_t = dp.feature_select_scaling(X_train_t, y_train_t, 12, 'entropy')

"""
get the selected features.
 modified from
 https://stackoverflow.com/questions/39839112/the-easiest-way-for-getting-feature-names-after-running-selectkbest-in-scikit-le
"""
selected_features_t = X_train_t[X_train_t.columns[feature_model_result_t.get_support()]]
# get the feature importances of the features
feature_importances_t = pd.Series(feature_model_result_t.estimator_.feature_importances_,
                                index = X_train_t.columns)
# show the important features for texture.,
sd.plot_features(feature_importances_t, "texture")

# show the distribution of the data, if its linearly seperable or not.
# combine the data along the column axis. Here we checked using the best features.
# 3 columns were selected, since we had several important features.
combined_data_t = pd.concat([selected_features_t.iloc[:, 1:4], y_train_t], axis=1)
# show a plot of the data
sd.show_distribution_plot(combined_data_t, "texture")


# perform scaling and model training
# for modelling training, the use of Support Vector machine was used.
svm_model_t= md.svm_model(X_train_t, y_train_t, 1.0, 12, 12, 0.001)

# performing modelling and scaling using the logistic regression algorithm
# using the l2 regularization parameter.
log_model_t = md.logistic_model(X_train_t,y_train_t,'l2','balanced',12)

# k nearest neighbour on train data set -> advanced part
knn_texture = md.nearest_neigh(X_train_t, y_train_t, 4, 'manhattan', 2)

# Evaluating both model performance for the support vector machine, the logistic model and knn
evaluation_report_t = ev.model_evaluation(svm_model_t, log_model_t, knn_texture,  X_valid_t,  y_valid_t, "texture")

print ("\n\n\n\n----------------------------------------------"
       "\nEvaluation report for Texture using different model")
print ("_________________________________________________")
print(evaluation_report_t)
print("_________________________________________________\n\n\n\n\n")

"""
-------------------------------------------------------------------------------------------
Test Data Analysis 
-------------------------------------------------------------------------------------------
"""
texture_pred = svm_model_t.predict(test_data)
texture_test = pd.DataFrame(texture_pred)
texture_test.to_csv('texture_test.csv', index=False, index_label=False, header=None)
