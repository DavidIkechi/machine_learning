import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
import warnings
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action="ignore")

"""
This function checks the column of each file to ascertain if it has non-sensical values
:param: filename represents the name of the file whose column is to be checked
"""


def check_missing_value(filename):
    print("_____________________________________________________"
          " \n Checking for number of missing data in each data set")
    # check for the rows with missing values and the total number of missing values in that row.
    for i in range(len(filename.index)):
        if filename.iloc[i].isnull().sum() > 0:
            print("Number of missing value in row {} is {}".format(i, filename.iloc[i].isnull().sum()))


"""
This function checks the nature of the data to ascertain if it's balanced or unbalanced.
:param: filename_column represents the column of the filename to investigate (target)
:param: title represents the name of the column

modified from https://medium.com/swlh/quick-guide-to-labelling-data-for-common-seaborn-plots-736e10bf14a9
"""


def check_data_nature(filename_column, title):
    labels = np.unique(filename_column)  # get the unique labels in the classes
    value_counts = filename_column.value_counts()  # get the total counts of each unique label in the class
    ax = sns.barplot(x=labels, y=value_counts)
    ax.set(xlabel=title, ylabel=title + " counts (arb)", title="Distribution for "+title)
    for p in ax.patches:
        # get the height of each bar
        height = p.get_height()
        # adding text to each bar
        ax.text(x=p.get_x() + (p.get_width() / 2), y=height + 3, s= '{: .0f}'.format(height), ha = 'center')
    plt.savefig(title+"_distribution.png")
    plt.show()


"""
This function performs feature selection and scaling for the data set
The feature selection was carried out to reproduce features that have high influence on the target (class)
Scaling was perform to scale the data, as our model is highly sensitive to scaling.
:param: X represents the training feature (X) 
:param : y represents the training target y
"""

def feature_select_scaling(X, y, depth_, criterion_):
    # get the best maximum depth, criterion and random state
    depth = depth_
    criterion = criterion_

    # treat missing values
    missing_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    X_impute = missing_imputer.fit_transform(X)

    # create a classifier to select the feature importances
    dec_classifier = SelectFromModel(DecisionTreeClassifier(max_depth=depth, random_state=5, criterion=criterion))

    return dec_classifier.fit(X_impute, y)
