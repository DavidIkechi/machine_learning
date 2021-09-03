import matplotlib.pyplot as plt
import seaborn as sns


"""
This function plots the first 10 best or important features selected by the DecisionTree.
:param: X represents the X train data set.
:param: model_feature represents the model gotten from GridSearchCV for Decision Tree, when
computing for best features that highly influenced our targets.
:param: title represents the title of the target.

modified from: https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
"""


def plot_features(X, title):
    # get the total important features for color
    # show a plot of the first 10 best feature importance for a visualization
    first_10= X.nlargest(10)

    labels = first_10.index  # get the unique labels in the classes
    y = first_10.values
    ax = sns.barplot(x=labels, y=y)
    ax.set(xlabel="features", ylabel=title + " Importances(arb)", title="Feature Importances for " + title +
                                                                        " for 10 features from all selected features ")
    plt.savefig(title + "_feature_distribution.png")
    plt.xticks(rotation=90)
    plt.show()


"""
:param: data represents the data set containing the X and y columns (for features and class respectively)

"""


def show_distribution_plot(data, title):
    ax = sns.pairplot(data=data, hue=title)
    plt.suptitle("Data Distribution for "+title+ " to check linearity")
    plt.tight_layout()
    plt.savefig(title+"_lin_non_lin_check.png")
    plt.show()

