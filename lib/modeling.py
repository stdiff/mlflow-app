import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

## for show_tree
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydot

## for ROCCurve class
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import recall_score, precision_score, f1_score
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

grid_params = {
    "ElasticNet": {
        'alpha': [0.01, 0.1, 1.0],
        'l1_ratio': [0.1, 0.4, 0.7, 1]
    },
    "LogisticRegression": {
        'C': [0.1, 1.0, 10]
    },
    "DecisionTree": {
        'max_leaf_nodes': [3, 6, 12, 24]
    },
    "RandomForest": {
        'n_estimators': [10, 30, 50],
        'max_depth': [3, 5]
    },
    "XGB": {
        'learning_rate': [0.01, 0.1],
        'n_estimators': [5, 10, 20],
        'max_depth': [5, 10]}
}


class MultiTransformer():
    def __init__(self, cls, columns):
        pass


def add_prefix_to_param(prefix:str, param:dict) -> dict:
    """
    Create a dict for scikit-learn ML pipeline from given grid_param

    :param prefix: name of the step
    :param param: ordinary grid_param
    :return: modified dict
    """
    return { "%s__%s" % (prefix,k): v for k,v in param.items()}


def cv_results_summary(grid:GridSearchCV) -> pd.DataFrame:
    """
    Make the result of CV more smaller.

    :param grid: fitted instance of GridSearchCV
    :return: part of DataFrame of cv_results_
    """

    param_list = [k for k in grid.cv_results_.keys()
                  if k.startswith("param_")]

    cols = ["rank_test_score","mean_test_score", "std_test_score",
            "mean_train_score"] + param_list

    df = pd.DataFrame(grid.cv_results_, columns=cols)
    df.set_index(cols[0], inplace=True)

    return df.sort_index()


def show_coefficients(model, columns) -> pd.DataFrame:
    """
    Show the coefficients of variables in a linear model.

    :param model: trained model
    :param columns: list of column names
    :return: DataFrame of coefficients
    """

    labels = model.classes_ if len(model.classes_) > 2 else [model.classes_[1]]
    df_coef = pd.DataFrame(model.coef_,
                           columns=columns,
                           index=labels)

    df_coef["intercept"] = pd.Series(model.intercept_,
                                     index=labels)

    return df_coef.T


def show_tree(clf, columns):
    """
    Visualize the given trained DecisionTree model on Jupyter.
    This function requires also pydot.

    :param clf: DecisionTree model
    :param columns: names of columns
    :return: Image instance (for Jupyter)
    """

    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data, feature_names=columns,
                    filled=True, rounded=True, special_characters=True)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]
    return Image(graph.create_png())


def show_feature_importance(model,columns:list) -> pd.Series:
    """
    Return the series of feature importance of given random forest model.
    XGB model and DecisionTree model can be accepted as well.

    :param model: RandomForest* or XGB*
    :param columns: list of column names
    :return: Series of feature importance
    """

    s = pd.Series(model.feature_importances_, index=columns,
                  name="importance")
    return s.sort_values(ascending=False)


class ROCCurve:
    def __init__(self, y_true, y_score, pos_label=1):
        """
        This class provides API to calculate performance metrics
        which are relevant to a binary classification.

        :param y_true: a binary array-like object
        :param y_score: an array-like object containing scores for
                        the positive label.
        :param pos_label: positive value in y_true.
        """

        self.y_true = np.array([1 if y == pos_label else 0 for y in y_true])

        if isinstance(y_score, pd.Series):
            ## remove index
            self.y_score = y_score.values
        else:
            self.y_score = y_score

        self.pos_label = pos_label
        self.fpr, self.tpr, self.thresholds = roc_curve(self.y_true, self.y_score, pos_label=1)
        self.thresholds = self.thresholds[1:]
        self.scores = None


    def get_auc(self) -> float:
        """
        return the area under the ROC curve

        :return: AUC
        """
        return roc_auc_score(self.y_true, self.y_score)


    def predict_thru_threshold(self, threshold) -> np.ndarray:
        """
        make a prediction by splitting scores by the given threshold

        :param threshold:
        :return: a binary array of predictions
        """
        return (self.y_score >= threshold).astype(int)


    def get_scores_thru_threshold(self, threshold:float=0.5) -> pd.Series:
        """
        compute the performance scores

        :param threshold: threshold
        :return: Series of scores
        """

        y_pred = self.predict_thru_threshold(threshold)

        s = pd.Series(name="performance")
        s["threshold"] = threshold
        s["recall"] = recall_score(self.y_true, y_pred)
        s["precision"] = precision_score(self.y_true, y_pred)
        s["accuracy"] = np.mean(self.y_true == y_pred)
        s["f1_score"] =  f1_score(self.y_true, y_pred)

        fp = np.logical_and(self.y_true == 0, y_pred == 1).sum()
        tn_fp = np.sum(self.y_true == 0)

        ## tn_pf is never zero because it is the number of
        ## negative labels in the data but just in case
        s["fp_rate"] = fp / tn_fp if tn_fp > 0 else np.nan

        return s


    def get_confusion_matrix(self, threshold:float=0.5) -> pd.DataFrame:
        """
        returns the crosstab of the prediction and the true value
        :param threshold:
        :return: the
        """
        y_pred = self.predict_thru_threshold(threshold)

        return pd.crosstab(
            pd.Series(y_pred, name="prediction"),
            pd.Series(self.y_true, name="true value")
        )


    def get_scores(self) -> pd.DataFrame:
        """
        computes recall, precision, FP-rate, accuracy and F1 score
        for each threshold. The result is returned as a DataFrame.
        This can take a while because of a for-loop.

        :return: DataFrame of metrics (Each row corresponds to a threshold.
        """
        if self.scores is None:
            cols = ["threshold", "recall", "precision", "fp_rate",
                    "accuracy", "f1_score"]
            rows = []
            for t in self.thresholds:
                row = self.get_scores_thru_threshold(t)
                rows.append([row[c] for c in cols])

            self.scores = pd.DataFrame(rows, columns=cols).set_index(cols[0])

        return self.scores

    def show_roc_curve(self):
        """
        plot ROC curve. The AUC is shown in the title
        """

        plt.plot(self.fpr, self.tpr)
        plt.plot([0, 1], [0, 1], "k:")
        plt.title("ROC curve (AUC = %0.4f)" % self.get_auc())
        plt.xlabel("False-Positive Rate")
        plt.ylabel("True-Positive Rate")

    def show_metrics(self):
        """
        plot curves of metrics (recall, precision, accuracy and f1 score.
        The false-positive rate is not shown because it is the only metric
        in the DataFrame which should be smaller and therefore it is
        very confusing.
        """

        self.get_scores().drop("fp_rate", axis=1).plot()
        plt.title("Scores for each threshold")
