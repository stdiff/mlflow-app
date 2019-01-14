from unittest import TestCase

from lib.modeling import *

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


class ModelingTest(TestCase):
    def test_add_prefix_to_param(self):
        pref = "YourPrefix"
        before = {"param_1": [1,2,3],
                  "param_2": [5,9,1]}
        correct_keys = ["YourPrefix__param_1","YourPrefix__param_2"]

        after = add_prefix_to_param(pref, before)

        self.assertTrue(isinstance(after,dict))
        self.assertEqual(correct_keys, sorted(after.keys()))
        self.assertEqual(after[correct_keys[0]], before["param_1"])
        self.assertEqual(after[correct_keys[1]], before["param_2"])


    def test_cv_results_summary_and_show_coefficients(self):
        data = load_iris()
        X = data["data"]
        y = [data.target_names[i] for i in data["target"]]
        cols = [s[:-5].replace(" ", "_") for s in data.feature_names]

        param_grid = { "C": [1,10], "penalty": ["l1","l2"]}
        model = GridSearchCV(LogisticRegression(),
                             param_grid,
                             scoring="accuracy",
                             cv=3,
                             refit=True,
                             iid=True)
        model.fit(X,y)
        df = cv_results_summary(model)

        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertEqual(df.index.name, "rank_test_score")
        self.assertEqual(df.shape, (4,5))
        self.assertEqual(list(df.index), sorted(list(df.index)))

        df_coef = show_coefficients(model.best_estimator_, cols)

        self.assertTrue(isinstance(df_coef, pd.DataFrame))
        self.assertTrue("intercept" in df_coef.index)
        self.assertEqual(df_coef.shape, (5,3))

        model = DecisionTreeClassifier(max_depth=5, random_state=5)
        model.fit(X,y)

        s = show_feature_importance(model, cols)

        self.assertTrue(isinstance(s, pd.Series))
        self.assertEqual(set(s.index), set(cols))
        self.assertTrue(s[0] >= s[len(s)-1])


    def test_ROCCurve(self):
        data = load_breast_cancer()
        X = data.data
        y = [data.target_names[i] for i in data.target]
        cols = data.feature_names
        pos_label = data.target_names[1]

        model = LogisticRegression(C=10)
        model.fit(X,y)

        ## test of show_coefficients for a binary classifier
        df_coef = show_coefficients(model,cols)

        self.assertTrue(isinstance(df_coef, pd.DataFrame))
        self.assertTrue("intercept" in df_coef.index)
        self.assertEqual(df_coef.shape, (len(cols)+1,1))

        Y_score = pd.DataFrame(model.predict_proba(X),
                               columns=model.classes_)
        y_score = Y_score[pos_label]
        roc = ROCCurve(y, y_score, pos_label=pos_label)

        auc = roc.get_auc()

        ## The following two lines just check the above codes.
        self.assertTrue(isinstance(y_score, pd.Series))
        self.assertTrue(auc > 0.5)

        ## test predict_thru_threshold
        y_hat = roc.predict_thru_threshold(0.5)
        self.assertTrue(isinstance(y_hat,np.ndarray))
        self.assertEqual(y_hat.shape, (len(y),))

        ## test get_scores_thru_threshold
        s = roc.get_scores_thru_threshold(0.5)

        self.assertTrue(isinstance(s, pd.Series))
        self.assertEqual(len(s), 6)
        self.assertEqual(s.dtype, np.float)

        ## A threshold can be larger than 1 if you use an
        ## ordinary regression model. Other values must be
        ## lie in [0,1]
        self.assertEqual((s[1:] <= 1).sum(), 5)
        self.assertEqual((s[1:] >= 0).sum(), 5)

        ## test get_confusion_matrix
        df_ct = roc.get_confusion_matrix(0.5)

        self.assertTrue(isinstance(df_ct, pd.DataFrame))
        self.assertEqual(df_ct.columns.name, "true value")
        self.assertEqual(df_ct.index.name, "prediction")
        self.assertEqual(list(df_ct.columns), [0,1])
        self.assertEqual(list(df_ct.index), [0,1])
        self.assertEqual(df_ct.loc[:,1].sum(), roc.y_true.sum())

        df_scores = roc.get_scores()

        self.assertTrue(isinstance(df_scores, pd.DataFrame))
        self.assertEqual(df_scores.shape,
                         (len(roc.thresholds), 5) )
        self.assertEqual(df_scores.index.name, "threshold")