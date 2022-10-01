from pathlib import Path
from pandas_profiling import ProfileReport

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

"""
Random Forest & param tuning
----
https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

"""

prjct_path = Path("/home/danny/Documents/RandomForest")

[f for f in Path(prjct_path).iterdir()]

df = pd.read_csv(prjct_path/"temps.csv")

# EDA
df.describe()
plt_col = ["temp_1", "temp_2"]
for c in plt_col:
    (df
     .assign(date = pd.to_datetime(df[["year","month","day"]]))
     .plot(x="date", y=c)
     )
"""
EDA with Data profiling
"""
profile = ProfileReport(df, title="Data profiling report")

"""
Modeling
"""
# Prep features
df = pd.concat([df,pd.get_dummies(df["week"])], axis=1)
df.drop("week", axis=1, inplace=True)

labels = df["actual"].to_numpy()
features = df.drop("actual", axis=1)

# Train model
train_X, test_X, train_y, test_y = train_test_split(features, labels, test_size=0.2, random_state=0)

train_y.shape
train_X.shape
test_y.shape
test_X.shape

# Baseline use average as baseline
# Baseline error, use mean_abs_error
baseline_pred = \
(df
 .loc[test_X.index,]
 .assign(baseline_error = lambda x : np.abs(x['average']-x['actual']))
)["baseline_error"].mean()


# Model training
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=1000, random_state=0)
rf.fit(train_X, train_y)

pred = rf.predict(test_X)

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

err = mean_absolute_error(test_y, pred)
mape = mean_absolute_percentage_error(test_y, pred)
accuracy = 1 - mape

"""
Visualize tree
"""

# sklearn
from sklearn.tree import plot_tree

tree = rf.estimators_[5]
fig, ax = plt.subplots(figsize=(25,20))
plot_tree(tree, max_depth=3,
          feature_names=rf.feature_names_in_,
          filled=True,  ax=ax)

# Graphviz
# install with conda use `conda install python-graphviz`
import graphviz
from sklearn.tree import export_graphviz

tree_viz = export_graphviz(tree, out_file=None,
                           feature_names=rf.feature_names_in_,
                           filled=True)
viz = graphviz.Source(tree_viz, format="png")
viz

# Dtreeviz
from dtreeviz.trees import dtreeviz

viz = dtreeviz(tree, features, labels, target_name="target",
               feature_names=rf.feature_names_in_)
viz

"""
Features Importance
----
https://mljar.com/blog/feature-importance-in-random-forest/

"""
# (A) : Features importance build in RandomForest Model
# (Gini importance) calculate the mean decrease imputity
# of how each features decrease the gini impure
# :- have bias on numerical features, high cordinallity categorical

feat_imp = pd.DataFrame({"feat_name":rf.feature_names_in_,
                          "imp":rf.feature_importances_})
feat_imp.sort_values("imp").plot(x="feat_name", y="imp", kind="barh")

# (B) Permutation based feature importance

from sklearn.inspection import permutation_importance

perm_imp = permutation_importance(rf, test_X, test_y)

pimp_df = pd.DataFrame({"feat_name":rf.feature_names_in_,
                       "imp":perm_imp.importances_mean})
plt.show(pimp_df.sort_values("imp").plot(x="feat_name", y="imp", kind="barh"))


"""
Features Importance with SHAP
----
https://towardsdatascience.com/using-shap-values-to-explain-how-your-machine-learning-model-works-732b3f40e137
https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/decision_plot.html

"""
import shap

# Fit explainer
explainer = shap.Explainer(rf.predict, test_X)
shap_values = explainer(test_X)

shap.summary_plot(shap_values, test_X, plot_type="bar")

# Features importance
# Calculate from mean avg value of each features
from scipy.special import softmax

def print_feature_importances_shap_values(shap_values, features):
    '''
    Prints the feature importances based on SHAP values in an ordered way
    shap_values -> The SHAP values calculated from a shap.Explainer object
    features -> The name of the features, on the order presented to the explainer
    '''
    # Calculates the feature importance (mean absolute shap value) for each feature
    importances = []
    for i in range(shap_values.values.shape[1]):
        importances.append(np.mean(np.abs(shap_values.values[:, i])))
    # Calculates the normalized version
    importances_norm = softmax(importances)
    # Organize the importances and columns in a dictionary
    feature_importances = {fea: imp for imp, fea in zip(importances, features)}
    feature_importances_norm = {fea: imp for imp, fea in zip(importances_norm, features)}
    # Sorts the dictionary
    feature_importances = {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse = True)}
    feature_importances_norm= {k: v for k, v in sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse = True)}
    # Prints the feature importances
    for k, v in feature_importances.items():
        print(f"{k} -> {v:.4f} (softmax = {feature_importances_norm[k]:.4f})")
    return None

print_feature_importances_shap_values(shap_values=shap_values, features=test_X.columns)

# Global effect
shap.plots.bar(shap_values)
shap.plots.beeswarm(shap_values)
shap.summary_plot(shap_values, plot_type="violin")

# Local (instance, obs) effect
shap.plots.bar(shap_values[0])
shap.plots.waterfall(shap_values[0])
# shap.plots.force(expected_value)

# To be Done : TreeExplainer
# explainer_rf = shap.TreeExplainer(rf)
# expected_value = explainer_rf.expected_value
# expected_value = expected_value[0]

# shap_values = explainer.shap_values(test_X)
# shap.decision_plot(expected_value[0], shap_values[0])

"""
Hyperparameter tuning
----
"""
# RandomSearch

from sklearn.model_selection import RandomizedSearchCV

random_grid = {"n_estimators":[10, 100, 500, 1000],
               "max_features":[1.0, "sqrt", "log2"],
               }

rf = RandomForestRegressor()
rf_cv = RandomizedSearchCV(estimator=rf,
                           scoring="neg_mean_absolute_error",
                           param_distributions=random_grid,
                           n_iter=100, cv=3, verbose=2, random_state=0)
rf_cv.fit(train_X, train_y)

rf_cv.best_params_

base_rf = RandomForestRegressor(n_estimators=10, random_state=0)
base_rf_model = base_rf.fit(train_X, train_y)
mean_absolute_error(base_rf_model.predict(test_X), test_y)

best_rf_model = rf_cv.best_estimator_
mean_absolute_error(best_rf_model.predict(test_X), test_y)