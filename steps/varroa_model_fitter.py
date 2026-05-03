# Making a process in ClearML
from clearml import Task

task = Task.init(project_name="Varroa Prediction Pipeline", task_name="Model Fitter")

#Getting tensors
upstream_id = task.get_parameter("General/upstream_id")

combined_binary = Task.get_task(task_id=upstream_id).artifacts["processed_tensors"].get()


# %% [markdown]
# ## Varroa Infestation Prediction - Model Fitter

# %% [markdown]
# #### Authors: Evan Kennedy, Mit Ghandi, Anbuchelvan Kumaravel

# %% [markdown]
# **Library Imports**

# %%
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

# %% [markdown]
# **Modelling**

# %% [markdown]
# Train-Validation-Test Split (60%, 20%, 20%)

# %%
#Note: Stratifying these splits by month to ensure roughly even distribution of seasonsality

#X and Y split
y_bin = combined_binary['infested']
X_bin = combined_binary.drop(columns=['sampling_id', 'infested', 'yard_id', 'hive_id', 'station_id'])

#Train-Validation-Test Split
from sklearn.model_selection import train_test_split

#Splitting off Test Set (20%)
X_bin_temp, X_bin_test, y_bin_temp, y_bin_test = train_test_split(
    X_bin, y_bin, test_size=0.2, stratify=X_bin['obs_month'], random_state=37
)

#Splitting remaining into Train and Validation Sets (60% & 20%)
X_bin_train, X_bin_val, y_bin_train, y_bin_val = train_test_split(
    X_bin_temp, y_bin_temp, test_size=0.25, stratify=X_bin_temp['obs_month'], random_state=37
)

# %% [markdown]
# Ensemble Regression (ie. Gradient Boosting w/ many Regression Trees)

# %%
#Optimizing hyperparameters, focusing on maximizing ROC-AUC
import optuna
from sklearn.metrics import log_loss, roc_auc_score
from optuna.samplers import TPESampler

#Running optimization
def objective(trial):
    params = {
        "max_depth":        trial.suggest_int("max_depth", 3, 15),
        "learning_rate":    trial.suggest_float("learning_rate", 0.001, 0.3),
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 5.0),
        "gamma":            trial.suggest_float("gamma", 0.0, 2.0),
    }

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        early_stopping_rounds=50,
        random_state=37,
        nthread=1,
        tree_method='exact',
        **params
    )
    model.fit(
        X_bin_train, y_bin_train,
        eval_set=[(X_bin_val, y_bin_val)],
        verbose=False
    )

    preds = model.predict_proba(X_bin_val)[:, 1]
    auc = roc_auc_score(y_bin_val, preds)
    logloss = log_loss(y_bin_val, preds)
    
    return auc

study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=37))
study.optimize(objective, n_trials=50)

# Best results
print("Best AUC-ROC:", study.best_value)
print("Best params:", study.best_params)

# %%
#Using optimized hyperparameters
best_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    early_stopping_rounds=50,
    random_state=37,
    **study.best_params
)
best_model.fit(
    X_bin_train, y_bin_train,
    eval_set=[(X_bin_val, y_bin_val)],
    verbose=False
)

# %% [markdown]
# Classification Performance

# %%
from sklearn.metrics import accuracy_score, log_loss

# Predicted class labels
y_bin_test_pred = best_model.predict(X_bin_test)

# Predicted probabilities
y_bin_test_prob = best_model.predict_proba(X_bin_test)

# Accuracy
accuracy = accuracy_score(y_bin_test, y_bin_test_pred)

# Log loss (binary or multiclass-safe)
loss = log_loss(y_bin_test, y_bin_test_prob)

print("Accuracy:", accuracy)
print("Log Loss:", loss)

# %%
from sklearn.metrics import roc_curve, roc_auc_score

#ROC-AUC Score
y_bin_test_probs = best_model.predict_proba(X_bin_test)[:, 1]
auc = roc_auc_score(y_bin_test, y_bin_test_probs)
print(f"ROC-AUC Score: {auc:.4f}")

#ROC Curve
fpr, tpr, thresholds = roc_curve(y_bin_test, y_bin_test_probs)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

#Random guessing diagonal for visual comparison
plt.plot([0, 1], [0, 1], 'k--')
plt.show()

# %% [markdown]
# Exporting Model

# %%
#Saving model
task.upload_artifact("trained_model", artifact_object=best_model)


