import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    average_precision_score, brier_score_loss, classification_report,
    confusion_matrix, precision_recall_curve, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier

SEED = 42


def prepare_features(frame, numerical, categorical) -> pd.DataFrame:
    """Make category handling consistent between training and inference."""
    x = frame[numerical + categorical].copy()
    for col in numerical:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    for col in categorical:
        x[col] = x[col].astype("object").where(x[col].notna(), "__MISSING__").astype(str)
    return x


def make_ensemble(numerical, categorical, components: int = 100):

    """
    This is the SVM portion of the ensemble. 
    Components:
        ColumnTransformer: This function is used to apply different preprocessing steps to specific columns of the input data. In this case, it applies a pipeline to the numerical columns.
            "num": This is the name of the step. It is used to identify the step and to tune hyperparameters later.
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler()): This creates a pipeline that first imputes missing values in the numerical columns using the median value, and then standardizes the data by removing the mean and scaling to unit variance.
            numerical: Tells the ColumnTransformer which columns to apply the transforms to. 
        RBFSampler: This is a kernel approximation that approximates the feature map of RBF using random Fourier features.
            gamma: Controls the RBF kernel shape. 
            n_components: The dimensionality of the Monte Carlo feature space. 
            random_state: Controls the psuedorandom number generator.
        SGDClassifier: Linear classification estimator that uses SGD. 
    """
    svm_features = make_pipeline(
            ColumnTransformer([
                ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numerical)
            ], remainder="drop"), 
            RBFSampler(gamma=1.0 / len(numerical), n_components=components, random_state=SEED),
            SGDClassifier(
                loss="hinge", alpha=1e-4, class_weight="balanced",
                max_iter=1000, tol=1e-3, random_state=SEED,
            ),
        )
    
    """
    CalibratedClassifierCV: Calibrates a classifier's predictred probabilities using cross-validation.
        estimator: The base classifier that is to be calibrated. In this model, it uses svm_features.
        method: The calibration algorithm to be used. Isotonic will be used because the sample size is very large.
        cv: The number of folds to use for cross-validation.
        ensemble=False: This means that the calibration will be done on the single estomator, rather than on cv estimators.
        The base model is trained on 100% of the training data. 
    """
    svm = CalibratedClassifierCV(
            estimator=svm_features, method="isotonic", cv=3, ensemble=False,
        )
    
    """
    OneHotEncoder: Converts categorical features into a binary one-hot numeric array. 
        handle_unknown="infrequent_if_exist": Determines behavior when encountering unseen categories. 
        min_frequency=20: Groups rare categories into a single infrequent category to prevent high dimensionality.
    """
    all_features = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), numerical),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), categorical),
        ])

    """
    XGBClassifier: Builds an ensemble of decision trees sequentially, where each new tree corrects errors made by previous ones using gradient descent. 
        n_estimators: Number of boosting rounds (decision trees) to be built.
        max_depth: Maximum depth of each decision tree.
        learning_rate: Step size shrinkage used to prevent overfitting.
        subsample: Fraction of training rows randomly sampled per tree.
        colsample_bytree: Fraction of ffeatures randomly sampled when building each tree.
        min_child_weight: Controls the minimum sum of instance weights needed in a child for a split to occur.
        reg_lambda: l2 regularization hyperparameter (Ridge regularization). Penalizes large leaf weights. 5 shrinks leaf weights close to zero.
        objective: Defines the loss function to be minimized.
        eval_metric: Metric used to for model evaluation.
        tree_method:"hist": Uses a historgram-based approximate algorithm. This buckets continuous features into bins.
        n_jobs: Number of parallel threads used to run xgboost. -1 uses all available threads.
    """
    xgb = make_pipeline(
        all_features,
        XGBClassifier(
            n_estimators=350, max_depth=4, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            min_child_weight=5, reg_lambda=5.0,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", n_jobs=-1, random_state=SEED,
        ),
    )

    """
    MLPClassifier: Multi-layer perceptron classifier. This is the FC layer of the ensemble.
        hidden_layer_sizes: The FC layer is 2x4x2x1.
        activation: Activation function for the hidden layers. 
        solver: Weight optimization algorithm. Adam is a stochastic gradient-based optimizer.
        alpha: L2 penalty (regularization term) parameter.
        max_iter: MAximum number of training epochs. 
        early_stopping: Stops training when validation score stops improving.
        batch_size: Number of samples per gradient update.
        learning_rate_init: Initial learning rate used.
        n_iter_no_change: This is the number of consecutive eopochs without improvement to wait before stopping early.
    """
    classifier = MLPClassifier(
        hidden_layer_sizes=(4, 2), activation="tanh", solver="adam",
        alpha=1e-3, max_iter=500, early_stopping=False,
        batch_size=512, learning_rate_init=0.003,
        n_iter_no_change=35, random_state=SEED,
    )

    """
    StackingClassifier: Combines multiple distinct classification models (base estimators) and uses a meta-classifier (final) to learn
    how to best combine their predictions.
        estimators: List of tuples defining the base estimators.
        final_estimator: The meta-classifier trained to combine base predictions. 
            StandardScalar: Standardizes features by removing the mean and scaling to unit variance.
        stack_method: Controls the predction method. predict_proba returns class probabilities for each base estimator.
        cv: Number of folds for cross-validation.
        passthrough=False: The final estimator will only be trained on the predictions generated by the base estimators.
    """
    return StackingClassifier(
        estimators=[("svm", svm), ("xgb", xgb)],
        final_estimator=make_pipeline(StandardScaler(), classifier),
        stack_method="predict_proba", cv=3, passthrough=False, n_jobs=1,
    )


def to_logit(probabilities):
    """
    Converts an array of probabilities into logits, and formats the output as a single column 2D matrix. 
    """
    p = np.clip(np.asarray(probabilities), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)

def calibrated_predict(ensemble, calibrator, x):
    """
    Computes the uncalibrated prediction then returns the calibrated prediction using the provided calibrator.
    """
    uncalibrated = ensemble.predict_proba(x)[:, 1]
    return calibrator.predict_proba(to_logit(uncalibrated))[:, 1]

 