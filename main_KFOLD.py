def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import numpy as np
import math 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier as ORC
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, LeaveOneOut, TimeSeriesSplit


def train_test_split_method(features, y, test_size=0.33, random_state=42):
    """
    Split the data into training and testing sets using train_test_split.
    train_test_split: This technique randomly divides the dataset 
    into two subsets: one for training the model and the other 
    for testing its performance. It's commonly used for general 
    machine learning tasks where the temporal or sequential order of 
    data points is not critical. The training set is used to fit 
    the model's parameters, while the test set evaluates its 
    performance on unseen data. By default, it ensures a balanced d
    istribution of classes between the training and testing sets.

    Args:
    - features: Input features
    - y: Target variable
    - test_size: The proportion of the dataset to include in the test split
    - random_state: Controls the randomness of the split

    Returns:
    - X_train: Training features
    - X_test: Testing features
    - y_train: Training target variable
    - y_test: Testing target variable
    """
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def k_fold_cross_validation(features, y, n_splits=5, shuffle=False, random_state=42):
    """
    Perform K-Fold Cross-Validation.
    K-Fold Cross-Validation: This technique involves splitting the 
    dataset into k folds and training the model k times, each time 
    using a different fold as the test set and the remaining folds 
    as the training set. This helps in utilizing the entire dataset 
    for both training and testing, providing a more robust estimate 
    of model performance.

    Args:
    - features: Input features
    - y: Target variable
    - n_splits: Number of folds (default is 5)
    - shuffle: Whether to shuffle the data before splitting
    - random_state: Controls the randomness of the split

    Returns:
    - splits: Generator that yields the train/test indices for each fold
    """
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    return kf.split(features, y)

def stratified_k_fold_cross_validation(features, y, n_splits=5, shuffle=False, random_state=42):
    """
    Perform Stratified K-Fold Cross-Validation.
    Stratified K-Fold Cross-Validation: Similar to K-Fold 
    Cross-Validation, but ensures that each fold contains 
    approximately the same proportion of the target classes 
    as the original dataset. This is particularly useful for 
    imbalanced datasets where one class is much more prevalent 
    than the others.

    Args:
    - features: Input features
    - y: Target variable
    - n_splits: Number of folds (default is 5)
    - shuffle: Whether to shuffle the data before splitting
    - random_state: Controls the randomness of the split

    Returns:
    - splits: Generator that yields the train/test indices for each fold
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    return skf.split(features, y)

def leave_one_out_cross_validation(features, y):
    """
    Perform Leave-One-Out Cross-Validation.
    Leave-One-Out Cross-Validation (LOOCV): In this method, a 
    single observation is used as the test set while the rest of 
    the data is used for training. This process is repeated for 
    each observation in the dataset. LOOCV provides a good estimate 
    of model performance but can be computationally expensive, 
    especially for large datasets.

    Args:
    - features: Input features
    - y: Target variable

    Returns:
    - splits: Generator that yields the train/test indices for each fold
    """
    loo = LeaveOneOut()
    return loo.split(features, y)

def time_series_split(features, y, n_splits=5):
    """
    Perform Time Series Split.
    Time Series Split: This technique is suitable for time-series 
    data where the order of observations matters. It ensures that 
    the training set contains data from earlier time periods than the 
    test set, preventing the model from learning from future data. 
    This is crucial for forecasting tasks.

    Args:
    - features: Input features
    - y: Target variable
    - n_splits: Number of splits (default is 5)

    Returns:
    - splits: Generator that yields the train/test indices for each split
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return tscv.split(features, y)


'''
    Load facial landmarks (5 or 68)
'''

X = np.load("X-68-Caltech.npy")
y = np.load("y-68-Caltech.npy")
num_identities = y.shape[0]

'''
    Transform landmarks into features
'''

features = []
for k in range(num_identities):
    person_k = X[k]
    features_k = []
    for i in range(person_k.shape[0]):
        for j in range(person_k.shape[0]):
            p1 = person_k[i,:]
            p2 = person_k[j,:]      
            features_k.append( math.sqrt( (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 ) )
    features.append(features_k)
features = np.array(features)

''' 
    Create an instance of the classifier
'''

clf = ORC(KNeighborsClassifier())

# Perform K-Fold Cross-Validation
kf_splits = k_fold_cross_validation(features, y, n_splits=5, shuffle=True, random_state=42)

# Initialize a list to store the cross-validation scores
gen_scores = []
imp_scores = []


# Iterate over each fold and perform training and testing
for train_index, test_index in kf_splits:
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    clf.fit(X_train, y_train)
    matching_scores = clf.predict_proba(X_test)
    
    classes = clf.classes_
    matching_scores = pd.DataFrame(matching_scores, columns=classes)

    for i in range(len(y_test)):    
        scores = matching_scores.loc[i]
        mask = scores.index.isin([y_test[i]])
        gen_scores.extend(scores[mask])
        imp_scores.extend(scores[~mask])

