'''
    <START SET UP>
    Suppress warnings and import necessary libraries.
    Import code for loading data and extracting features.
'''

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import numpy as np
import math 
import get_images
import get_landmarks
from pathlib import Path 
## import main_KFOLD
import pandas as pd
from numpy.linalg import norm #to check why i cant calc the euclidean distance 

# k-Nearest Neighbors - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.neighbors import SVC

# Naive Bayes - https://scikit-learn.org/stable/modules/naive_bayes.html#
# from sklearn.naive_bayes import MultinomialNB

# Additional classifiers:
# Support Vector Machine - https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# Neural Network - https://scikit-learn.org/stable/modules/neural_networks_supervised.html#multi-layer-perceptron

'''
    <END SET UP>
'''

'''
    Load facial landmarks (5 or 68)
'''
data_set = Path("C:/Users/Ethan/Documents/GitHub/Face-and-Keystroke-Authentication/project_data/project_data")

Ximages, Ylabels, imageResHeight, imageRedWidth = get_images.get_images(data_set)
Ximages_array = np.array(Ximages)
num_images = Ximages_array.shape[0]
print(num_images)


# Specify the number of coordinates for facial landmarks (5 or 68)
num_coords = 68  # or 68 if you need

# Call the get_landmarks function to extract landmarks
landmarks, new_labels = get_landmarks.get_landmarks(Ximages, Ylabels, num_coords=num_coords)

'''
    Transform landmarks into features
'''


features = []
for k in range(len(Ylabels)):
    person_k = Ximages[k]
    features_k = []
    
    for i in range(person_k.shape[0]):
        for j in range(person_k.shape[1]):
            p1 = person_k[i,:]
            p2 = person_k[j,:]
            distance = norm(p1 - p2)  # Calculate Euclidean distance using numpy
            features_k.append(distance)
            #features_k.append( math.sqrt( (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 ) )
    features.append(features_k)
features = np.array(features)

#



''' 
    Create an instance of the classifier
'''


clf = KNeighborsClassifier()

# Perform K-Fold Cross-Validation
kf_splits = main_KFOLD.k_fold_cross_validation(features, Ylabels, n_splits=5, shuffle=True, random_state=42)

# Initialize a list to store the cross-validation scores
gen_scores = []
imp_scores = []


# Iterate over each fold and perform training and testing
for train_index, test_index in kf_splits:
    print("Ylabels shape:", Ylabels.shape)
    print("train_index:", train_index)
    print("test_index:", test_index)
    
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = Ylabels[train_index], Ylabels[test_index]
   
    clf.fit(X_train, y_train)
    matching_scores = clf.predict_proba(X_test)
    
    classes = clf.classes_
    matching_scores = pd.DataFrame(matching_scores, columns=classes)

    for i in range(len(y_test)):    
        scores = matching_scores.loc[i]
        mask = scores.index.isin([y_test[i]])
        gen_scores.extend(scores[mask])
        imp_scores.extend(scores[~mask])



#Put the resolution in the beginning
#TARGET VALUE 
#ASSIGN QUALITY SCORE TO IMAGE ARRAY 
#ADD QUALITY SCORE TO FEATURE ARRAY 
#Ylables is y 
#save resolution to list -> 
#set k to 2 
#use kfold 