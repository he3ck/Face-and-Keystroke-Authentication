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
import main_KFOLD
# k-Nearest Neighbors - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
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


Ximages, Ylabels = get_images.get_images(data_set)
Ximages_array = np.array(Ximages)
num_identities = Ximages_array.shape[0]


# Specify the number of coordinates for facial landmarks (5 or 68)
num_coords = 68  # or 68 if you need

# Call the get_landmarks function to extract landmarks
landmarks, new_labels = get_landmarks.get_landmarks(Ximages, Ylabels, num_coords=num_coords)
#num_identities = Ximages.shape

'''
    Transform landmarks into features
'''

features = []
for k in range(num_identities):
    person_k = Ximages_array[k]
    features_k = []
    for i in range(person_k.shape[0]):
        for j in range(person_k.shape[0]):
            p1 = person_k[i,:]
            p2 = person_k[j,:]      
            #features_k.append( math.sqrt( (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 ) )
            distance = np.linalg.norm(p1 - p2)  # Compute Euclidean distance
            features_k.append(distance)  # Append the distance to the features list
    features.append(features_k)
features = np.array(features)



''' 
    Create an instance of the classifier
'''
model = NearestCentroid()
indexesForSplit = main_KFOLD.stratified_k_fold_cross_validation(features, Ylabels, 2, True, 42)

for i, (train_index, test_index) in enumerate(indexesForSplit):
    testSetFeatures = []
    trainSetFeatures = []
    trainSetTargets = []
    num_correct = 0
    num_incorrect = 0
    total_guesses = 0
###    print(f"Fold {i}")
###    print(f"    Train: index={train_index}")
###    print(f"    Test: index={test_index}")
    if i == 0:
        model = NearestCentroid()
    else:
        model = KNeighborsClassifier()
    for t in train_index:
        trainSetFeatures.append(features[t])
        trainSetTargets.append(Ylabels[t])
    for t in test_index:
        testSetFeatures.append(features[t])
    model.fit(trainSetFeatures, trainSetTargets)
    predictionResults = model.predict(testSetFeatures)
    for r in range(0, len(predictionResults)):
        if predictionResults[r] == Ylabels[test_index[r]]:
            num_correct += 1
        else:
            num_incorrect += 1
        total_guesses += 1
    percentageCorrect = num_correct*100.0/total_guesses
    percentageIncorrect = num_incorrect*100.0/total_guesses
    print(f"The percent correct is {round(percentageCorrect, 2)}%, the percent incorrect is {round(percentageIncorrect, 2)}%")

    


print("we got to this point")
