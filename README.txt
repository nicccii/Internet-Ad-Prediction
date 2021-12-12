To run B8IT108_CA.ipynb python code:

1. To run the code with the dataset undersampled:

a). Go to section labelled as 'Random Undersampling'

b). uncomment this code:

# from collections import Counter
# from imblearn.under_sampling import RandomUnderSampler
# # summarize class distribution
# print(Counter(y))
# # define undersample strategy
# #undersample = RandomUnderSampler(sampling_strategy='majority')
# undersample = RandomUnderSampler(sampling_strategy=0.5)
# # fit and apply the transform
# X, y = undersample.fit_resample(X, y)
# # summarize class distribution
# print(Counter(y))


c).Then ensure this code for oversampling in section labelled as 'Oversampling with SMOTE'

d). comment this code

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot
from numpy import where

# summarize class distribution after smote
counter = Counter(y)
print(counter)
# transform the dataset
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
# summarize the new class distribution
counter = Counter(y)
print(counter)
# scatter plot of examples by class label
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()

###############################################################################################################

2. To run the code with the datset oversampled:

a). Go to section commented as 'Oversampling with SMOTE'

b). uncomment this code:

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot
from numpy import where

# summarize class distribution after smote
counter = Counter(y)
print(counter)
# transform the dataset
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
# summarize the new class distribution
counter = Counter(y)
print(counter)
# scatter plot of examples by class label
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()

c).Then ensure this code for undersampling in section labelled as 'Random Undersampling'

d). comment this code

# from collections import Counter
# from imblearn.under_sampling import RandomUnderSampler
# # summarize class distribution
# print(Counter(y))
# # define undersample strategy
# #undersample = RandomUnderSampler(sampling_strategy='majority')
# undersample = RandomUnderSampler(sampling_strategy=0.5)
# # fit and apply the transform
# X, y = undersample.fit_resample(X, y)
# # summarize class distribution
# print(Counter(y))

##################################################################################################################
3. To run the code with features of the dataset reduced:

a). Go to section labelled 'Feature reduction'

b).uncomment this code and chnage the value of k in the code as desired

# from pandas import read_csv
# from numpy import set_printoptions
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import f_classif

# names = dataset.columns
# #dataframe = read_csv(url, names=names)
# # array = dataset.values
# # feature extraction
# test = SelectKBest(score_func=f_classif, k=10)
# fit = test.fit(X, y)
# # summarize scores
# set_printoptions(precision=3)
# print(fit.scores_)
# features = fit.transform(X)
# X=features
# # summarize selected features
# print(features[0:5,:])

###################################################################################################################
4. To run the code with all the features of the dataset:

a). Go to section labelled 'Feature reduction'

b).Ensure this code is commented

# from pandas import read_csv
# from numpy import set_printoptions
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import f_classif

# names = dataset.columns
# #dataframe = read_csv(url, names=names)
# # array = dataset.values
# # feature extraction
# test = SelectKBest(score_func=f_classif, k=10)
# fit = test.fit(X, y)
# # summarize scores
# set_printoptions(precision=3)
# print(fit.scores_)
# features = fit.transform(X)
# X=features
# # summarize selected features
# print(features[0:5,:])

###################################################################################################################

5. To run the code with with each of the algorithms implemented in the code:

a). Go to section labelled 'Training the Training set'

b). To implement K-NN, uncomment this code and comment out other algorithms:

# from sklearn.neighbors import KNeighborsClassifier
# classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
# classifier.fit(X_train, y_train)

c). This SVM code is not commented as it produced the best performance. To implement SVM, ensure to uncomment this code and comment out other algorithms:

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

d). To implement Kernel SVM, uncomment this code and comment out other algorithms:

# from sklearn.svm import SVC
# classifier = SVC(kernel = 'rbf', random_state = 0)
# classifier.fit(X_train, y_train)

e). To implement Naive Bayes , uncomment this code and comment out other algorithms:

# from sklearn.svm import SVC
# classifier = SVC(kernel = 'rbf', random_state = 0)
# classifier.fit(X_train, y_train)

f). To implement Decision Tree model , uncomment this code and comment out other algorithms:

# from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
# classifier.fit(X_train, y_train)

g). To implement Random Forest model , uncomment this code and comment out other algorithms:

# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
# classifier.fit(X_train, y_train)

h). To implement XGBoost model , uncomment this code and comment out other algorithms:

# from xgboost import XGBClassifier
# classifier = XGBClassifier()
# classifier.fit(X_train, y_train))

#####################################################################################################################

6. Go to section labelled 'Applying k-Fold Cross Validation':

a). comment out the code if this part of the code is slow (this is not necessary if the code is fast to execute)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))