# Classification Project: Ozone day or non-ozone Day

# Load libraries
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from pickle import dump
from pickle import load

#Set the frequency domain the all files will be
timeLength = 5 #Seconds
frameRate = 1/44100
fLow = 1/timeLength
##fHigh = 1/(2*frameRate)
fHigh = 4000
fZero = []
i = fLow

while i <= fHigh:
    fZero.append(i)   
    i += fLow
print(fZero)
fZero.append("voiceState")
# Load dataset
url = 'data2.csv'
##names = ["Subject","Location" ,"0.2" ,"0.4" ,"0.6" ,"0.8" ,"1" ,"1.2" ,"1.4" ,"1.6" ,"1.8" ,"2" ,"2.2" ,"2.4" ,"2.6" ,"2.8" ,"3" ,"3.2" ,"3.4" ,"3.6" ,"3.8" ,"4" ,"4.2" ,"4.4" ,"4.6" ,"4.8" ,"5" ,"5.2" ,"5.4" ,"5.6" ,"5.8" ,"6" ,"6.2" ,"6.4" ,"6.6" ,"6.8" ,"7" ,"7.2" ,"7.4" ,"7.6" ,"7.8" ,"8" ,"8.2" ,"8.4" ,"8.6" ,"8.8" ,"9" ,"9.2" ,"9.4" ,"9.6" ,"9.8" ,"10" ,"10.2" ,"10.4" ,"10.6" ,"10.8" ,"11" ,"11.2" ,"11.4" ,"11.6" ,"11.8" ,"12" ,"12.2" ,"12.4" ,"Condition"]
##namesA = [
dataset = read_csv(url, header=None, names=fZero, dtype="float64")
##dataset = dataset.astype('float64') 
#descriptive stats
#shape
#type of data for each columne
set_option('display.max_rows', 500)
print(dataset.dtypes)
#print "header" (fist 20 rows)
set_option('display.width', 100)
print(dataset.head(20)) #we can see that we may have to normalize
##descriptions, change precision to 3 places
set_option('precision', 3)
print(dataset.describe())

print(dataset.groupby('voiceState').size())
# Prepare Data

# Split-out validation dataset
array = dataset.values
X = array[:,0:20000].astype(float)
Y = array[:,20000]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)


### Evaluate Algorithms
##
##
### Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'accuracy'
##
# Standardize the dataset
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LogisticRegression(solver='liblinear'))])))
pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA', LinearDiscriminantAnalysis())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC(gamma='auto'))])))
results = []
names = []
print(Y_validation)
for name, model in pipelines:
        
	kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
	model.fit(X_train, Y_train)
	predictions = model.predict(X_validation)
	print(predictions)
	print(confusion_matrix(Y_validation, predictions))
# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
##
##### feature extraction
####test = SelectKBest(score_func=f_classif, k=4)
####fit = test.fit(X, Y)
##### summarize scores
####set_printoptions(precision=3)
####print(fit.scores_)
####features = fit.transform(X)
##### summarize selected features
####print(features[0:5,:])
####
# Tune scaled KNN
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
neighbors = [1,3,5,7,9,11,13,15,17,19,21]
param_grid = dict(n_neighbors=neighbors)
model = KNeighborsClassifier()
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)

##filename = 'Tuned_KNN.sav'
##dump(grid_result, open(filename, 'wb'))

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
##
##
# Tune scaled SVM
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = dict(C=c_values, kernel=kernel_values)
model = SVC(gamma='auto')
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)

#you can save this model here
# save the model to disk
##filename = 'Tuned_SVM.sav'
##dump(grid_result, open(filename, 'wb'))

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
# ensembles
model = AdaBoostClassifier(n_estimators = 50)
adb_results = model.fit(X_train, Y_train)
##filename = 'Tuned_ADB.sav'
##dump(adb_results, open(filename, 'wb'))

cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
print(cv_results)

