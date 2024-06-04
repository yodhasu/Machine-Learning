import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as tts
import seaborn as sb
import warnings
# from sklearn.utils.tests import ignore_warnings
# from sklearn.exceptions import ConvergenceWarning


# ML model function

def DT(Xtrain, Ytrain, Xtest):
    dt = DecisionTreeClassifier()
    dt_fit = dt.fit(Xtrain, Ytrain)
    
    dt_pred = dt_fit.predict(Xtest)
    return dt_pred
    # print(metrics.classification_report(Ytest, dt_pred))

# @ignore_warnings(category=ConvergenceWarning)
def LGR(Xtrain, Ytrain, Xtest):
    warnings.filterwarnings("ignore")
    lgr = LogisticRegression()
    lgr_fit = lgr.fit(Xtrain, Ytrain)
    
    lgr_pred = lgr_fit.predict(Xtest)
    return lgr_pred
    # print(metrics.classification_report(Ytest, lgr_pred))
    
def class_rep(target, yhat):
    print(metrics.classification_report(target, yhat))

# main method

# file processing
df = pd.read_csv('AIDS_Classification.csv')

# features and target selection
feats = df[['time', 'age', 'wtkg', 'hemo', 'homo', 'drugs', 'karnof', 'oprior', 'z30', 'race', 'symptom', 'preanti', 'cd40', 'cd420', 'cd80', 'cd820']]
target = df['infected']

Xtrain, Xtest, Ytrain, Ytest = tts(feats, target, test_size= 0.3, random_state=24)

#evaluation
class_rep(Ytest, DT(Xtrain, Ytrain, Xtest))
class_rep(Ytest, LGR(Xtrain, Ytrain, Xtest))

# random sample test

rand_samp = {'time': [1069, 300], 'age': [24, 49], 'wtkg' : [45.78, 79.81], 'hemo': [0, 1], 'homo': [1, 1], 'drugs': [1, 1], 'karnof': [69, 23], 'oprior': [0, 1], 'z30': [0, 1], 'race': [0, 1], 'symptom': [1, 0], 'preanti': [1120, 935], 'cd40': [169, 213], 'cd420': [420, 240], 'cd80': [75, 100], 'cd820': [169, 69]}

rand_df = pd.DataFrame(rand_samp, index=None)

dt_real = DT(feats, target, rand_df)
lgr_real = LGR(feats, target, rand_df)
print("Test on random sample")
print("="*100)

print(rand_df)
print("")
print(f"Decission Tree prediction on random sample: {[x for x in dt_real]}")
print(f"Logistic Regression prediction on random sample: {[x for x in lgr_real]}")