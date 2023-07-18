#libraries
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
%matplotlib inline

#read input data
df = pd.read_csv('LogisticsRegression_HrData/HR_comma_sep.csv')
df.head()

#ret data types, non-Null values count
df.info()

#ret summary of various statistic count, mean, standard deviation, minimum and maximum values 
# and the quantiles of the data
df.describe()

df.corr()

df.left.value_counts()

df.groupby('left').mean()


#Satisfaction Level: Satisfaction level seems to be relatively low (0.44) compared to the retained ones (0.66)
#Average Monthly Hours: Average monthly hours are higher in employees leaving the firm (199 vs 207)
#Promotion Last 5 Years: Employees who are given promotion are likely to be retained at firm


pd.crosstab(df.salary,df.left).plot(kind='bar')


pd.crosstab(df.Department,df.left).plot(kind='bar')




subdf = df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
subdf.head()
salary_dummies = pd.get_dummies(subdf.salary, prefix="salary")
df_with_dummies = pd.concat([subdf,salary_dummies],axis='columns')
df_with_dummies.drop('salary',axis='columns',inplace=True)
df_with_dummies.head()
X = df_with_dummies
X.head()
y = df.left

##Different classier/Regression models

def learningModel(model,X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.accuracy ##Accuracy
    


X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.3)

LogReg_Accuracy = learningModel(LogisticRegression(),X_train, X_test, y_train, y_test)
LinReg_Accuracy = learningModel(LinearRegression(),X_train, X_test, y_train, y_test)
RanForest_Accuracy = learningModel(RandomForestClassifier(n_estimators=25),X_train, X_test, y_train, y_test)