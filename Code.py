import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("50_Startups.csv")
print(df.head(),end='\n\n')

print(df.describe(),end='\n\n')

feature_values = {col: df[col].nunique() for col in df.columns}
print("Feature Values: ",feature_values,end='\n\n') #removes all the 0 values; gives you the count of all the non-0 unique vals

categorical_features = [feature for feature in df.columns if df[feature].dtypes == 'O']
print("Categorical Features: ",categorical_features,end='\n\n')

font = {'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 16,
        }

#using Seaborn
sns.set(style="dark")
#sns.set_context("poster")
sns.cubehelix_palette(as_cmap=True)

plt.figure(figsize=(24,12))
total = float(len(df))
ax =sns.countplot(x="Profit", data=df)
plt.xticks(rotation=90)
plt.title("Plot for Profit", fontsize=20)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width() / 2.
    y = p.get_height()
    ax.annotate(percentage, (x, y),ha='center',va='bottom')
plt.show()


plt.figure(figsize=(24, 12))
fig = sns.boxplot(x='Administration', y="Profit", data=df.sort_values('Profit',ascending=False))
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(24, 12))
fig = sns.boxplot(x='Marketing Spend', y="Profit", data=df.sort_values('Profit',ascending=False))
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(24, 12))
fig = sns.boxplot(x='R&D Spend', y="Profit", data=df.sort_values('Profit',ascending=False))
plt.xticks(rotation=90)
plt.show()

numerical_features = [feature for feature in df.columns if ((df[feature].dtypes != 'O') & (feature not in ['Profit']))]
print('\nNumerical variables: ', numerical_features)

discrete_feature = [feature for feature in numerical_features if df[feature].nunique() < 25]
print("\nDiscrete Variables Count: {}".format(len(discrete_feature)))

continuous_features = [feature for feature in numerical_features if feature not in discrete_feature + ['Profit']]
(continuous_features)

font = {'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 16,
        }
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
fig.subplots_adjust(wspace=0.1, hspace=0.3)
fig.suptitle('Scatter Plot Of Continous Variables', fontdict=font)
fig.subplots_adjust(top=0.95)

axes = axes.ravel()

for i, col in enumerate(continuous_features):
    # using log transformation
    sns.distplot(df[col], ax=axes[i])

axes = axes.ravel()

for i, col in enumerate(continuous_features):
    x = df[col]
    y = df['Profit']
    sns.scatterplot(x, ax=axes[i])

fig, ax = plt.subplots(1, 3, figsize=(20, 5))
for variable, subplot in zip(numerical_features, ax.flatten()):
    sns.boxplot(df[variable], ax=subplot)

plt.figure(figsize=(10, 6))

#Reversing the Elements
corr_matrix = df.corr()
corr_matrix_reversed = corr_matrix.iloc[::-1, ::-1]

sns.heatmap(corr_matrix_reversed, annot=True, cmap='Blues')
plt.show()


#Model Selection and Splitting
"""MODEL SELECTION"""
X=df.iloc[:,:3]
y=df.iloc[:,3]

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.3, random_state=42)

feature_cols = [x for x in train.columns if x != 'Profit']
X_train = train[feature_cols]
y_train = train['Profit']

X_test  = test[feature_cols]
y_test  = test['Profit']

"Splitting The Data"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)


#1.Linear Regression


regressor=LinearRegression()
regressor.fit(X_train,y_train)
#y_pred=regressor.predict(X_test)

#Models for the Accuracy of the Data
scoretr=r2_score(y_train,regressor.predict(X_train))
scorets=r2_score(y_test,regressor.predict((X_test)))
msets=mean_squared_error(y_test,regressor.predict(X_test))
msetr=mean_squared_error(y_train,regressor.predict(X_train))
hd1='LINEAR REGRESSION'
hd2='LASSO REGRESSION'
hd3='QUANTILE REGRESSION'
hd4='ELASTIC NET REGRESSION'
hd5='POISSON REGRESSION'
hd6='COMPARING THE MODELS:'
ut1='\x1B[4m'+hd1+'\x1B[0m'
ut2='\x1B[4m'+hd2+'\x1B[0m'
ut3='\x1B[4m'+hd3+'\x1B[0m'
ut4='\x1B[4m'+hd4+'\x1B[0m'
ut5='\x1B[4m'+hd5+'\x1B[0m'
ut6='\x1B[4m'+hd6+'\x1B[0m'

print(ut1)
print('TRAIN:')
print('MSE : '+str(msetr))
print('RMSE : '+str(math.sqrt((msetr))))
print('R2 Score : '+str(scoretr))
print('\nTEST:')
print('MSE : '+str(msets))
print('RMSE : '+str(math.sqrt((msets))))
print('R2 Score : '+str(scorets))


#2.Lasso Regression

las=Lasso()
las.fit(X_train,y_train)
y_predi=las.predict(X_test)

scoreltr=r2_score(y_train,las.predict(X_train))
scorelts=r2_score(y_test,las.predict(X_test))
mselts=mean_squared_error(y_test,las.predict(X_test))
mseltr=mean_squared_error(y_train,las.predict(X_train))
print('\n')
print(ut2)
print('TRAIN:')
print('MSE : '+str(mseltr))
print('RMSE : '+str(math.sqrt((mseltr))))
print('R2 Score : '+str(scoreltr))
print('\nTEST:')
print('MSE : '+str(mselts))
print('RMSE : '+str(math.sqrt((mselts))))
print('R2 Score : '+str(scorelts))

#3.Quantile Regression

qr=QuantileRegressor()
qr.fit(X_train,y_train)
y_predic=qr.predict(X_test)

scoreqrtr=r2_score(y_train,qr.predict(X_train))
scoreqrts=r2_score(y_test,qr.predict(X_test))
mseqrts=mean_squared_error(y_test,qr.predict(X_test))
mseqrtr=mean_squared_error(y_train,qr.predict(X_train))
print('\n')
print(ut3)
print('TRAIN:')
print('MSE : '+str(mseqrtr))
print('RMSE : '+str(math.sqrt((mseqrtr))))
print('R2 Score : '+str(scoreqrtr))
print('\nTEST:')
print('MSE : '+str(mseqrts))
print('RMSE : '+str(math.sqrt((mseqrts))))
print('R2 Score : '+str(scoreqrts))
print('\n')


# 4.Elastic Net

from sklearn.linear_model import ElasticNet


qr=ElasticNet()
qr.fit(X_train,y_train)
y_predic=qr.predict(X_train)
scoreenr=r2_score(y_train,qr.predict(X_train))
scoreens=r2_score(y_test,qr.predict(X_test))
mseqrts=mean_squared_error(y_test,qr.predict(X_test))
mseqrtr=mean_squared_error(y_train,qr.predict(X_train))
print('\n')
print(ut4)
print('TRAIN:')
print('MSE : '+str(mseqrtr))
print('RMSE : '+str(math.sqrt((mseqrtr))))
print('R2 Score : '+str(scoreenr))
print('\nTEST:')
print('MSE : '+str(mseqrts))
print('RMSE : '+str(math.sqrt((mseqrts))))
print('R2 Score : '+str(scoreens))
print('\n')


#5.Poisson Regressor

from sklearn.linear_model import PoissonRegressor
pr=PoissonRegressor()
pr.fit(X_train,y_train)
y_predic=pr.predict(X_train)

scoreprtr=r2_score(y_train,pr.predict(X_train))
scoreprts=r2_score(y_test,pr.predict(X_test))
mseprts=mean_squared_error(y_test,qr.predict(X_test))
mseprtr=mean_squared_error(y_train,pr.predict(X_train))

print('\n')
print(ut5)
print('TRAIN:')
print('MSE : '+str(mseqrtr))
print('RMSE : '+str(math.sqrt((mseqrtr))))
print('R2 Score : '+str(scoreqrtr))
print('\nTEST:')
print('MSE : '+str(mseqrts))
print('RMSE : '+str(math.sqrt((mseqrts))))
print('R2 Score : '+str(scoreqrts))
print('\n')


#Tabular Columns

print('\n\n')
print(ut6)
from tabulate import tabulate
md=[['Linear Regression',msetr,math.sqrt(msetr),scoretr],['Lasso Regression',mseltr,math.sqrt(mseltr),scoreltr],['Quantile Reression',mseqrtr,math.sqrt(mseqrtr),scoreqrtr],['Elastic Net Regression',mseqrtr,math.sqrt(mseqrtr),scoreenr],['Poisson Regression',mseprtr,math.sqrt(mseprtr),scoreprtr]]
head=['Model Name','MSE','RMSE','R2']
print(tabulate(md,headers=head,tablefmt='grid'))