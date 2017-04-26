import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing,cross_validation, svm
from sklearn.linear_model import LinearRegression


#df = quandl.get("GOOG/NASDAQ_GOOGL", authtoken="sD9npaAKgpHsQdwVfZ5p")
df = quandl.get("WIKI/GOOGL", authtoken="sD9npaAKgpHsQdwVfZ5p")




############   SET UP THE DATA
df = df [ ['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',] ]
df[ 'HL_PCT' ] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df[ 'PCT_change' ] = (df['Adj. Close'] - df['Adj. Open'] ) / df['Adj. Open'] * 100.0
#define a new data frame
df = df[ [ 'Adj. Close' , 'HL_PCT' , 'PCT_change', 'Adj. Volume'  ] ]




########### FIX THE DATABASE
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace= True )
forecast_out = int ( math.ceil( 0.01 * len(df) ) )  #no. of days to forecast

print ( "no of days forecasted ", forecast_out )

df ['label'] = df[forecast_col].shift(-forecast_out)
#print (df.head() )
#print (df.tail() )
df.dropna( inplace = True )
#print ( df.tail() )

X = np.array( df.drop(['label'], 1) )
y = np.array ( df['label'] )
X = preprocessing.scale(X)
df.dropna(inplace = True )
y = np.array( df['label'] )



#############  USE BUILT-IN ML ALGORITHMS


X_train, X_test, y_train, y_test = cross_validation.train_test_split ( X, y, test_size = 0.2 )


#clf = svm.SVR();                        ## USING SVM
clf = LinearRegression(n_jobs = -1)               ##FOR LINEAR REGRESSION
clf.fit( X_train, y_train )
accuracy = clf.score(X_test, y_test )

print ( accuracy) 

