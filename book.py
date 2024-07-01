import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler


def load_application_train():
    data = pd.read_csv("application_train.csv")
    return data

df = load_application_train()
print(df.shape)

def load():
    data = pd.read_csv("titanic.csv")
    return data

df = load()
print(df.shape)

#We will try to detect outliers in a numerical 'Age' column by using boxplot.

sns.boxplot(x=df["Age"])
plt.show()

#need to find Q1(25th percentile) Q3(75th percentile), and then minimum and maximum values by using 1.5IQR.
q1 = df["Age"].quantile(0.25)
q3 = df["Age"].quantile(0.75)
iqr = q3 - q1
up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr

print(df[(df["Age"] < low) | (df["Age"] > up)]) 

#index of outliers:
print(df[(df["Age"] < low) | (df["Age"] > up)].index) 

#checking outlier with any() function
print(df[(df["Age"] < low) | (df["Age"] > up)].any(axis=None)) 

#It returns true because there are some rows that lie beyond the
#thresholds we specified. We used a parameter called axis=None since we don't 
#care rows or columns. 

print(df[(df["Age"] < low)].any(axis=None))

#NOW WITH A GENRALIZED FUNCTION:
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

print(outlier_thresholds(df,"Age"))

#age is below -6.68 or higher 64.81, we can say that it is an outlier


low, up = outlier_thresholds(df, "Fare")
print(df[(df["Fare"] < low) | (df["Fare"] > up)].head())



#writing a function to check for outliers
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] <low_limit)].any(axis=None):
        return True
    else:
        return False

print(check_outlier(df, "Age"))  #there is 1 outlier 
print(check_outlier(df, "Fare")) #there is 1 outlier

def grab_col_names(dataframe, cat_th=10, car_th=20):
    '''
    Returns categorical columns list, numerical columns list and categorical but cardinal column list.

    Parameters
    ----------
    dataframe: dataframe
        main dataframe
    cat_th: int, float
        threshold for the number of unique variable of a column that seems numerical but actually categorical
    car_th: int, float
        threshold for the number of unique variable of a column that seems categorical but actually cardinal
    
    Returns
    -------
    cat_cols: list
        list of categorical columns
    num_cols: list
        list of numerical columns
    cat_but_car: list
        list of of cardinal columns
    
    Notes
    ------
    -> cat_cols + num_cols + cat_but_car = the number of columns of dataframe
    -> cat_cols includes num_but_cat
    -> Categorical variables with numerical appearance are also included in categorical variables.

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))
    '''


    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car




cat_cols, num_cols, cat_but_car = grab_col_names(df)


num_cols = [col for col in num_cols if col not in "PassengerId"]
print(num_cols)  



#We will check outliers in numerical columns.
for col in num_cols:
    print(col, check_outlier(df, col))


#Let's import application_train.csv
dff = load_application_train()

cat_cols, num_cols, cat_but_car = grab_col_names(dff)

#'SK_ID_CURR' is a numerical column, it is irrelevant so we will remove it from num_cols
num_cols.remove('SK_ID_CURR')

print()
print()

#Let's see which numerical columns have outliers.
for col in num_cols:
    print(col, check_outlier(dff, col))

#lets remove outlier from titanic data :

df=load()
low,up=outlier_thresholds(df, "Fare")
print(df.shape)

#There are 116 outliers for 'Fare' variable, therefore if we only remove Fare outliers,our new data will have (775,12) 
print(df[~((df["Fare"] < low) | (df["Fare"] > up))].shape)

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols.remove('PassengerId')

for col in num_cols:
    df = remove_outlier(df,col)

print(df.shape)

#reassgining the thrweshholds:
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

df= load()
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols.remove('PassengerId')

#let us see which column has outliers 

for col in num_cols:
   print(col,check_outlier(df,col))

#replace outliers with thresholds:

for col in num_cols:
    replace_with_thresholds(df,col)

#replacing this we shoudnt get any outliers:
for col in num_cols:
    print(col,check_outlier(df,col))
