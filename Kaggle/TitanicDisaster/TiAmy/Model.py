# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

#load packages
import sys #access to system parameters https://docs.python.org/3/library/sys.html
print("Python version: {}". format(sys.version))

import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features
print("pandas version: {}". format(pd.__version__))

import matplotlib #collection of functions for scientific and publication-ready visualization
print("matplotlib version: {}". format(matplotlib.__version__))

import numpy as np #foundational package for scientific computing
print("NumPy version: {}". format(np.__version__))

import scipy as sp #collection of functions for scientific computing and advance mathematics
print("SciPy version: {}". format(sp.__version__))

import IPython
from IPython import display #pretty printing of dataframes in Jupyter notebook
print("IPython version: {}". format(IPython.__version__))

import sklearn #collection of machine learning algorithms
print("scikit-learn version: {}". format(sklearn.__version__))

#misc libraries
import random
import time

#ignore warnings
import warnings
#warnings.filterwarnings('ignore')
warnings.simplefilter("ignore")
print('-'*25)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter)
# will list the files in the input directory

from subprocess import check_output
# unix系统中每一个进程会返回一个状态码 默认0为进程执行正常,其他为错误码
print(check_output(["ls", "../data"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis,gaussian_process
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.plotting import scatter_matrix

#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8
# 这个加在这里就不会出现那么多烦人的warning了
warnings.simplefilter("ignore")

def load_data():
    data_raw = pd.read_csv('../data/train.csv')
    X_val  = pd.read_csv('../data/test.csv')
    X_train = data_raw.copy(deep = True)
    data_cleaner = [X_train, X_val]
    return data_cleaner

def show_data_na(data_cleaner):
    X_train,X_val = data_cleaner[0],data_cleaner[1]
    print('Train columns with null values:\n', X_train.isnull().sum())
    print("-"*10)

    print('Test/Validation columns with null values:\n', X_val.isnull().sum())
    print("-"*10)

def show_data_na(data_cleaner):
    X_train,X_val = data_cleaner[0],data_cleaner[1]
    print('Train columns with null values:\n', X_train.isnull().sum())
    print("-"*10)

    print('Test/Validation columns with null values:\n', X_val.isnull().sum())
    print("-"*10)

def show_data_info(data_cleaner):
    X_train,X_val = data_cleaner[0],data_cleaner[1]
    print('Train columns info:\n', X_train.info())
    print("-"*10)
    print('Test/Validation info:\n', X_val.info())
    print("-"*10)

def show_data_describe_column(data_cleaner,column):
    X_train,X_val = data_cleaner[0],data_cleaner[1]
    print('Train columns describe:\n', X_train[column].describe())
    print('Test/Validation value counts:\n', X_train[column].value_counts())
    print("-"*10)
    print('Test/Validation describe:\n', X_val[column].describe())
    print('Test/Validation value counts:\n', X_val[column].value_counts())
    print("-"*10)
#show_data_info_column('Title')

def show_data_value_counts_column(data_cleaner,column):
    X_train,X_val = data_cleaner[0],data_cleaner[1]
    print('Test/Validation value counts:\n', X_train[column].value_counts())
    print("-"*10)
    print('Test/Validation value counts:\n', X_val[column].value_counts())
    print("-"*10)
#show_data_info_column('Title')


# 可看出Embarked缺失只有两个值，而且在训练集中，这里我们选择直接删掉.
def completing_embarked_na(data_cleaner):
    for X in data_cleaner:
        X.dropna(
            axis = 0
            , subset = ['Embarked']
            , inplace = True
        )
        X.reset_index()
    #X_train.drop_
#print('before : ' , X_train.shape)
#completing_embarked_na()
#print('after : ' , X_train.shape)

# 只有验证集中有Fare的丢失，这里填充平均值
def completing_fare_na(data_cleaner):
    for X in data_cleaner:
        X['Fare'].fillna(X['Fare'].median(), inplace = True)
#print('before : ' , X_val.shape)
#completing_fare_na()
#print('after : ' , X_val.shape)

# 这几个特征我们直接删掉 PassengerID，Ticket,这几个看起来用处不大，Cabin缺失值过多。
def drop_useless(data_cleaner,drop_column= ['PassengerID','Ticket','Cabin']):
    for X in data_cleaner:
        X.drop(columns=drop_column, axis=1, inplace = True)
        X.reset_index()
#drop_useless()
#show_data_info()

# 家庭大小
def create_familysize(data_cleaner):
    for X in data_cleaner:
        X['FamilySize'] = X ['SibSp'] + X['Parch'] + 1
#create_familysize()

# 头衔
def create_title(data_cleaner):
    for X in data_cleaner:
        X['Title'] = X['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
        stat_min = 10
        title_names = (X['Title'].value_counts() < stat_min)
        X['Title'] = X['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
        X.reset_index()
#create_title()
#show_data_info()

# 是否独身一人
def create_isAlone(data_cleaner):
    for X in data_cleaner:
        X['IsAlone'] = 1
        X['IsAlone'].loc[X['FamilySize'] > 1] = 0

# 票价分布
def create_FareBin(data_cleaner,bins = 4):
    for X in data_cleaner:
        X['FareBin'] = pd.qcut(X['Fare'], bins)

# 年龄
def create_AgeBin(data_cleaner,bins = 5):
    for X in data_cleaner:
        X['AgeBin'] = pd.cut(X['Age'].astype(int), bins)

# 随机森林预测Age缺失值.
def set_missing_ages(df,columns = ['Age','Fare', 'FamilySize', 'Pclass']):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[columns]
    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df['Age'].notna()].as_matrix()
    unknown_age = age_df[age_df['Age'].isna()].as_matrix()
    # y即目标年龄
    y = known_age[:, 0]
    # X即特征属性值
    X = known_age[:, 1:]
    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(n_estimators=2000, n_jobs=-1).fit(X,y)
    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1:])
    # 用得到的预测结果填补原缺失数据
    df.loc[df['Age'].isna(), 'Age' ] = predictedAges
    return df, rfr

# 填充age的空值.
def completing_age_na(data_cleaner,fill,columns=['Age','Fare', 'FamilySize', 'Pclass']):
    for X in data_cleaner:
        if fill == 0 :
            # 均值填充
            X['Age'].fillna(X['Age'].median(), inplace = True)
        elif fill == 1:
            # 随机森林预测缺失值.
            set_missing_ages(X,columns = columns)
        elif fill == 2:
            # Todo 分为离散数据,有年龄，没年龄.
            raise Exception('填充方式错')
        else :
            raise Exception('填充方式错')

def convert_X(data_cleaner):
    label = LabelEncoder()
    for X in data_cleaner:
        X['Sex_Code'] = label.fit_transform(X['Sex'])
        X['Embarked_Code'] = label.fit_transform(X['Embarked'])
        X['Title_Code'] = label.fit_transform(X['Title'])
        X['AgeBin_Code'] = label.fit_transform(X['AgeBin'])
        X['FareBin_Code'] = label.fit_transform(X['FareBin'])


def get_clean_data():
    data_cleaner = load_data()
    # 填充上船港口
    completing_embarked_na(data_cleaner)
    # 填充票价
    completing_fare_na(data_cleaner)
    # family大小
    create_familysize(data_cleaner)
    # 是否独身
    create_isAlone(data_cleaner)
    # 票价分层
    create_FareBin(data_cleaner)
    # 处理名字,转化为Title
    create_title(data_cleaner)
    # 完善Age,随机森林，
    completing_age_na(data_cleaner, 1, columns=['Age', 'Pclass', 'SibSp', 'Parch', 'Fare'])
    # 年龄分层
    create_AgeBin(data_cleaner)
    # 丢弃无用的列
    drop_useless(data_cleaner, ['PassengerId', 'Cabin', 'Ticket', 'Name'])
    # 离散化数据
    convert_X(data_cleaner)
    X_train, X_val = data_cleaner[0], data_cleaner[1]
    # 训练集分类.
    y = ['Survived']
    X_x = [
        'Sex', 'Pclass',
        'Embarked', 'Title',
        'SibSp', 'Parch',
        'Age', 'Fare',
        'FamilySize', 'IsAlone'
    ]
    # pretty name/values for charts
    # data1_x_calc = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'Age', 'Fare']
    # coded for algorithm calculation
    X_xy = y + X_x

    # define x variables for original w/bin features to remove continuous variables
    X_x_bin = [
        'Sex_Code', 'Pclass',
        'Embarked_Code', 'Title_Code',
        'FamilySize', 'AgeBin_Code',
        'FareBin_Code'
    ]
    X_xy_bin = y + X_x_bin

    # define x and y variables for dummy features original
    X_dummy_train = pd.get_dummies(X_train[X_x])
    X_dummy_val = pd.get_dummies(X_val[X_x])
    X_x_dummy = X_dummy_train.columns.tolist()

    return X_xy, X_xy_bin, (X_dummy_train, X_dummy_val, X_x_dummy), data_cleaner

X_xy, X_xy_bin, dummy_dataset, dataset = get_clean_data()

GridSearchCV()
