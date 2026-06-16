## CUSTOMER CHURN PREDICTION
# Churn = customer leaving a company exm. : netflix user stops subscription , telecom user switches SIM , bank cutomers closes accounts etc
# GOAL : to predict whether a cutomer will leave or not 
# INPUT : age , monthly charges , tenure(how long customer stayed) , services used , payment method 
# OUTPUT : 0 -> Not Leaving , 1-> Leaving
# REAL WORLD VALUE : companies used this to : identify risk customers , take actions (offers,dicounts) , increase revenue
# PIPELINE : load dataset-> data preprocessing-> spilt data->train model-> predict-> evaluate (accuracy,confusion matrix)->save model(model.pkl),streamlit app
# teleco customer churn
# import dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE #Synthetic Minority Over-sampling Technique
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import pickle
# data loading and understanding
df = pd.read_csv("churn.csv")
# print(df.shape) # (7043,21)
# print(df.head()) # prints first 5 rows
# pd.set_option("display.max_columns",None) # it prints all the 21 columns
# print(df.head())
# print(df.info())
df = df.drop(columns=["customerID"])
# print(df.head(2))
# print(df.columns)
# printing the unique values in all the columns
numerical_features_list = ["tenure","MonthlyCharges","TotalCharges"]
for col in df.columns :
    if col not in numerical_features_list :
        pass # print(col,df[col].unique())
        # print("-"*50)
# print(df.isnull().sum()) # this shows that are not any missing values in any column
# df["TotalCharges"] = df["TotalCharges"].astype(float) # but when we tried to convert the datatype of column = TotalCharges 
# initially it was of str datatype but we want to convert it into float datatype 
# it means that this column has missing values kept as " " which cannot be converted to float datatype
# print(len(df[df["TotalCharges"] == " "] )) # prints the no. of rows where the column total charges is empty
# these missing values can occur when the author is unable to collect the data but in some cases these missing values also depict an important
# information here it can mean that the customer not even compeleted a single month hence tenure(0) * monthly charge = 0 or " "
df["TotalCharges"] = df["TotalCharges"].replace({" " : "0.0"})
df["TotalCharges"] = df["TotalCharges"].astype(float)
# print(df.info())
# checking the class distribution of target column
# print(df["Churn"].value_counts()) # it simply returns the count of yes or no
# from the O/P yes = 1869 and no = 5174 so its clearly imbalanced so we have to perform oversampling or undersampling
# Exploratory Data Analysis (EDA)
# print(df.shape)
# print(df.columns)
# print(df.head(2))
# print(df.describe()) # calculate statistical measures and works only on numerical features not on categorical fetures
# Numerical feature analysis
# Understanding the distribution of the numerical features
def plot_histogram(df,column_name) :
    plt.figure(figsize=(5,3))
    sns.histplot(df[column_name],kde=True) # kde : kernel density estimatiom
    plt.title(f"Distribution of {column_name}")
    # calculate the mean and median values for the columns
    col_mean = df[column_name].mean()
    col_median  = df[column_name].median()
    # add vertical lines for mean and median
    plt.axvline(col_mean,color="red",linestyle="--",label="Mean")
    plt.axvline(col_median,color="green",linestyle=":",label="Median")
    plt.legend()
    plt.show()
# plot_histogram(df,"tenure")
# plot_histogram(df,"MonthlyCharges")
# plot_histogram(df,"TotalCharges")
# we cannot train model like regression(linear,logistic,lasso),SVM on dataset having non-uniform distribution 
# tree models can handle the non-uniform distribution
# but for models like regression,svm we have to perform feture scaling to get uniform distribution of the dataset in a bell shaped curve
# box plot for numerical features ; box plot helps to find the outliers
def plot_boxplot(df,column_name) :
    plt.figure(figsize=(5.0,3.0))
    sns.boxplot(y=df[column_name])
    plt.title(f"Box Plot of {column_name}")
    plt.ylabel(column_name)
    plt.show()
# plot_boxplot(df,"tenure")
# plot_boxplot(df,"MonthlyCharges")
# plot_boxplot(df,"TotalCharges")  
# Correlation Heatmap for Numerical features 
# correlation matrix - heatmap
plt.figure(figsize=(8,4))
sns.heatmap(df[["tenure","MonthlyCharges","TotalCharges"]].corr(),annot=True,cmap="coolwarm",fmt=".2f")
plt.title("Correlation Heatmap")
# plt.show()
# 1 means I positive correlation ,0 means I negative correlation ; when two columns are highly correlated we drop one of the columns as this
# would create multicolinearity issue also diagonal can be ignored because it is nothing but tenure against tenure
# categorical features - analysis
# for categorical features COUNT PLOT is plotted to show the distribution
object_cols = df.select_dtypes(include="str").columns.to_list()
object_cols = object_cols + ["SeniorCitizen"] # this column is included because it may contain 0 or 1 but it refers thatwhether the customer is young or not
# print(object_cols)
for col in object_cols:
    plt.figure(figsize=(5,3))
    sns.countplot(x=df[col])
    plt.title(f"Count Plot of {col}")
    # plt.show()
# if there is a imbalance in the target column it will create an issue not matter what model we are using
# Data Preprocessing
# label encoding of target column
df["Churn"] = df["Churn"].replace({"Yes" : 1,"No" : 0})
df["Churn"] = df["Churn"].astype(int)
# (df.head(3))
# label encoding of other categorical features
# identifying columns with str datatype
object_columns = df.select_dtypes(include="str").columns
# print(object_columns)
# initialise a dictionary to save these encoders
encoders = {}
# apply label encoding and store the encoders
for column in object_columns :
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])
    encoders[column] = label_encoder
# save the encoders to a pickle file
with open("encoders.pkl","wb") as f :
    pickle.dump(encoders,f)
# print(encoders)
# print(df.head()) # all categorical fetures are conerted to label encoders (0,1,2,...)
# Training and Testing data split
# splitting the features and target
X = df.drop(columns=["Churn"])
Y = df["Churn"]
# print(X)
# print(Y)
# split training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
# print(Y_train.shape) # (5634,)
# print(Y_train.value_counts()) # 0 : 4138 ,1:1496 creating imbalance in the target column
# SMOTE : synthetic minority oversampling technique
smote = SMOTE(random_state=42)
# SMOTE is done only on the training data 
X_train_smote , Y_train_smote = smote.fit_resample(X_train,Y_train)
# print(Y_train_smote.shape) # (8276,)
# print(Y_train_smote.value_counts()) #(0:4138,1:4138)
# Model Training
# training with default hyperparameters
# dictionary of models
models = {
    "DecisionTree" : DecisionTreeClassifier(random_state=42),
    "Random Forest" : RandomForestClassifier(random_state=42),
    "XGBoost" : XGBClassifier(random_state=42)
}
# why random_state=42 is used for each model ?
# dictionary to store cross validation results 
cv_scores = {}
# perform 5-fold cross validation for each model
for model_name,model in models.items() :
   # print(f"Training {model_name} with default parameters")
   scores = cross_val_score(model, X_train_smote, Y_train_smote, cv=5, scoring="accuracy")
   cv_scores[model_name] = scores
   # print(f"{model_name} cross validation accuracy :{np.mean(scores):.2f}" )
   # print("-"*70)
# Random Forest gives the highest accuraccy compared to other models with default parameters
rfc = RandomForestClassifier(random_state=42)
model.fit(X_train_smote,Y_train_smote)
# Model Evaluation
Y_test_pred = model.predict(X_test)
# print("Accuracy score :\n",accuracy_score(Y_test,Y_test_pred))
# print("Confusion Matrix :\n",confusion_matrix(Y_test,Y_test_pred))
# print("Classification Report :\n",classification_report(Y_test,Y_test_pred))
# saved the trained model as a pickle file 
rfc.fit(X_train,Y_train)
model_data = {"model":rfc,"features_names":X.columns.tolist()}
with open("Customer_churn_model.pkl","wb") as f :
    pickle.dump(model_data,f)
# load the saved model and build a prediction system
with open("Customer_churn_model.pkl","rb") as f:
    model_data = pickle.load(f)
loaded_model = model_data["model"]
feature_names = model_data["features_names"]
# print(loaded_model)
#print(feature_names)
customer_data = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
}
customer_datadf = pd.DataFrame([customer_data])
with open("encoders.pkl","rb") as f :
    encoders = pickle.load(f)
# encode categorical features using the saved encoders
for column,encoder in encoders.items() :
    customer_datadf[column] = encoder.transform(customer_datadf[column])
# make a prediction
prediction = loaded_model.predict(customer_datadf)
pred_prob = loaded_model.predict_proba(customer_datadf)
print(prediction)
# results
print(f"Prediction : {'Churn' if prediction[0] == 1 else 'No Churn'}")
print(f"Prediction Probability : {pred_prob}")