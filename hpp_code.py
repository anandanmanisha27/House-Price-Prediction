import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file = r"C:\Users\kanna\Downloads\HousePricePrediction.xlsx"
dataset = pd.read_excel(file)

# Printing first 5 records of the dataset
print(dataset.head(5))

dataset.shape

# Identifying categorical and numerical variables
obj = (dataset.dtypes == 'object')# creates a boolean serires
object_cols = list(obj[obj].index)# for which it is true, it extracts those indexes and gets the values. Here indices is nothing but the column name or first word in excel

print("Categorical variables:", len(object_cols))
print(object_cols)

int_ = (dataset.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:", len(num_cols))

fl = (dataset.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:", len(fl_cols))







# Correlation heatmap
numerical_dataset = dataset.select_dtypes(include=['number'])
plt.figure(figsize=(12,6))
sns.heatmap(numerical_dataset.corr(), cmap='magma', fmt='.2f', linewidths=2, annot=True)
plt.title("Correlation Heatmap")
plt.show()  # Ensure the heatmap is displayed


unique_values = []
for col in object_cols:
  unique_values.append(dataset[col].unique().size)
plt.figure(figsize=(10,6))
plt.title('No. Unique values of Categorical Features')
plt.xticks(rotation=45)
sns.barplot(x=object_cols,y=unique_values)


plt.xlabel("Categorical Features")
plt.ylabel("Number of Unique Values")
plt.show()  # Ensure the bar plot is displayed



plt.figure(figsize=(18, 36))
plt.title('Categorical Features: Distribution')
plt.xticks(rotation=90)
index = 1


for col in object_cols:
    y=dataset[col].value_counts()
    plt.subplot(11,4,index)
    plt.xticks(rotation=90)
    sns.barplot(x=list(y.index),y=y)
    index+=1
plt.show()



dataset.drop(['Id'],axis=1,inplace=True)
dataset['SalePrice']=dataset['SalePrice'].fillna(dataset['SalePrice'].mean())
new_dataset = dataset.dropna()

new_dataset.isnull().sum()


from sklearn.preprocessing import OneHotEncoder

s = (new_dataset.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)
print('No. of. categorical features: ', 
      len(object_cols))


#OH_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
OH_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
OH_cols.index = new_dataset.index
OH_cols.columns = OH_encoder.get_feature_names_out()
df_final = new_dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)

# This code is modified by Susobhan Akhuli



from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
X=df_final.drop(['SalePrice'],axis=1)
Y=df_final['SalePrice']

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0)


from sklearn import svm

from sklearn.svm import SVC

from sklearn.metrics import mean_absolute_percentage_error

model_SVR=svm.SVR()
model_SVR.fit(X_train,Y_train)
Y_preds=model_SVR.predict(X_valid)

print(mean_absolute_error(Y_valid,Y_preds))



from sklearn.ensemble import RandomForestRegressor

model_RFR = RandomForestRegressor(n_estimators=10)
model_RFR.fit(X_train, Y_train)
Y_pred = model_RFR.predict(X_valid)

mean_absolute_percentage_error(Y_valid, Y_pred)



from sklearn.linear_model import LinearRegression

model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
Y_pred = model_LR.predict(X_valid)

print(mean_absolute_percentage_error(Y_valid, Y_pred))

plt.figure(figsize=(8, 6))
plt.scatter(Y_valid,Y_preds)
plt.xlabel("Actual price")
plt.ylabel("predicted price")
plt.title("actual price Vs predicted price")
plt.show()


#new_house=pd.DataFrame({'MSSubClass':110,"MSZoning":"RM","LotArea" :9000,"LotConfig":"FR2","BldgType":"1Fam","OverallCond":5,"YearBuilt":2005,"YearRemodAdd":2005,"Exterior1st":"Wd Sdng","BsmtFinSF2":30,"TotalBsmtSF":1000})





new_house = pd.DataFrame({
    'MSSubClass': [110],
    'MSZoning': ["RM"],
    'LotArea': [9000],
    'LotConfig': ["FR2"],
    'BldgType': ["1Fam"],
    'OverallCond': [5],
    'YearBuilt': [2005],
    'YearRemodAdd': [2005],
    'Exterior1st': ["Wd Sdng"],
    'BsmtFinSF2': [30],
    'TotalBsmtSF': [1000]
})
# Ensure new_house matches feature engineering steps
new_house_encoded = new_house.copy()

# Apply One-Hot Encoding to categorical features
OH_cols_new = pd.DataFrame(OH_encoder.transform(new_house[object_cols]))
OH_cols_new.columns = OH_encoder.get_feature_names_out()
OH_cols_new.index = new_house.index

# Drop categorical columns from new_house and add encoded columns
new_house_encoded.drop(object_cols, axis=1, inplace=True)
new_house_encoded = pd.concat([new_house_encoded, OH_cols_new], axis=1)

# Ensure columns match the trained model's input
new_house_encoded = new_house_encoded.reindex(columns=X_train.columns, fill_value=0)

# Make the prediction
predicted = model_SVR.predict(new_house_encoded)
print("Predicted price:", predicted)
