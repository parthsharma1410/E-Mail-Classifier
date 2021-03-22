#importing pandas
import pandas as pd

#reading data from csv
df = pd.read_csv('cars.csv')
df

#shape of data
df.shape

#info of data
df.info()

#head of data
df.head()

#describing the data
df.describe()

#displaying columns
df.columns

#reading specific columns 
df[["price","mileage"]]

#finding the null values in the dataset
df.isnull()

#dropping null values from the dataset
df.dropna(axis=0)


#filling the null with the default
df.fillna(value='0')

#filling null spaces with forward fill
df.ffill()

#filling the null with the backward fill
df.bfill()

#filling the null spaces with mean fill
val = str(df["lot"].mean())
df.fillna(value = val)