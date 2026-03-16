import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from matplotlib import pyplot as plt



#load the datset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


#Adds a new column that encodes the various string category values into numbers.
train["Color"]    = LabelEncoder().fit_transform(train["Color"])
train["Spectral_Class"] = LabelEncoder().fit_transform(train["Spectral_Class"])
test["Color"]    = LabelEncoder().fit_transform(test["Color"])
test["Spectral_Class"] = LabelEncoder().fit_transform(test["Spectral_Class"])



x_train = train[["Temperature", "L", "R", "A_M", "Color", "Spectral_Class"]]
y_train = train["Type"]

#testing for best iteration value for the model

interation_tests = [100, 500, 1000, 2000, 5000]

for interation in interation_tests:
    model = LogisticRegression(max_iter=interation, random_state=42)
    model.fit(x_train, y_train)

    y_label = test['Type']
    y_test = test[["Temperature", "L", "R", "A_M", "Color", "Spectral_Class"]]
    pred = model.predict(y_test)
    print("\nAccuracy:", accuracy_score(y_label, pred))


