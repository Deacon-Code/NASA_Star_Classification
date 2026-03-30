import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold



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


model = DecisionTreeClassifier(random_state=42, max_depth=3, criterion="gini")


#fitting the model
model.fit(x_train, y_train)

#testing on the holdout data
y_label = test['Type']
y_test = test[["Temperature", "L", "R", "A_M", "Color", "Spectral_Class"]]
pred = model.predict(y_test)
print("\nAccuracy:", accuracy_score(y_label, pred))


print("TESTING K FOLD CROSS VALIDATION")
whole_data = pd.read_csv("Stars.csv")
#convert the categorical features to numeric using label encoding
whole_data["Color"]    = LabelEncoder().fit_transform(whole_data["Color"])
whole_data["Spectral_Class"] = LabelEncoder().fit_transform(whole_data["Spectral_Class"])
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

print ("SPLIT: ", list(enumerate(kf.split(whole_data))))  # This will show the indices of the splits
for fold, (train_index, test_index) in enumerate(kf.split(whole_data)):
    print(f"\n=== Fold {fold+1} ===")
    print(f"Train indices: {train_index[:5]}...{train_index[-5:]}"
          f" | Test indices: {test_index[:5]}...{test_index[-5:]}")
    train_fold = whole_data.iloc[train_index]
    test_fold = whole_data.iloc[test_index]

    x_train_fold = train_fold[["Temperature", "L", "R", "A_M", "Color", "Spectral_Class"]]
    y_train_fold = train_fold["Type"]
    x_test_fold = test_fold[["Temperature", "L", "R", "A_M", "Color", "Spectral_Class"]]
    y_test_fold = test_fold["Type"]

    model.fit(x_train_fold, y_train_fold)
    fold_pred = model.predict(x_test_fold)
    fold_acc = accuracy_score(y_test_fold, fold_pred)
    fold_accuracies.append(fold_acc)
    print(f"Fold {fold+1} Accuracy: {fold_acc:.4f}")