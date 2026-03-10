import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_predict


# Uploading all data to train neurons
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# Filling missing values in Age with median and creating a new feature HasCabin
median_age = train['Age'].median()
train['Age'] = train['Age'].fillna(median_age)
# print('Filled Age with median: ', median_age)

train['HasCabin'] = train['Cabin'].notna().astype(int)

train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])

train = train.drop('Cabin', axis = 1)


# Adding features FamilySize, IsAlone, and Title so we can feed more information to the neurons

train['FamilySize'] = train['SibSp'] + train['Parch'] + 1

train['IsAlone'] = train['FamilySize'].eq(1).astype(int)

train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\\.')

def simplify_title(title):
    common_titles = ['Mr', 'Miss', 'Mrs', 'Master']

    if title in common_titles:
        return title
    elif title == 'Mlle' or title == 'Ms':
        return 'Miss'
    elif title == 'Mme':
        return 'Mrs'
    else:
        return 'Rare'

train['Title'] = train['Title'].apply(simplify_title)
# print(train['Title'].value_counts())


le = LabelEncoder()

train['Sex'] = le.fit_transform(train['Sex'])
train['Title'] = le.fit_transform(train['Title'])

# print(train[['Sex', 'Title']].head(10))

# Training data

feautures = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone', 'Title', 'HasCabin']

x = train[feautures]
y = train['Survived']

model = RandomForestClassifier(n_estimators=100, random_state=42)

scores = cross_val_score(model, x, y, cv=5)
print("Each fold score: ", scores)
print("Average score: ", scores.mean())
print("Standard deviation: ", scores.std())

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0, scoring='accuracy')
grid.fit(x, y)

#print("Best parameters: ", grid.best_params_)
#print("Best score: ", grid.best_score_)

#print("Best model: ", best_model)
best_model = grid.best_estimator_
y_pred = cross_val_predict(best_model, x, y, cv=5)

cm = confusion_matrix(y, y_pred)
print("Confusion Matrix: \n", cm)

disp = ConfusionMatrixDisplay(cm, display_labels=best_model.classes_)

disp.plot()
plt.title("Confusion Matrix for Random Forest Classifier")
plt.show()


# Making in 83 percent
# testing data
be = LabelEncoder()
test['Age'] = test['Age'].fillna(test['Age'].median())
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
test['HasCabin'] = test['Cabin'].notna().astype(int)
test = test.drop('Cabin', axis=1)

test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
test['IsAlone'] = (test['FamilySize'] == 1).astype(int)
test['Title'] = test['Name'].str.extract(r' ([A-Za-z]+)\.')
test['Title'] = test['Title'].apply(simplify_title)
test['Sex'] = be.fit_transform(test['Sex'])
test['Title'] = be.fit_transform(test['Title'])

x_test = test[feautures]
predictions = best_model.predict(x_test)

submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': predictions
})

submission.to_csv('submission.csv', index=False)
print(submission.head(10))
print(f"Total predictions: {len(submission)}")
print(f"Predicted survived: {predictions.sum()}")
print("Saved submission.csv")