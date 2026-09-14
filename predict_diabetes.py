#Diabetes prediction using Logistic regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

#Load the Pima Indians Diabetes Dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

df= pd.read_csv(url, names=column_names)

print("Diabetes Dataset:")
print(df.head())

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix: {conf_matrix}")
print(f"Classification Report: {class_report}")

#visualize confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

new_data = pd.DataFrame({
    'Pregnancies': [5],
    'Glucose': [120],
    'BloodPressure': [72],
    'SkinThickness': [35],
    'Insulin': [80],
    'BMI': [32.0],
    'DiabetesPedigreeFunction': [0.5],
    'Age': [42]
})

predicted_outcome = model.predict(new_data)

print(f"Predicted Output: {'Diabetic' if predicted_outcome[0] == 1 else 'Non-Diabetic'}")