import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

data = pd.read_csv('jhatu.csv')

X = data.drop('label', axis=1) 
y = data['label']              

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)

print(f"Model Accuracy: {score * 100:.2f}%")

with open('model.p', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully as 'model.p'")