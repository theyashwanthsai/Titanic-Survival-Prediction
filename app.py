import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
from sklearn.model_selection import train_test_split

df = pd.read_csv("datasets/titanic.csv")
df.drop(['Name'], axis=1, inplace=True)

df['Age'].fillna(df['Age'].mean(), inplace=True)
# df['Embarked'].fillna('S', inplace=True)
df['Sex'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
# df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

def predict_survival_rate(pclass, sex, age, sibsp, parch, fare):
    input = np.array([[pclass, sex, age, sibsp, parch, fare]])
    prediction = model.predict(input)
    if prediction[0] == 0:
        return 'not survive'
    else:
        return 'survive'


st.title('Titanic Survival Prediction')
pclass = st.selectbox('Passenger Class', [1, 2, 3])
sex = st.selectbox('Sex', ['female', 'male'])
if sex == 'female':
    sex = 0
else:
    sex = 1
age = st.slider('Age', 0, 100, 30)
sibsp = st.slider('Number of Siblings/Spouses Aboard', 0, 10, 0)
parch = st.slider('Number of Parents/Children Aboard', 0, 10, 0)
fare = st.slider('Fare', 0, 600, 10)
# embarked = st.selectbox('Embarked', ['S', 'C', 'Q'])
if st.button('Predict'):
    result = predict_survival_rate(pclass, sex, age, sibsp, parch, fare)
    st.success(f'The passenger is likely to {result}.')