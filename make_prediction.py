import pandas as pd
import pickle
from sklearn.metrics import classification_report

NUM = 5

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('target_numbers_vals.pkl','rb') as file:
    target_numbers_vals = pickle.load(file)

data_test = pd.read_csv('data_test.csv')
y_test = data_test['cut'].values
X_test = data_test.drop('cut', axis=1)

# Выполнение предсказаний
y_pred = model.predict(X_test)

print(f"Печать {NUM} предсказаний:")
for i in range(NUM):
    predicted_val = y_pred[i][0]
    true_val = y_test[i]
    print(
        f"fact: {target_numbers_vals[predicted_val]:<10} | pred: {target_numbers_vals[true_val]:<10} | delta: {true_val-predicted_val}"
    )

print("\nОценка модели:")
print(classification_report(y_test, y_pred, zero_division=0))
print("score =", round(model.score(X_test,y_test), 2))
