import pandas as pd
import pickle

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('target_numbers_vals.pkl','rb') as file:
    target_numbers_vals = pickle.load(file)

data_test = pd.read_csv('data_test.csv')
y_test = data_test['cut'].values
X_test = data_test.drop('cut', axis=1)

# Выполнение предсказаний
y_pred = model.predict(X_test)

predicted_val = y_pred[0]
true_val = y_test[0]


print(predicted_val, true_val)
