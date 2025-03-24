import pandas as pd
from catboost import CatBoostClassifier
import pickle

data_train = pd.read_csv('data_train.csv')

y_train = data_train['cut'].values
X_train = data_train.drop('cut', axis=1)

model = CatBoostClassifier(iterations=550, learning_rate=0.05, depth=5).fit(X_train,y_train, verbose=False)

# Сохранить в файл
with open('model.pkl', 'wb') as file: # Открыть файл в режиме записи байтов ('wb')
    pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)