import pandas as pd
import pickle


DATASET_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/refs/heads/master/diamonds.csv"

# Загрузка данных
df = pd.read_csv(DATASET_URL)

# Подготовка данных
target_vals = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'] # По возрастанию качества
target_numbers_vals = dict(enumerate(target_vals))
target_vals_numbers = {val: num for num, val in target_numbers_vals.items()}
df['cut'] = df['cut'].map(target_vals_numbers)
df_enc = pd.get_dummies(df, columns=['color', 'clarity']) # One Hot Encoding

# Деление на обучающую и тестовую выборку
rows_num = df_enc.shape[0]
train = df_enc.loc[:int(rows_num*0.9)]
test = df_enc.loc[rows_num*0.9:]

# Экспорт в файлы
train.to_csv('data_train.csv', index = False)
test.to_csv('data_test.csv', index = False)
# Сохранение соответствия категорий качества цифрам для дальнейшей интерпретации результатов
with open('target_numbers_vals.pkl','wb') as file:
    pickle.dump(target_numbers_vals, file)