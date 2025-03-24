import pandas as pd

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


# Экспорт в csv файлы
train.to_csv('data_train.csv', index = False)
test.to_csv('data_test.csv', index = False)