import pandas as pd
from sklearn.model_selection import train_test_split
adult_data = pd.read_csv('./data/adult.csv')
adult_data_train, adult_data_test = train_test_split(adult_data, test_size=0.25, random_state=42)
adult_data_train.to_csv('./data/processed_train.csv', index=False)
adult_data_test.to_csv('./data/processed_test.csv', index=False)