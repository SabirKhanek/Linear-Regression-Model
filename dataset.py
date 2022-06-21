import numpy as np
from dataset_preprocessor import stock_dataset_processor
import pandas as pd
from easygui import fileopenbox   # to get path of csv file

def open_stock_dataset(path: str, epoch=300, learning_rate=0.00001) -> pd.DataFrame:
    print(f'Using epoch: {epoch} and learning_rate: {learning_rate} configure in case of error')
    print('Data Source             : '+path)
    print('Feature to be predicted : closing_price')
    # fetch stock dataset from given path
    
    dataset = pd.read_csv(path)
    dataset = stock_dataset_processor(dataset)

    # Training and Testing Rows
    total_rows = np.array(dataset).shape[0]
    test_rows = int(0.2 * total_rows)
    train_rows = total_rows - test_rows
    print(f"Training Rows: {train_rows}\nTest Rows: {test_rows}")

    # seperating y column from dataset
    dataset_y = dataset['closing_price']
    dataset_x = dataset.drop('closing_price', axis=1)

    # converting datafram -> numpy_array
    dataset_y = dataset_y.to_numpy()
    dataset_x = dataset_x.to_numpy()

    # seperating test data
    test_data_x = dataset_x[-(test_rows):].tolist()
    test_data_y = dataset_y[-(test_rows):].tolist()

    # seperating training data
    dataset_x = dataset_x[:train_rows].tolist()
    dataset_y = dataset_y[:train_rows].tolist()

    return dataset_x, dataset_y, test_data_x, test_data_y, epoch, learning_rate


def amazon_dataset() -> tuple[list[list], list[list], list[list], list[list], int, float]:
    print('\nSelected Dataset        : Amazon Stock Data 1D (1997-05-15 to 2020-08-14)')
    print('Data Source             : local preprocessed form of dataset from https://www.kaggle.com/datasets/aayushmishra1512/faang-complete-stock-data')
    print('Feature to be predicted : closing_price')

    # cost function will be optimized till that point
    suggested_epoch = 200
    
    # greater than that cost graph will diverge instead of converging
    # more information will be provided in the documentation
    suggested_learning_rate = 0.0000005

    dataset = pd.read_csv('local_datasets/amazon_stock_dataset.csv')

    total_rows = np.array(dataset).shape[0]
    test_rows = int(0.2 * total_rows)
    train_rows = total_rows - test_rows

    # seperating y column from dataset
    dataset_y = dataset['closing_price']
    dataset_x = dataset.drop('closing_price', axis=1)

    # converting datafram -> numpy_array
    dataset_y = dataset_y.to_numpy()
    dataset_x = dataset_x.to_numpy()

    # seperating test data
    test_data_x = dataset_x[-(test_rows):].tolist()
    test_data_y = dataset_y[-(test_rows):].tolist()

    # seperating training data
    dataset_x = dataset_x[:train_rows].tolist()
    dataset_y = dataset_y[:train_rows].tolist()

    return dataset_x, dataset_y, test_data_x, test_data_y, suggested_epoch, suggested_learning_rate

def select_dataset() -> tuple[list[list], list[list], list[list], list[list], int, float]:
    print('Regression Model Test Simulations')
    print('Dataset will be loaded in such a way that in each row')
    print('closing price will be placed with information of previous day')
    print('Select from local datasets:')
    print('1. Amazon stock price dataset')
    print('2. Apple Stock price dataset')
    print('3. Facebook Stock price dataset')
    print('4. Open custom stock data')
    inp = input("Enter your choice: ")
    inp = int(inp)
    if inp in range(1, 5):
        if inp == 1:
            dataset_x , dataset_y, test_data_x, test_data_y, epoch, learn_rate = amazon_dataset()
        if inp == 2:
            dataset_x , dataset_y, test_data_x, test_data_y, epoch, learn_rate = open_stock_dataset('local_datasets/Apple.csv')
        if inp == 3:
            dataset_x , dataset_y, test_data_x, test_data_y, epoch, learn_rate = open_stock_dataset('local_datasets/Facebook.csv')
        if inp == 4:
            print('Select dataset file in the opened dialog box')
            path = fileopenbox()
            dataset_x , dataset_y, test_data_x, test_data_y, epoch, learn_rate = open_stock_dataset(path)

    return dataset_x, dataset_y, test_data_x, test_data_y, epoch, learn_rate
