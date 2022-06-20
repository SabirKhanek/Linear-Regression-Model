import pandas as pd

def property_dataset() -> tuple[list[list], list[list], list[list], list[list], int, float]:
    print('Selected Dataset: Property Prices Dataset')
    print('Data Source: local preprocessed form of dataset from https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data')
    # cost function will be optimized till that point
    suggested_epoch = 10000
    
    # greater than that cost graph will diverge instead of converging
    # more information will be provided in the documentation
    suggested_learning_rate = 0.000000001


    # read from training dataset and dropping extra columns
    dataset = pd.read_csv('local_datasets/property_train_data.csv').drop(['Unnamed: 0', 'Id'], axis=1)
    
    # seperating y column from dataset
    dataset_y = dataset['SalePrice']
    dataset_x = dataset.drop('SalePrice', axis=1)

    # converting datafram -> numpy_array -> lists
    dataset_y = dataset_y.to_numpy().tolist()
    dataset_x = dataset_x.to_numpy().tolist()

    # read from training dataset and dropping extra columns
    test_dataset = pd.read_csv('local_datasets/property_test_data.csv').drop(['Unnamed: 0', 'Id'], axis=1)

    # seperating y column from dataset
    test_data_y = test_dataset['SalePrice']
    test_data_x = test_dataset.drop('SalePrice', axis=1)

    # converting datafram -> numpy_array -> lists
    test_data_y = test_data_y.to_numpy().tolist()
    test_data_x = test_data_x.to_numpy().tolist()

    del dataset
    del test_dataset

    return dataset_x, dataset_y, test_data_x, test_data_y, suggested_epoch, suggested_learning_rate


def car_dataset():
    print('Selected Dataset: CarDekho 2nd hand car dataset')
    print('Data Source: local preprocessed form of dataset from https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho')
    # cost function will be optimized till that point
    suggested_epoch = 100000
    
    # greater than that cost graph will diverge instead of converging
    # more information will be provided in the documentation
    suggested_learning_rate = 0.0000000001


    # We'll read only required columns from the cardekho.com dataset
    dataset = pd.read_csv("local_datasets/cardekho_data.csv", usecols= ['name','year', 'km_driven', 'selling_price'])

    # now we want to filter data for 1 specific car model
    dataset.set_index('name')
    dataset = dataset[dataset['name'] == 'Maruti Swift Dzire VDI']

    # generating new indexes just to discard old indexes before filtering
    dataset.reset_index(inplace=True)   
    del dataset['name']
    del dataset['index']
    
    # seperating y column from dataset
    dataset_y = dataset['selling_price']
    dataset_x = dataset.drop('selling_price', axis=1)

    # converting datafram -> numpy_array -> lists
    dataset_y = dataset_y.to_numpy().tolist()
    dataset_x = dataset_x.to_numpy().tolist()

    test_data_x = dataset_x[-19:]
    test_data_y = dataset_y[-19:]

    dataset_x = dataset_x[:50]
    dataset_y = dataset_y[:50]

    return dataset_x, dataset_y, test_data_x, test_data_y, suggested_epoch, suggested_learning_rate
