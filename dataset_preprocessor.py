import pandas as pd

def stock_dataset_processor(dataset: pd.DataFrame) -> pd.DataFrame:
    print(f'Stock price data is from {dataset["Date"][0]} to {dataset["Date"][len(dataset["Date"])-1]}')
    
    dataset['Date'] = dataset['Close']

    dataset = dataset.drop(columns='Volume', axis=1)

    for column in dataset.columns:
        if column != 'Close':
            dataset[column] = dataset[column].shift(1)
            dataet = dataset.rename(columns={column: ('precious_day_' + column)})
    
    dataset = dataset.rename(columns={'Close' : 'closing_price'})
    dataset = dataset.dropna()

    return dataset
    
