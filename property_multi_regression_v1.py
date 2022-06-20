import pandas as pd
import matplotlib.pyplot as plt
from alive_progress import alive_bar; import time

class multiple_regression_model:
    coeff_ = []
    intercept = 0
    train_log = []

    def f_zero(self, n: int) -> list:
        lst = []
        for i in range(0, n):
            lst.append(0)
        return lst

    def predict(self, features: list) -> float or int: 
        y = self.intercept
        for i in range(0, len(features)):
            y += features[i] * self.coeff_[i]
        return y

    def cost_function_mse(self, features, actual):
        cost = 0
        for i in range(0, len(features)):
            cost += (actual[i] - self.predict(features[i])) ** 2
        cost = (1/len(features)) * cost
        return cost
    
    def gradient_descent(self, features, actual, epoch, learning_rate=0.01):
        dataset_length = len(features)
        n_features = len(features[0])
        print(f"Dataset length: {dataset_length}\nNo. of features: {n_features}")
        self.coeff_ = self.f_zero(n_features)
        with alive_bar(total=epoch, title='Training: ') as bar:
            for counter in range(0, epoch):
                new_intercept = 0
                new_coeff_ = self.f_zero(n_features)
                for i in range(0, dataset_length):
                    new_intercept += (-2/dataset_length) * (actual[i]-self.predict(features[i]))
                
                for i in range(0, dataset_length):
                    for j in range(0, n_features):
                        new_coeff_[j] += ((-2 * features[i][j]) / dataset_length) * (actual[i] - self.predict(features[i]))


                self.intercept -= learning_rate * new_intercept
                for i in range(0, n_features):
                    self.coeff_[i] -= learning_rate * new_coeff_[i]
                temp_cost = self.cost_function_mse(features, actual)
                self.train_log.append(temp_cost)
                bar()


model = multiple_regression_model()

dataset = pd.read_csv('train_data.csv').drop(['Unnamed: 0', 'Id'], axis=1)
dataset_y = dataset['SalePrice']
dataset_x = dataset.drop('SalePrice', axis=1)
dataset_y = dataset_y.to_numpy().tolist()
dataset_x = dataset_x.to_numpy().tolist()

test_dataset = pd.read_csv('train_data.csv').drop(['Unnamed: 0', 'Id'], axis=1)

test_data_y = test_dataset['SalePrice']
test_data_x = test_dataset.drop('SalePrice', axis=1)
test_data_y = test_data_y.to_numpy().tolist()
test_data_x = test_data_x.to_numpy().tolist()

del dataset
del test_dataset

model.gradient_descent(dataset_x, dataset_y, 100, 0.000000001)

test_err = model.cost_function_mse(test_data_x, test_data_y)

print(f"Accuracy: {round(1-test_err, 4) * 100}%")
print(f"Training Error: {round(model.train_log[-1], 4) * 100}%")

plt.plot(list(range(0, 100)), model.train_log)
plt.show()