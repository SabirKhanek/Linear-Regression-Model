"""
Multiple Regression Model
Authors:
       Sabir Khan
       Hurmat Zahra
Note:
    Libraries we are using here is just for dataset handling and related operations
    dataset is meant to be used to train our model
    there is also a test dataset which include features of real word objects
    through which we measure accuracy of our model by predicting against those features
    then we visualize differnt simulations on graphs
pandas         : for dataset manipulation
matplotlib     : graph visualization simulations
numpy          : Arrays and 2D arrays(Matrices) functionality and functions
alive_progress : To measure training time and training speed

We are using numpy functions to perform arithmetic operations on matrices 
because our main scope is to build an efficient prediction model

However, in V1 of this model we didn't use any of the matrix functions of the numpy
It works though training time is much slower than V2 regression model
""" 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from alive_progress import alive_bar; import time
import dataset                    # local py file
from easygui import fileopenbox   # to get path of csv file

class multiple_regression_model:
    """
    :params
        coeff_         : it'll store slopes for every feature in our best fit line
        train_log      : it'll keep record of Mean Squared Error of training data after every epoch iteration
                         we can then plot a graph to check whether error decreases or increases with epoch iteration
        cost_diff_log  : it'll store difference between errors generated in last iteration and current iteration
        trained_epch   : it'll store how many epochs iteration are done during training time
    """
    coeff_ : np.array = np.empty(shape=(0))
    train_log : list = []
    cost_diff_log : list = []
    trained_at_epoch : int = 0

    """
    :params
        lst: 2D array (matrix) order: (n x m)
    :out
        lst: 2D array (matrix) order: (n x (m+1)) extra column is padded by 1
    """
    def append_one(self, lst: list[list]) -> list[list]:
        # select a row in each iteration and append 1 at the last of it
        for i in range(0, len(lst)):
            lst[i].append(1)
        return lst

    """
    :params
        features: matrix including features of testing data (n x m) which will be 
                  padded by (n x (m+1)) where extra column is padded by 1 to generate intercept of the best fit
    :out
        y_predicted:  which include predictions against every feature set produced by multiplication of thetas
    """
    def predict(self, features: list[list]) -> np.array:
        # add a column with 1 on right side of the matrix
        features = self.append_one(features)
        # converting 2d list in np array matrix so that multiplication of vectors can evaluated
        features = np.array(features)
        # (n x m) . (m x 1) resultant matrix is stored in y_predicted
        y_predicted = np.dot(features, self.coeff_)
        return y_predicted
    
    """
    This function will return Mean Squared Error when y is predicted beforehand
    :params
        data_length  : number of rows in dataset means number of objects
        actual       : actual values of y
        y_predicted  : values of y predicted by the model
    :out
        cost         : return MSE based on actual and predicted values
    """
    def cost_function_mse_from_predicted(self, data_length : int, actual : list or np.array, y_predicted: list or np.array) -> float:
        cost = 0

         # calculate mean squared error of every object with features in dataset and sum it in cost feature
        for i in range(0, data_length):
            cost += (actual[i] - y_predicted[i]) ** 2

        # divide sum of all cost with number of objects
        cost = cost / data_length
        return cost

    """
    This function will return Mean Squared Error when predicted values are not given before so it has to be predicted
    :params
        features  :  matrix including features of testing data (n x m) which will be passed to self.predict()
        actual    :  actual 'y' values against every object in feature row 
    :out
        cost         : return MSE based on actual and predicted values returned by self.predict(features)
    """
    def cost_function_mse(self, features: list[list] or np.array, actual: list or np.array) -> float:
        # predict function returns np array as order(n,1)
        y_predicted = self.predict(features)

        # convert that array as order (1, n)
        y_predicted = y_predicted.reshape(y_predicted.shape[0], 1)

        cost = 0
        # calculate mean squared error of every object with features in dataset and sum it in cost feature
        for i in range(0, len(features)):
            cost += (actual[i] - y_predicted[i]) ** 2
        
        # divide sum of all cost with number of objects
        cost = cost / len(features)
        return cost

    
    """
    Training method which will optimize error for the dataset through every epoch iteration till
    MSE is lowest and cannot be optimized anymore
    :params
        features       : matrix including features of training data (n x m)
        actual         : actual 'y' values against every object in feature row
        epoch          : number of times the data is iterated through the model 
        learning_rate  : it specifies the speed towards optimization
                         it'll be explained further in report documentation 
    """    
    def gradient_descent(self, features, actual, epoch, learning_rate=0.000001):
        # getting number of objects in the dataset of which features are included i.e., n_rows in features
        dataset_length = len(features)

        # storing number of epochs (dataset iteration) happened to train the model 
        self.trained_at_epoch = epoch

        # padding by 1 so (n x m) becomes (n x (m+1)) where 
        # extra column is padded by 1 to generate intercept of the best fit
        features = self.append_one(features)

        # get number of features for each object(row) i.e., n_columns in features 
        n_features = len(features[0])
        
        # Display number of objects and number of features in the dataset of 
        # order (objects x features)
        print(f"Dataset length: {dataset_length}\nNo. of features: {n_features-1}")
        
        # print learning rate and number of epochs
        print(f"Epochs target: {epoch}\nLearning rate: {learning_rate}\n")

        # initializing matrix of zeros for slope coefficients with order (n_features, 1)
        self.coeff_ = np.zeros(shape=(n_features, 1))

        # converting 2d feature list and list of actual values 
        # in np.array matrix for efficient matrix arithmetics
        features = np.array(features)
        actual = np.array(actual)

        # converting (1 x n) actual matrix to (n x 1) so that multiplication with 
        # features (m x n) can evaluated        
        actual = actual[:].reshape(actual.shape[0], 1)

        # alive_bar is used to display progress of epoch iteration
        with alive_bar(total=epoch, title='Training: ') as bar:
            for counter in range(0, epoch):
                # initializing matrix of zeros for temp slope coefficients with order (n_features, 1)
                new_coeff_ = np.zeros(shape=(n_features, 1))
                
                # current y predictions based on slope coefficients are generated by
                # matrix multipllication of 
                # (objects x features) . (features x 1)
                # resultant: (objects x 1) which will have predicted values of y
                # against every object in every row
                y_predicted = np.dot(features, self.coeff_)

                # partial derivative of squared error to generate slope of every feature
                # features is transposed t match the order for the matrix
                # as (actual - y_predicted) generates matrix of (objects x 1)
                # features have order of (objects x features)
                # (objects x features) . (objects x 1) is not possible
                # => (features x objects) . (objects x 1) is possible
                new_coeff_ = -(2/dataset_length) * np.dot(features.T , (actual - y_predicted))

                # now optimize best fit thetas by subtracting the dervivative multiplied by step rate
                self.coeff_ -= learning_rate * new_coeff_

                # keeping log of cost after every epoch for analysis
                temp_cost = np.sum(np.square(actual - y_predicted)) 
                temp_cost = temp_cost / dataset_length

                # keeping log of differences between cost after every epoch iteration
                cost_diff = 0
                if counter > 100:
                    cost_diff = self.train_log[-1] - temp_cost
                
                self.train_log.append(temp_cost)
                self.cost_diff_log.append(cost_diff) 
                
                # updating progress bar of training
                bar()


def cost_intercept_parabola(model: multiple_regression_model, features: list[list], actual: list) -> tuple[list, list]:
    copy_coeff_ = model.coeff_
    minima = model.coeff_[-1, 0]
    copy_coeff_[-1, 0] = minima - 40

    cost_list = []
    intercept = []

    features = np.array(features)

    for i in range(81):
        y_predicted = np.dot(features, copy_coeff_)
        y_predicted = y_predicted.reshape(1, y_predicted.shape[0]).tolist()
        y_predicted = y_predicted[0]

        # error calculation
        cost = 0

         # calculate mean squared error of every object with features in dataset and sum it in cost feature
        for i in range(0, features.shape[0]):
            cost += (actual[i] - y_predicted[i]) ** 2

        # divide sum of all cost with number of objects
        cost = cost / features.shape[0]
        cost_list.append(cost)
        intercept.append(copy_coeff_[-1, 0])
        copy_coeff_[-1, 0] += 1

    return cost_list, intercept


if __name__ == "__main__":
    dataset_x , dataset_y, test_data_x, test_data_y, epoch, learn_rate = dataset.select_dataset()

    # initializing our model    
    model = multiple_regression_model()

    # training the model with training data
    model.gradient_descent(dataset_x, dataset_y, epoch, learn_rate)

    # getting predictions by fetching test data features to the model
    
    y_predicted = model.predict(test_data_x)
    
    # reshaping initially vertical vector to horizontal vector and converting them to lists i.e., (n, 1) -> (1, n)
    y_predicted = y_predicted.reshape(1, y_predicted.shape[0]).tolist()
    y_predicted = y_predicted[0]
    test_data_y = np.array(test_data_y).reshape(1, len(test_data_y))
    test_data_y = test_data_y.tolist()[0]


    # measuring Mean Squared Error of predicted values
    test_err = model.cost_function_mse_from_predicted(data_length= len(test_data_x), actual=test_data_y, y_predicted=y_predicted)

    # displaying actual vs predicted table
    print('\nActual vs Predicted: ')
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')
    pd.set_option('display.precision', 3)
    result_df = pd.DataFrame({'Actual': test_data_y,
                            'Predicted': y_predicted,
                            'Difference': np.array(test_data_y)-np.array(y_predicted)})

    print(result_df)

    # Displaying final error % after training and prediction accuracy achieved from testing data 
    print(f"Training Error: {np.round(model.train_log[-1], decimals=4)}")

    # parabolic graph
    cost_intercept_costs,  cost_intercept_intercepts = cost_intercept_parabola(model=model, features=test_data_x, actual=test_data_y)


    # Plot the simulations using matplotlib
    figure, axis = plt.subplots(2, 2, figsize=(10, 10))
    axis[0][0].plot(cost_intercept_intercepts, cost_intercept_costs)
    axis[0][0].set_xlabel('Y - Intercept')
    axis[0][0].set_ylabel('Mean Squared Error (MSE)')
    axis[0][0].set_title('Error variation wrt Y - Intercept')

    axis[0][1].plot(list(range(len(y_predicted))), y_predicted, color='red', label='Predicted')
    axis[0][1].plot(list(range(len(test_data_y))), test_data_y, color='blue', label='Actual')
    axis[0][1].legend()
    axis[0][1].set_title('Predicted vs Actual')

    axis[1][1].plot(list(range(0, model.trained_at_epoch)), model.train_log)
    axis[1][1].set_xlabel('Epoch')
    axis[1][1].set_ylabel('MSE (Mean Squared Error)')
    axis[1][1].set_title('Error variance with increasing Epochs')
    
    plt.show()
