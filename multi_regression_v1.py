"""
Multiple Regression Model
Authors:
       Sabir Khan
       Hurmat Zahra
Note:
    There is also a test dataset which include features of real word objects
    through which we measure accuracy of our model by predicting values based on features
    then we visualize differnt simulations on graphs
dataset        : initialized in dataset.py which will be used to fetch data from csv in form of python lists
matplotlib     : graph visualization simulations
alive_progress : To measure training time and training speed of epochs passes just for analyzing process
easygui        : We are using a module from this library which is only used to get path of a csv file from user 

In this version of model we didn't use any of the library function except we fetched
data from csv file using pandas library in another module that we included here 
""" 
import dataset
import matplotlib.pyplot as plt
from alive_progress import alive_bar

class multiple_regression_model:
    """
    :params
        coeff_     : list of slope coefficients of the features
        intercept  : y-intercept of the best fit line
        train_log  : list of training error in every epoch iteration
    """
    coeff_ = []
    intercept = 0
    train_log = []

    """
    :params
        n          : number of features for every object
    :out
        list       : return a list of size n_features filled with 0 
    """
    def f_zero(self, n: int) -> list:
        lst = []
        for i in range(0, n):
            lst.append(0)
        return lst

    """
    :params
        features  : features of the object
    :out
        y         : predict y value based on feature

    in every iteration product of slope coefficient with features is summed in y for a single object with features
    """
    def predict(self, features: list) -> float or int: 
        y = self.intercept
        for i in range(0, len(features)):
            y += features[i] * self.coeff_[i]
        return y

    """
    :params
        features    : list of object with features that is 2D list with order (object x features)
    :out
        y_predicted : list of predicted values for every object
     in every iteration y value is predicted for an object and added to a list at last the list is returned
    """
    def predict_test_data(self, features: list[list]) -> list:
        y_predicted = []
        for object in features:
            prediction = self.predict(object)
            y_predicted.append(prediction)
        return y_predicted

    """
    :params
        features : matrix (object x features)
        actual   : actual value of each object
     :out
        cost     : mean squared error

    in each iteration cost accumulated for every object
    """
    def cost_function_mse(self, features: list[list], actual: list) -> float:
        cost = 0
        for i in range(0, len(features)):
            cost += (actual[i] - self.predict(features[i])) ** 2
        cost = (1/len(features)) * cost
        return cost
    
    def gradient_descent(self, features, actual, epoch, learning_rate=0.01):
        # Get number of rows that is columns
        dataset_length = len(features)

        # Get number of columns that is objects in training dataset and display it
        n_features = len(features[0])
        print(f"Dataset length: {dataset_length}\nNo. of features: {n_features}")

        # initialize that list coeff_ which will store slope coefficients for each feature
        self.coeff_ = self.f_zero(n_features)

        # alive bar will visualize progress bar of training process on screen
        with alive_bar(total=epoch, title='Training: ') as bar:
            # in each iteration the MSE will be reduce that is an epoch of training dataset is completed
            for counter in range(0, epoch):
                # store temporary intercept and coefficients of features in the list
                new_intercept = 0
                new_coeff_ = self.f_zero(n_features)
                
                # add partial derivative wrt intercept of intercept for every object in new_intercept
                for i in range(0, dataset_length):
                    new_intercept += (-2/dataset_length) * (actual[i]-self.predict(features[i]))
                
                # outer loop will select object in every iteration
                # inner loop will select a feature in every iteration
                # in every outer loop iteration partial derivative is stored in list of new coefficient for every sum
                for i in range(0, dataset_length):
                    for j in range(0, n_features):
                        new_coeff_[j] += ((-2 * features[i][j]) / dataset_length) * (actual[i] - self.predict(features[i]))

                # move intercept towards slope on the basis of learning rate
                self.intercept -= learning_rate * new_intercept
                
                # move every feature towards the slope on the basis of learning rate
                for i in range(0, n_features):
                    self.coeff_[i] -= learning_rate * new_coeff_[i]
                
                # keep track of costs in every iteration by appending in the list
                temp_cost = self.cost_function_mse(features, actual)
                self.train_log.append(temp_cost)
                
                # update progress on the progress bar
                bar()

if __name__ == '__main__':    
    dataset_x , dataset_y, test_data_x, test_data_y, epoch, learn_rate = dataset.select_dataset()
    
    model = multiple_regression_model()

    # train the model with training data
    model.gradient_descent(dataset_x, dataset_y, epoch, learn_rate)

    # test the model with testing data
    y_predicted = model.predict_test_data(test_data_x)

    # displaying actual vs predicted table
    print('\nActual vs Predicted: ')
    dataset.pd.set_option('display.width', 1000)
    dataset.pd.set_option('display.colheader_justify', 'center')
    dataset.pd.set_option('display.precision', 3)
    result_df = dataset.pd.DataFrame({'Actual': test_data_y,
                                    'Predicted': y_predicted})
    print(result_df)

    # Displaying final error % after training and prediction accuracy achieved from testing data 
    print(f"Training Error: {round(model.train_log[-1], 4)}")
    
    # plot actual vs predicted graph
    plt.plot(list(range(len(y_predicted))), y_predicted, color='red', label='Predicted')
    plt.plot(list(range(len(test_data_y))), test_data_y, color='blue', label='Actual')
    plt.legend()
    plt.title('Predicted vs Actual')
    plt.show()

    # plot Error variance with respect to epoch propogation
    plt.plot(list(range(0, epoch)), model.train_log)
    plt.xlabel('Epoch')
    plt.ylabel('MSE (Mean Squared Error)')
    plt.title('Error variance with increasing Epochs')
    plt.show()
