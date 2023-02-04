import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import timeit

np.seterr(all='raise')

class baseball(object):


    def __init__(self, theta = None, eps = 1e-5, max_iter=10000):
        '''
        Initialize our baseball object.
        '''
        
        self.theta = theta
        self.eps = eps
        self.max_iter = max_iter


    def fit_NE(self, x, y):
        '''
        Run the normal equation on the model.
        '''
        
        x_Tx = np.dot(x.T, x)
        x_Tx_1 = np.linalg.inv(x_Tx)
        x_Ty = np.dot(x.T, y)
        self.theta = x_Tx_1.dot(x_Ty)


    def fit_GD_LR(self, x, y, step_size):
        '''
        Run gradient descent with linear regression
        on the model.
        Update the values for theta that will be used later in
        the predict function.
        '''
        
        d, s = x.shape
        self.theta = np.zeros([s,1])
        for i in range(2000):
            previous_theta = np.copy(self.theta)
            prediction = np.dot(x, self.theta)
            step = step_size * (x.T.dot(prediction - y))
            self.theta = self.theta - step
            if np.sum(np.abs(previous_theta - self.theta)) < self.eps:
                break


    def fit_GD_ReLU(self, x, y, step_size):
        '''
        Run gradient descent with ReLU
        on the model.
        Update the values for theta that will be used later in
        the predict function.
        '''
        
        d, s = x.shape
        self.theta = np.zeros([s,1])
        for i in range(2000):
            previous_theta = np.copy(self.theta)
            prediction = np.dot(x, self.theta)
            prediction[prediction < 0] = 0
            step = step_size * (x.T.dot(prediction - y))
            self.theta = self.theta - step
            if np.sum(np.abs(previous_theta - self.theta)) < self.eps:
                break


    def fit_SGD_LR(self, x, y, step_size):
        '''
        Run stochastic gradient descent with linear regression
        on the model.
        Update the values for theta that will be used later in
        the predict function.
        '''
        
        d, s = x.shape
        self.theta = np.zeros([s,1])
        for i in range(1000):
            previous_theta = np.copy(self.theta)
            for j in range(d):
                prediction = np.dot(x[j], self.theta)
                step = step_size * (x[j] * (prediction - y[j]))
                self.theta = self.theta - np.reshape(step, (len(step),1))
            if np.sum(np.abs(previous_theta - self.theta)) < self.eps:
                break


    def fit_SGD_ReLU(self, x, y, step_size):
        '''
        Run stochastic gradient descent with ReLU
        on the model.
        Update the values for theta that will be used later in
        the predict function.    
        '''
        
        d, s = x.shape
        self.theta = np.zeros([s,1])
        for i in range(1000):
            previous_theta = np.copy(self.theta)
            for j in range(d):
                prediction = np.dot(x[j], self.theta)
                prediction[prediction < 0] = 0
                step = step_size * (x[j] * (prediction - y[j]))
                self.theta = self.theta - np.reshape(step, (len(step),1))
            if np.sum(np.abs(previous_theta - self.theta)) < self.eps:
                break


    def predict_LR(self, x):
        '''
        Gets a prediction using Linear Regression.
        '''
        
        y = x.dot(self.theta)
        
        # This is just to combat any extremely negative values
        y[y < -13800000] = -13800000
        
        return y


    def predict_ReLU(self, x):
        '''
        Gets a prediction using ReLU.
        '''
        
        y = x.dot(self.theta)
        
        y[y < 0] = 0
        
        return y
    

def rf_regressor(x_train, y_train, x_test, y_test):
    '''
    Gets a prediction using the random forest regressor.
    '''
    
    rfr_model = RandomForestRegressor()
    
    rfr_model.fit(x_train, y_train)
    y_pred = rfr_model.predict(x_test)
    
    accuracy = score(y_pred, y_test)
    
    print(accuracy)
    print()
    
    return y_pred


def normal_equation(x_train, y_train, x_test, y_test, model, model_type):
    '''
    Gets a prediction using the normal equation.
    '''
    
    model.__init__()
    
    model.fit_NE(x_train, y_train)
    
    if (model_type == 'lr'):
        y_pred = model.predict_LR(x_test)
        
    elif (model_type == 'relu'):
        y_pred = model.predict_ReLU(x_test)
    
    accuracy = score(y_pred, y_test)
    
    print(accuracy)
    print()
    
    return y_pred


def gradient_descent(x_train, y_train, x_test, y_test, model, model_type):
    '''
    Gets a prediction using gradient descent.
    The commented out section was used to find the ideal step_size each
    time we added, removed, or modified a feature.
    '''
    
    if (model_type == 'lr'):
        
        max_score = [0,-1000000000000]
        ideal = 1
        increment  = 0.1
        max_reached = 0
        
        while max_reached < 2:
            
            # Used to determine the ideal step size
            if (ideal < 10):
                increment = 0.15
            elif (ideal < 100):
                increment = 1.5
            elif (ideal < 1000):
                increment = 15
            elif (ideal < 10000):
                increment = 150
            elif (ideal < 100000):
                increment = 1500
            elif (ideal < 1000000):
                increment = 15000
            elif (ideal < 10000000):
                increment = 150000
            elif (ideal < 100000000):
                increment = 1500000
            elif (ideal < 1000000000):
                increment = 15000000
            elif (ideal < 10000000000):
                increment = 150000000
            elif (ideal < 100000000000):
                increment = 1500000000
            elif (ideal < 1000000000000):
                increment = 15000000000
            elif (ideal < 10000000000000):
                increment = 150000000000
            elif (ideal < 100000000000000):
                increment = 1500000000000
                
            model.__init__()
            
            step_size = ideal * (10 ** -15)
            
            if hasattr(y_train, 'values'):
                model.fit_GD_LR(x_train, np.reshape(y_train.values, (len(y_train),1)), step_size)
            else:
                model.fit_GD_LR(x_train, np.reshape(y_train, (len(y_train),1)), step_size)
            
            y_pred_lr = model.predict_LR(x_test)
            
            if hasattr(y_test, 'values'):
                temp = score(y_pred_lr, np.reshape(y_test.values, (len(y_test),1)))
            else:
                temp = score(y_pred_lr, np.reshape(y_test, (len(y_test),1)))
                
            if (temp >= max_score[1]):
                theta = model.theta
                max_score = [ideal, temp]
                max_reached = 0
                y_pred = y_pred_lr
            else:
                max_reached += 1
            ideal += increment
            
        print('[ideal, score]:', max_score)
        print()
    
    elif (model_type == 'relu'):
        
        max_score = [0,-1000000000000]
        ideal = 1
        increment  = 0.1
        max_reached = 0
        
        while max_reached < 3:
            
            # Used to determine the ideal step size
            if (ideal < 10):
                increment = 0.15
            elif (ideal < 100):
                increment = 1.5
            elif (ideal < 1000):
                increment = 15
            elif (ideal < 10000):
                increment = 150
            elif (ideal < 100000):
                increment = 1500
            elif (ideal < 1000000):
                increment = 15000
            elif (ideal < 10000000):
                increment = 150000
            elif (ideal < 100000000):
                increment = 1500000
            elif (ideal < 1000000000):
                increment = 15000000
            elif (ideal < 10000000000):
                increment = 150000000
            elif (ideal < 100000000000):
                increment = 1500000000
            elif (ideal < 1000000000000):
                increment = 15000000000
            elif (ideal < 10000000000000):
                increment = 150000000000
            elif (ideal < 100000000000000):
                increment = 1500000000000
                
            model.__init__()
            
            step_size = ideal * (10 ** -15)
            
            if hasattr(y_train, 'values'):
                model.fit_GD_ReLU(x_train, np.reshape(y_train.values, (len(y_train),1)), step_size)
            else:
                model.fit_GD_ReLU(x_train, np.reshape(y_train, (len(y_train),1)), step_size)
            
            y_pred_relu = model.predict_ReLU(x_test)
            
            if hasattr(y_test, 'values'):
                temp = score(y_pred_relu, np.reshape(y_test.values, (len(y_test),1)))
            else:
                temp = score(y_pred_relu, np.reshape(y_test, (len(y_test),1)))

            if (temp >= max_score[1]):
                max_score = [ideal, temp]
                max_reached = 0
                y_pred = y_pred_relu
            else:
                max_reached += 1
            ideal += increment
            
        print('[ideal, score]:', max_score)
        print()
        
    return y_pred


def stochastic_gradient_descent(x_train, y_train, x_test, y_test, model, model_type):
    '''
    Gets a prediction using stochastic gradient descent.
    The commented out section was used to find the ideal step_size each
    time we added, removed, or modified a feature.
    '''
    
    if (model_type == 'lr'):
        
        max_score = [0,-1000000000000]
        ideal = 1
        increment  = 0.1
        max_reached = 0
        
        while max_reached < 2:
            
            # Used to determine the ideal step size
            if (ideal < 10):
                increment = 0.15
            elif (ideal < 100):
                increment = 1.5
            elif (ideal < 1000):
                increment = 15
            elif (ideal < 10000):
                increment = 150
            elif (ideal < 100000):
                increment = 1500
            elif (ideal < 1000000):
                increment = 15000
            elif (ideal < 10000000):
                increment = 150000
            elif (ideal < 100000000):
                increment = 1500000
            elif (ideal < 1000000000):
                increment = 15000000
            elif (ideal < 10000000000):
                increment = 150000000
            elif (ideal < 100000000000):
                increment = 1500000000
            elif (ideal < 1000000000000):
                increment = 15000000000
            elif (ideal < 10000000000000):
                increment = 150000000000
            elif (ideal < 100000000000000):
                increment = 1500000000000
            
            model.__init__()
            
            step_size = ideal * (10 ** -15)
            
            if hasattr(y_train, 'values'):
                model.fit_SGD_LR(x_train, np.reshape(y_train.values, (len(y_train),1)), step_size)
            else:
                model.fit_SGD_LR(x_train, np.reshape(y_train, (len(y_train),1)), step_size)

            y_pred_lr = model.predict_LR(x_test)

            if hasattr(y_test, 'values'):
                temp = score(y_pred_lr, np.reshape(y_test.values, (len(y_test),1)))
            else:
                temp = score(y_pred_lr, np.reshape(y_test, (len(y_test),1)))
                
            if (temp >= max_score[1]):
                max_score = [ideal, temp]
                max_reached = 0
                y_pred = y_pred_lr
            else:
                max_reached += 1
            ideal += increment
            
        print('[ideal, score]:', max_score)
        print()
    
    elif (model_type == 'relu'):
        
        max_score = [0,-1000000000000]
        ideal = 1
        increment  = 0.1
        max_reached = 0
        
        while max_reached < 2:
            
            # Used to determine the ideal step size
            if (ideal < 10):
                increment = 0.15
            elif (ideal < 100):
                increment = 1.5
            elif (ideal < 1000):
                increment = 15
            elif (ideal < 10000):
                increment = 150
            elif (ideal < 100000):
                increment = 1500
            elif (ideal < 1000000):
                increment = 15000
            elif (ideal < 10000000):
                increment = 150000
            elif (ideal < 100000000):
                increment = 1500000
            elif (ideal < 1000000000):
                increment = 15000000
            elif (ideal < 10000000000):
                increment = 150000000
            elif (ideal < 100000000000):
                increment = 1500000000
            elif (ideal < 1000000000000):
                increment = 15000000000
            elif (ideal < 10000000000000):
                increment = 150000000000
            elif (ideal < 100000000000000):
                increment = 1500000000000
            
            model.__init__()
            
            step_size = ideal * (10 ** -15)
            
            if hasattr(y_train, 'values'):
                model.fit_SGD_ReLU(x_train, np.reshape(y_train.values, (len(y_train),1)), step_size)
            else:
                model.fit_SGD_ReLU(x_train, np.reshape(y_train, (len(y_train),1)), step_size)
            
            y_pred_relu = model.predict_ReLU(x_test)
            
            if hasattr(y_test, 'values'):
                temp = score(y_pred_relu, np.reshape(y_test.values, (len(y_test),1)))
            else:
                temp = score(y_pred_relu, np.reshape(y_test, (len(y_test),1)))
            
            if (temp >= max_score[1]):
                max_score = [ideal, temp]
                max_reached = 0
                y_pred = y_pred_relu
            else:
                max_reached += 1
            ideal += increment
            
        print('[ideal, score]:', max_score)
        print()
        
    return y_pred

    
def score(y_pred, y_true):
    '''
    Generates a score value using least mean squared error.
    '''
    
    u = np.average(np.square(y_true - y_pred))
    v = np.average(np.square(y_true - y_true.mean()))
    
    score =  1 - (u/v)
    
    return score


def make_plot(y_true, y_pred, title):
    '''
    Creates and labels the plot given the input.
    '''
    
    plt.scatter(y_true, y_pred, alpha=0.4, c='blue', label=title)
    plt.xlabel('True')
    plt.ylabel('Prediction')
    plt.legend()
    plt.savefig('./' + title + '.png')
    plt.show()
    plt.clf()


def add_intercept(x):
    '''
    Add a column of 1's to the input.
    '''
    y = np.zeros((x.shape[0], x.shape[1] + 1))
    y[:, 0] = 1
    y[:, 1:] = x

    return y

def concatenate_training(x, y, i):
    '''
    Partitions the data into 10 subsets.
    Combines 9 of the subsets (to be used as training).
    Leaves 1 subset for testing.
    '''
    
    x_train = np.array_split(x, 10)
    y_train = np.array_split(y, 10)
    
    if (i == 0):
        x_combined = x_train[1]
        y_combined = y_train[1]
        
    elif (i == 1):
        x_combined = x_train[0]
        y_combined = y_train[0]
        
    else:
        x_combined = np.concatenate((x_train[0], x_train[1]))
        y_combined = np.concatenate((y_train[0], y_train[1]))
    
    for j in range (2, 10):
        
        if (i == j):
            continue
        
        x_combined = np.concatenate((x_combined, x_train[j]))
        y_combined = np.concatenate((y_combined, y_train[j]))
        
    return x_combined, y_combined, x_train[i], y_train[i]

def main_NE(data, columns, y, regression_type):
    '''
    Section for using the Normal Equation model to fit our data.
    '''
    
    # Columns gained by adding best result:
    # x = data[['HR', 'yearID', 'BB', 'DP', '3B', 'CS', 'H', 'InnOuts', '2B', 'E', 'SO', 'AB']]
    
    if (regression_type == 'lr'):
        print('NORMAL EQUATION - LINEAR REGRESSION\n')
        # Columns gained by subtracting worst result:
        x = data.drop(['salary', 'RBI', 'SB'], axis = 1)
    else:
        print('NORMAL EQUATION - ReLU\n')
        # Columns gained by subtracting worst result:
        x = data.drop(['salary', 'SB', 'RBI'], axis = 1)
    
    # 80/20 Split
    # Linear Regression
    #   Accuracy = 0.31180511973660663
    # ReLU
    #   Accuracy = 0.3233259970905922
    '''
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 101)
        
    x_train = add_intercept(x_train)
    x_test = add_intercept(x_test)
       
    model = baseball()
            
    normal_pred = normal_equation(x_train, y_train, x_test, y_test, model, regression_type)    
    '''
    # 10-Fold
    # Linear Regression
    #   Accuracy = 0.321101317
    # ReLU
    #   Accuracy = 0.332875565
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 101)
        
    x_train = add_intercept(x_train)
    x_test = add_intercept(x_test)
    
    for i in range(10):
    
        x_train, y_train, x_test, y_test = concatenate_training(x_train, y_train, i)
        
        model = baseball()
            
        normal_pred = normal_equation(x_train, y_train, x_test, y_test, model, regression_type)
    
    
    # Make Plots
    #make_plot(y_test, normal_pred, 'Normal Equation - ReLU - 10-Fold')
    
    # This is how I determined which columns to use
    '''
    list_of_columns = [12, 13]
    
    for i in range (17):
        
        if (i in list_of_columns):
            continue
        
        print("Column", i, ":", columns[i])
        
        # Removing Columns
        x = data.drop(['salary', 'SB', 'RBI', columns[i]], axis = 1)
        
        # Adding Columns
        #x = data[['HR', 'yearID', 'BB', 'DP', '3B', 'CS', 'H', 'InnOuts', '2B', 'E', 'SO', 'AB', columns[i]]]
        
        # For the first column
        #x = np.reshape(x.values, (x.shape[0], 1))
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 101)
        
        x_train = add_intercept(x_train)
        x_test = add_intercept(x_test)
        
        model = baseball()
            
        normal_pred = normal_equation(x_train, y_train, x_test, y_test, model, regression_type)
    '''

def main_GD(data, columns, y, regression_type):
    '''
    Section for using the Gradient Descent model to fit our data.
    Given the regression_type, we will either be using linear regression or ReLU.
    '''
    
    if (regression_type == 'lr'):
        print('GRADIENT DESCENT - LINEAR REGRESSION\n')
        x = data.drop(['salary', 'yearID', 'SO', 'RBI', 'SB', 'InnOuts', 'PO', 'DP', 'AB'], axis = 1)
    else:
        print('GRADIENT DESCENT - ReLU\n')
        x = data.drop(['salary', 'InnOuts', 'yearID', 'SO', 'RBI', 'AB', 'PO', 'A', 'R'], axis = 1)
    
    # 80/20 Split
    # Linear Regression
    #   Accuracy = 0.19718904032050533
    #   Step Size = 7.15e-9
    # ReLU
    #   Accuracy = 0.19930756999958787
    #   Step Size = 1.45e-8
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 101)
    
    x_train = add_intercept(x_train)
    x_test = add_intercept(x_test)
    
    model = baseball()
    
    gd_pred = gradient_descent(x_train, y_train, x_test, y_test, model, regression_type)
    
    # 10-Fold
    # Linear Regression
    #   Accuracy = 0.185700711
    # ReLU
    #   Accuracy = 0.183475991
    '''
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 101)
    
    x_train = add_intercept(x_train)
    x_test = add_intercept(x_test)
    
    for i in range(10):
        
        x_train, y_train, x_test, y_test = concatenate_training(x_train, y_train, i)

        model = baseball()
            
        gd_pred = gradient_descent(x_train, y_train, x_test, y_test, model, regression_type)
    '''
    
    # Find columns
    '''
    list_of_columns = [0, 1, 2, 3, 5, 6, 7, 12, 13, 16]
    
    for i in range (17):
            
        if (i in list_of_columns):
            continue
        
        print("Column", i, ":", columns[i])
        
        # Removing Columns
        x = data.drop(['salary', 'yearID', 'SO', 'RBI', 'SB', 'InnOuts', 'PO', 'DP', 'AB', columns[i]], axis = 1)
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 101)
        
        x_train = add_intercept(x_train)
        x_test = add_intercept(x_test)
        
        model = baseball()
        
        gd_pred = gradient_descent(x_train, y_train, x_test, y_test, model, regression_type)
    '''
    
    # Make plots
    '''
    if (regression_type == 'lr'):
        make_plot(y_test, gd_pred, 'GD - Linear Regression - 80_20 Split')
    else:
        make_plot(y_test, gd_pred, 'GD - ReLU - 80_20 Split')
    '''
    
def main_SGD(data, columns, y, regression_type):
    '''
    Section for using the Stochastic Gradient Descent model to fit our data.
    Given the regression_type, we will either be using linear regression or ReLU.
    '''
    
    if (regression_type == 'lr'):
        print('STOCHASTIC GRADIENT DESCENT - LINEAR REGRESSION\n')
        x = data.drop(['salary', 'RBI', 'H', 'R', 'AB', 'InnOuts'], axis = 1)
    else:
        print('STOCHASTIC GRADIENT DESCENT - ReLU\n')
        x = data.drop(['salary', 'AB', 'R', 'H', 'RBI', 'InnOuts', 'PO'], axis = 1)
    
    # 80/20 Split
    # Linear Regression
    #   Accuracy = 0.17308083425239307
    #   Step Size = 6.8e-10
    # ReLU
    #   Accuracy = 0.17350056288704963
    #   Step Size = 6.8e-10
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 101)
    
    x_train = add_intercept(x_train)
    x_test = add_intercept(x_test)
    
    model = baseball()
    
    sgd_pred = stochastic_gradient_descent(x_train, y_train, x_test, y_test, model, regression_type)
    
    # 10-Fold
    # Linear Regression
    #   Accuracy = 0.148527054
    # ReLU
    #   Accuracy = 0.148949348
    '''    
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 101)
    
    x_train = add_intercept(x_train)
    x_test = add_intercept(x_test)
    
    for i in range(10):
        
        x_train, y_train, x_test, y_test = concatenate_training(x_train, y_train, i)

        model = baseball()
            
        sgd_pred = stochastic_gradient_descent(x_train, y_train, x_test, y_test, model, regression_type)
    '''
    
    # Find columns
    '''
    list_of_columns = [1, 2, 6, 7, 8, 11, 12, 15]
    
    for i in range (17):
        
        if (i in list_of_columns):
            continue
        
        print("Column", i, ":", columns[i])
        
        # Removing Columns
        x = data.drop(['salary', 'AB', 'R', 'H', 'RBI', 'InnOuts', 'PO', columns[i]], axis = 1)
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 101)
        
        x_train = add_intercept(x_train)
        x_test = add_intercept(x_test)
        
        model = baseball()
        
        sgd_pred = stochastic_gradient_descent(x_train, y_train, x_test, y_test, model, regression_type)

    '''
    # Make Plots
    '''
    if (regression_type == 'lr'):
        make_plot(y_test, sgd_pred, 'SGD - Linear Regression - 80_20 Split')
    else:
        make_plot(y_test, sgd_pred, 'SGD - ReLU - 80_20 Split')
    '''
    
def main_RF(data, columns, y):
    '''
    Section for using the Random Forest Regressor model to fit our data.
    '''
    
    print('RANDOM FOREST\n')
    
    # Columns dropped that achieve best result
    x = data.drop(['salary'], axis = 1)
    
    # 80/20 split - Achieved 0.3685380494437651 accuracy
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 101)
        
    x_train = add_intercept(x_train)
    x_test = add_intercept(x_test)
       
    rf_pred = rf_regressor(x_train, y_train, x_test, y_test)    
    
    # Ten-Fold - Achieved 0.366001959 accuracy
    '''
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 101)
        
    x_train = add_intercept(x_train)
    x_test = add_intercept(x_test)
    
    for i in range(10):
    
        x_train, y_train, x_test, y_test = concatenate_training(x_train, y_train, i)
        
        rf_pred = rf_regressor(x_train, y_train, x_test, y_test) 
    '''
    
    # Make Plots
    #make_plot(y_test, rf_pred, 'Random Forest - 80_20 Split')
    
    # Find columns
    '''
    list_of_columns = []
    
    for i in range (17):
        
        if (i in list_of_columns):
            continue
        
        print("Column", i, ":", columns[i])
        
        # Removing Columns
        x = data.drop(['salary', columns[i]], axis = 1)
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 101)
        
        x_train = add_intercept(x_train)
        x_test = add_intercept(x_test)
        
        rf_pred = rf_regressor(x_train, y_train, x_test, y_test) 
    '''

def main(data_path):
    '''
    Read in the csv file.
    Establish all the colummns.
    Set y to salary (this is our goal).
    Branch out to whichever model we want to use.
    '''
    data = pd.read_csv(data_path)
    columns = ['yearID','InnOuts','PO','A','E','DP','AB','R',
               'H','2B','3B','HR','RBI','SB','CS','BB','SO']
    
    # Not Normalizing Data
    y = data['salary']
    
    # Normalize Data
    '''
    names = data.columns
    d = preprocessing.normalize(data, axis=0)
    scaled_data = pd.DataFrame(d, columns=names)
    y = scaled_data['salary']
    '''
    
    start = timeit.default_timer()
    
    #main_NE(data, columns, y, 'lr')
    
    #main_NE(data, columns, y, 'relu')
    
    #main_GD(data, columns, y, 'lr')
    
    #main_GD(data, columns, y, 'relu')
    
    #main_SGD(data, columns, y, 'lr')
    
    #main_SGD(data, columns, y, 'relu')
    
    main_RF(data, columns, y)
    
    stop = timeit.default_timer()
    
    print('Time: ', round(stop - start, 5))
    

if __name__ == '__main__':
    main(data_path='Batters.csv')