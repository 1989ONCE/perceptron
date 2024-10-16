import random
from sklearn.model_selection import train_test_split
import numpy as np


class perceptron():
    def __init__(self):
        self.train_acc = 0.0
        self.test_acc = 0.0
        self.train_result = []
        self.test_result = []
        self.weight = np.array([])

        # member that stores the best experiment result
        self.best_weight = []
        self.best_epoch = 0
        self.best_train_acc = 0.0
        self.best_test_acc = 0.0
        self.class_list = []

    def init_weight(self, dim):
        self.weight = np.array([round(random.uniform(-1, 1), 2) for i in range(dim+1)])
        self.best_weight = self.weight
        print('Initial Weight:', self.weight)

    def update_weight(self, lr, prev_w, expected_y, vj, xi):
        pred_y = self.sgn(vj)

        if pred_y == expected_y:
            return prev_w
        elif pred_y == -1 and expected_y == 1:
            return prev_w + lr * xi
        elif pred_y == 1 and expected_y == -1:
            return prev_w - lr * xi

    def split_data(self, dataset):
        samples = len(dataset)
        x = [ [-1] + list(dataset[i][0]) for i in range(samples)]
        y = [dataset[i][1] for i in range(samples)]
        self.class_list = list(set(y))

        X_train, x_test, Y_train, y_test = train_test_split(x, y, train_size=2/3, test_size=1/3, shuffle=True)
        return X_train, x_test, Y_train, y_test


    def train(self, X_train, Y_train, lr, epoch_limit):
        self.train_result = []
        epoch = 0
        Y_train = self.unify_class(Y_train)

        while epoch < epoch_limit: # 到「設定最大的迭代次數」或是「完全線型分割了training data」就停止
            epoch += 1
            pred_y = []
            for i in range(len(X_train)):
                xi = np.array(X_train[i])
                yi = Y_train[i]
                vj = np.dot(self.weight.T, xi)
                pred_y.append(self.sgn(vj))
                self.weight = self.update_weight(lr, self.weight, yi, vj, xi)
            
            self.train_acc = self.evaluate(Y_train, pred_y)
            self.train_result.append([self.weight.copy(), self.train_acc])

            if self.train_acc > self.best_train_acc:
                self.best_train_acc = self.train_acc
                self.best_weight = self.weight.copy()
                self.best_epoch = epoch
            print('=========== Epoch', epoch, '==========\nTrain Accuracy:', self.train_acc)
            
        print('Best Weight:', self.best_weight)
        print('Best Epoch:', self.best_epoch)
        print('Best Train Accuracy:', self.best_train_acc)

    def test(self, x_test, y_test):
        self.test_result = []
        epoch = 0
        y_test = self.unify_class(y_test)
        
        while epoch < len(self.train_result):
            
            pred_y = []
            for i in range(len(x_test)):
                xi = np.array(x_test[i])
                vj = np.dot(self.train_result[epoch][0], xi)
                pred_y.append(self.sgn(vj))
            epoch += 1
            self.test_acc = self.evaluate(y_test, pred_y)
            self.test_result.append([self.weight.copy(), self.test_acc])
            print('=========== Epoch', epoch, '==========\nTest Accuracy:', self.test_acc)
            if self.test_acc > self.best_test_acc:
                    self.best_test_acc = self.test_acc
            

    def sgn(self, vj): # 硬限制函數
        if vj >= 0:
            return 1
        else:
            return -1
        
    def evaluate(self, y, pred_y):
        correct = 0
        y = self.unify_class(y)
        
        for i in range(len(y)):
            if pred_y[i] == y[i]:
                correct += 1
        return round(correct / len(y), 3)
    
    def unify_class(self, Y):
        self.class_list = list(set(Y)) 
        return [1 if y == self.class_list[0] else -1 for y in Y]
    
    def get_best_weight(self):
        return self.best_weight
    
    def get_best_epoch(self):
        return self.best_epoch
    
    def get_best_train_acc(self):
        return self.best_train_acc
    
    def get_all_train_result(self):
        return self.train_result

    def get_all_test_result(self):
        return self.test_result
