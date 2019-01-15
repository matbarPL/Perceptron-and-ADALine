# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 10:27:37 2018

@author: Mateusz
"""
import numpy as np
from matplotlib import pyplot as  plt
from inspect import getsourcefile
from random import shuffle
import os
from math import sqrt
from Visualize import *


class Neuron():
    def __init__(self, learn_rate, low, high, activation, extSet = 0 ):
        '''class representing single artificial neuron
        attributes:
            -path - path where file is stored
            -alpha - learning rate
            -weights - weights of a single neuron
            -low, high - range for distribution from which weights are drawn
            In this case there are 3 weights:
                -w_0 weight for bias 
                -w_1 weight for x_1
                -w_2 weight for x_2
            activation - activation function either bipolar or binar'''
        self.path= os.path.split(os.path.abspath(__file__))[0]
        os.chdir(self.path)
        self.learn_rate = learn_rate 
        self.low = low
        self.high = high 
        self.extSet = extSet
        self.weights = self.read_weights()
        self.set_activation_function(activation)
        
        
    def set_activation_function(self, activation):
        self.activation = {'bipolar':self.bipolar, 'binar':self.binar}[activation]
        self.set_sets(self.extSet)
        
    def write_to_file(self, itemsNr, name):
        pairs = []
        itemsNr = int(itemsNr)
        for i in range(itemsNr//2):
                to_write = list(map(lambda x: round(x,2), np.random.uniform(low = 0, size = 2) ))
                res = self.get_res(to_write)
                x1, x2 = map(str,to_write)
                pairs.append([x1,x2,res])
        for i in range(itemsNr//4):
                to_write = [0,0]
                res = self.get_res(to_write)
                x1, x2 = map(str,to_write)
                pairs.append([x1,x2,res])
        for i in range(itemsNr//4):
                to_write = [1,1]
                res = self.get_res(to_write)
                x1, x2 = map(str,to_write)
                pairs.append([x1,x2,res])
        shuffle(pairs)
        with open(name, 'w') as f:
            for pair in pairs:
                f.write(';'.join(pair)+'\n')


    def set_sets(self, extSet):
        if extSet == 0:
            print ('training and testing for not extended sets')
            self.train_set = self.read_set('train_set.txt')
            self.test_set = self.read_set('test_set.txt')
        else:
            self.write_to_file(0.8*extSet, 'train_set_ext.txt')
            self.write_to_file(0.2*extSet, 'test_set_ext.txt')
            self.train_set = self.read_set('train_set_ext.txt')
            self.test_set = self.read_set('test_set_ext.txt')
    
    def read_weights(self):
        '''function for reading weights from file'''
        with open('weights.txt', 'r') as f:
            return np.array(list(map(float,f.readlines()[0].split(';'))))
        
        
    def bipolar(self, net):
        '''bipolar function, if total actiavtion is greater than 0 then 1 is returned; 0 otherwise''' 
        return 1 if net >= 0 else -1
    
    
    def binar(self, net):
        '''binar function, if total actiavtion is greater than 0 then 1 is returned; -1 otherwise''' 
        return 1 if net >= 0 else 0
    
    
    def train(self, trn_way, eps = 0.4):
        '''function for single neuron training
            parameters:
                - trn_way: ['adaline', 'perceptron']
            perceptron learning rule:
                w(t+1) = w(t) + alpha*delta*x_i;
                delta = (exp_val - clc_val);
                clc_val = f(net);
                net = x_i*weights.T
            adaline learning rule:
                w(t+1) = w(t) + alpha*delta*x_i;
                delta = (expected_value - calculated_value)
                clc_val = f(net);
                net = x_i*weights.T'''
        if trn_way not in ['adaline', 'perceptron']:
            print ('Training way must be set as either adaline or perceptron')
            return
        elif trn_way=='adaline':
            return self.train_adaline(eps)
        else:
           return self.train_perceptron()     

    def save_weights(self):
        with open('weights.txt', 'w') as f:
            f.write(';'.join(list(map(str,self.weights))))
        
    def train_perceptron(self):
        '''function for training single perceptron'''
        self.weights = np.random.uniform(low = self.low, high = self.high, size = (1,3))[0]
        changed = True
        net = 0
        epochs_nr = 0
        hist = []
        
        while changed :
            epochs_nr+=1
            ctr_changed = 0
            for item in self.train_set:
                old_weights = np.copy(self.weights)
                hist.append(old_weights)
                net = np.dot(item[:3],np.transpose(self.weights[:3]))
                y = self.activation(net)
                if y != item[-1]:
                    self.update_weights(item, y)
                    ctr_changed +=1
                
            if ctr_changed == 0:
                changed = False
        
        self.hist_weights = np.copy(hist)
        self.save_weights()
        return epochs_nr
    
    def update_weights(self, x_vec, y):
        '''update weights for percepton algorithm'''
        for i in range(len(self.weights)):
            delta = self.learn_rate*x_vec[i]*(x_vec[-1] - y)
            self.weights[i] += delta
        
    def test(self):
        '''test trained artificial neural neuron'''
        acc = dict(zip(['TP','TN','FP','FN'], [0]*4))
        for vec in self.test_set:
            net = vec[:3].dot(np.transpose(self.weights[:3]))
            y = self.activation(net)
            exp_res = vec[3]
            if y == 1:
                if exp_res==1:
                    acc['TP'] +=1
                else:
                    acc['FP'] +=1
            else:
                if exp_res == 0 or exp_res == -1:
                    acc['TN'] +=1
                else:
                    acc['FN'] +=1
                    
        self.acc = acc
        return acc
            
            
    def read_set(self, name):
        vectors = []
        with open(name, 'r') as f:
            for line in f.readlines():
                splt_line = line.split(';')
                vectors.append(np.array([1,splt_line[0], splt_line[1], splt_line[2]], dtype= float))
                
        return vectors
            

    def get_res(self, to_write):
        if self.activation == self.binar:
            if (to_write[0] == 1 and to_write[1] == 1):
                return str(1)
            else:
                return str(0)    
        else:
            if (to_write[0] == 1 and to_write[1] == 1):
                return str(1)
            else:
                return str(-1)  
            
                
    def train_adaline(self, eps):
        '''Adaptive Linear search training'''
        def change_weights(E_k, x_k):
            for i in range(len(self.weights)):
                self.weights[i] = self.weights[i] - 2*self.learn_rate*E_k*x_k[i]
        
        self.weights = np.random.uniform(low = self.low, high = self.high, size = (1,3))[0] 
        L = len(self.train_set)
        hist = []
        epochs_nr = 0
        LMS_tot = 1000
        
        while LMS_tot/L > eps:
            epochs_nr += 1
            LMS_old = LMS_tot
            LMS_tot = 0
            for item in self.train_set:
                w_t = np.transpose(self.weights)
                E_k = (item[-1] - np.dot(w_t,item[:3]))**2
                LMS_tot += E_k
                change_weights(E_k, item[:3])
            if LMS_old < LMS_tot: #if the LMS starts increasing sample weights once again 
                self.weights = np.random.uniform(low = self.low, high = self.high, size = (1,3))[0]
            hist.append(self.weights)
            print (LMS_tot/L)
        print('done')
        self.hist_weights = np.copy(hist)
        return epochs_nr
        
        
    def test_single(self, x1, x2):
        net = self.weights[0] + self.weights[1]*x1 + self.weights[1]*x2
        y = self.activation(net)
        return y
                
    def visualize(self, option):
        vis = Visualize(self.path)
        if option == 'decision_regions':
            vis.decision_regions(self.weights,'decision_regions_bipolar')
            plt.show()
        elif option == 'draw':
            vis.draw(self.hist_weights,'learning_process')
        elif option =='confusion_matrix':
            vis.confusion_matrix(self.acc)
            
    
if __name__ == '__main__':
    neuron = Neuron(0.00001, -1, 1, 'bipolar',100)
    #neuron.train('adaline')
    print (neuron.test_single(0,0))

  