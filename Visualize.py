# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 21:57:07 2018

@author: Mateusz
"""
import pylab
import os
from matplotlib import pyplot as plt
plt.ioff()
import imageio
import subprocess
from random import shuffle
import numpy as np
from math import sqrt 
import seaborn as sn
import pandas as pd


class Visualize():
    def __init__(self, path):
        '''class to visualize training results'''
        self.path = path

    def printProgressBar(self,iteration, total, prefix = '', suffix = '', decimals = 1, length = 70, fill = '█'):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    
        if iteration == total: 
            print()
            
    def draw(self, histWeights, keepPictures = True, gif=True):
        filenames = []
        for i in range(len(histWeights)):
            self.printProgressBar(i,len(histWeights)+1)
            f = os.path.join(self.path,str(i)+'.png')
            self.decision_regions(histWeights[i],f)
            filenames.append(f)  
            plt.close()
            
        gif_path = os.path.join(self.path,'learning_process.gif')
        if (gif == True):
            images = []
            for filename in filenames:
                images.append(imageio.imread(filename))
            imageio.mimsave(gif_path, images,fps = int(sqrt(len(histWeights))/2))
            
        if (not keepPictures):
            for filename in filenames:
                os.remove(filename)
        subprocess.call(gif_path,shell=True)
            
    def confusion_matrix(self, acc, task, ekspNr):
        array = [[acc['TP']/ekspNr, acc['TN']/ekspNr], [acc['FP']/ekspNr, acc['FN']/ekspNr] ]
        df_cm = pd.DataFrame(array, index = [i for i in ["TRUE", "FALSE"]],
                          columns = [i for i in ["POSITIVES","NEGATIVES"]])
        plt.figure()
        sn.heatmap(df_cm, annot=True, fmt='g')
        plt.savefig('confusion_matrix_'+ str(task) +'.png')
        
    def decision_regions(self,weights,f):
        fig,ax = plt.subplots()
        ax.scatter([0,0,1], [0,1,0], label = '1', facecolors='none', edgecolors='b')
        ax.scatter([1], [1], label = '1', facecolors='none', edgecolors='r')
        ax.text(0,0, '(0,0)')
        ax.text(1,0, '(1,0)')
        ax.text(0,1, '(0,1)')
        ax.text(1,1, '(1,1)')
        x = np.arange(0,1.2,0.01)
        y = -1/weights[2]*(weights[0] + weights[1]*x)
        ax.plot(x,y,c='g')
        ax.set_title('Podział przestrzeni po wyuczeniu')
        plt.legend(['linia podzialu', '0','1'])
        pylab.savefig(f)
        
