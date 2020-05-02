import json
import csv
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', 500)
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import geopandas as gpd
import shapely
from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon
from shapely.affinity import scale
import matplotlib.pyplot as plt

import glob
import os
import datetime

#From https://github.com/aaronkandel/Markov-Chain-Modeling/blob/master/markovChainFunction.m

def MarkovChainFunction(data, bins):
    """ Data should be numpy array; bins is an integer """
    
    #Normalize data
    datMin = min(data)
    datMax = max(data)
    datNorm = (data - datMin)/(datMax - datMin)
    
    # Create Markov Transition Table:
    mesh = np.linspace(0, 1, bins)
    meshReal = (mesh*(datMax - datMin) + datMin)
    dmesh = mesh[1] - mesh[0]
    dmeshReal = meshReal[1] - meshReal[0]
    
    markovArray = np.zeros((len(mesh), len(mesh)))
    cumMarkovArray = np.zeros((len(mesh), len(mesh)))
    
    # Populate Markov Transition Table:
    for i in range(1,(len(data)-1)):
        datNow = datNorm[i]
        datBefore = datNorm[i-1]

        dn = np.floor(datNow / dmesh) # Get index....TODO: DO WE NOT WANT TO ROUND DOWN**? Ask Aaron
        db = np.floor(datBefore / dmesh) # Get index

        markovArray[int(db), int(dn)] = markovArray[int(db), int(dn)] + 1;  #Update MTT
    
    # Transform data in transition table to probability:
    markovArray = markovArray/np.sum(markovArray, axis=1, keepdims = True) #? from https://stackoverflow.com/questions/16202348/numpy-divide-row-by-row-sum  
    cumMarkovArray = np.cumsum(markovArray, axis=1)
    
    # Eliminate potential NaNs from potential /0:
    ind = np.isnan(markovArray)
    markovArray[ind] = 0
    cumMarkovArray[ind] = 0
    
    return markovArray, cumMarkovArray, datMin, datMax, dmeshReal


def MarkovSim(data, cumMarkovArray, datMin, datMax, dmeshReal):
    #Simluate Markov Chain Model
    datSim = np.zeros_like(data)
    datSim[0] = np.floor(data[0]/dmeshReal) # Set same initial condition

    np.random.seed(1)
    rands = np.random.random_sample(len(data))

    for i in range(len(data)-1): #length(data)

        # Find current row on MTT:
        curState = int(datSim[i])

        # Create random transition:
        r = rands[i]
        check = np.where(cumMarkovArray[curState, :] > r)

        # Transition to next state:
        nextState = check[0][0]
        datSim[i+1] = nextState

    # Un-Normalize Simulated Data:
    datSim = datSim/max(datSim)
    datSim = datSim*(datMax - datMin) + datMin + dmeshReal*.5
    
    return datSim



def MarkovChainFunction_24hrs(data, bins, hrs=24):
    """ Data should be numpy array; bins is an integer """
    
    #Normalize data
    datMin = min(data)
    datMax = max(data)
    datNorm = (data - datMin)/(datMax - datMin)
    
    # Create Markov Transition Table:
    mesh = np.linspace(0, 1, bins)
    meshReal = (mesh*(datMax - datMin) + datMin)
    dmesh = mesh[1] - mesh[0]
    dmeshReal = meshReal[1] - meshReal[0]
    
    markovArray = np.zeros((hrs, len(mesh), len(mesh)))
    cumMarkovArray = np.zeros((hrs, len(mesh), len(mesh)))
    
    # Populate Markov Transition Table:
    hr = 0
    for i in range(1,(len(data)-1)):
        datNow = datNorm[i]
        datBefore = datNorm[i-1]

        dn = np.floor(datNow / dmesh) # Get index
        db = np.floor(datBefore / dmesh) # Get index

        markovArray[hr, int(db), int(dn)] = markovArray[hr, int(db), int(dn)] + 1#Update MTT
        
        if hr < hrs-1:
            hr += 1
        else:
            hr = 0
    
    # Transform data in transition table to probability:
    for i in range(markovArray.shape[0]):
        markovArray[i,:,:] = markovArray[i,:,:]/np.sum(markovArray[i,:,:], axis=1, keepdims=True)
        cumMarkovArray[i,:,:] = np.cumsum(markovArray[i,:,:], axis=1)    

    # Eliminate potential NaNs from potential /0:
    ind = np.isnan(markovArray)
    markovArray[ind] = 0
    cumMarkovArray[ind] = 0
    
    return markovArray, cumMarkovArray, datMin, datMax, dmeshReal


def MarkovSim_24hrs(data, cumMarkovArray, datMin, datMax, dmeshReal, hrs=24, rand_seed=1):
    #Simluate Markov Chain Model
    datSim = np.zeros_like(data)
    datSim[0] = np.floor(data[0]/dmeshReal) # Set same initial condition // TODO: CHECK DMESHREAL

    np.random.seed(rand_seed)
    rands = np.random.random_sample(len(data))

    hr = 0
    for i in range(len(data)-1): #length(data)

        # Find current row on MTT:
        curState = int(datSim[i])

        # Create random transition:
        r = rands[i]
        check = np.where(cumMarkovArray[hr, curState, :] > r)
        
        #Dealing with case where random transition leads into a dead end
        while len(check[0])==0:
            curState -= 1 #this is ~conservative--RESETS TO LOWER VALUE. COULD DO OPPOSITE
            check = np.where(cumMarkovArray[hr, curState, :] > r)

        # Transition to next state:
        nextState = check[0][0]
        datSim[i+1] = nextState
        
        if hr < hrs-1:
            hr += 1
        else:
            hr = 0

    # Un-Normalize Simulated Data:
    datSim = datSim/max(datSim)
    datSim = datSim*(datMax - datMin) + datMin + dmeshReal*.5
    
    return datSim