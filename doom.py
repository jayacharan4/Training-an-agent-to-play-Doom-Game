#Importing all neccessary packages

import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
import math
import os 
import sys
import timeit
from vizdoom import *

def get_input_shape(Image,Filter,Stride):
  layer1 = math.ceil(((Image - Filter + 1) / Stride))
  out1 = math.ceil((layer1 / Stride))
    
  layer2 = math.ceil(((o1 - Filter + 1) / Stride))
  out2 = math.ceil((layer2 / Stride))
    
  layer3 = math.ceil(((o2 - Filter + 1) / Stride))
  out3 = math.ceil((layer3  / Stride))
  return int(o3)
