#import numpy as nump

# load additional module
# import pickle

# # define a list of places
# # hist = [0.12452, 0.12452, 0.12452, 0.12452]
# data_file = 'history.txt'
# # with open(data_file, 'wb') as filehandle:
# #     # store the data as binary data stream
# #     pickle.dump(hist, filehandle)

# with open(data_file, 'rb') as filehandle:
#     # read the data as binary data stream
#     hist = pickle.load(filehandle)
# print(type(hist))
# print(hist)


import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/MyCode')
from ViolenceAlexNet import *
from ViolenceDataset import *
# from test.util import *
# from ViolenceModels.freezeModel import set_parameter_requires_grad 
model = ViolenceModel(2)