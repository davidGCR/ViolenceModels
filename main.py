#import numpy as nump

# load additional module
import pickle

# define a list of places
# hist = [0.12452, 0.12452, 0.12452, 0.12452]
data_file = 'history.txt'
# with open(data_file, 'wb') as filehandle:
#     # store the data as binary data stream
#     pickle.dump(hist, filehandle)

with open(data_file, 'rb') as filehandle:
    # read the data as binary data stream
    hist = pickle.load(filehandle)
print(type(hist))
print(hist)