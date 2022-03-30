### Create a simple additive model for SigProcOpenPython

# The model takes two input values as inputs and puts their sum to the output

import dill
import os

print(f"\n*** running {__file__}\n")

# save models in the same directory where create_spop_predict.py present
ABSPATH = os.path.abspath(__file__)
DNAME = os.path.dirname(ABSPATH)
# move to models directory
os.chdir(DNAME)
os.chdir('models')
CNAME = os.getcwd()


##########################################################################
### create a simple function which returns the negative  value
MODELFILENAME = 'negate.spop_predict'
print(f"*** preparing {MODELFILENAME}")
def spop_predict(X): # do not change the function name
    return -X
dill.dump(spop_predict, open(MODELFILENAME, 'wb'))
print(f'*** {MODELFILENAME} dill file saved in the folder {CNAME}\n')
##########################################################################


##########################################################################
### create a simple function which returns additive value of two functions
MODELFILENAME = 'addtwo.spop_predict'
print(f"*** preparing {MODELFILENAME}")
def spop_predict(X): # do not change the function name
    # X[0] and X[1] must be available, otherwise exception raised (no error checking here)
    return X[0] + X[1]
dill.dump(spop_predict, open(MODELFILENAME, 'wb'))
print(f'*** {MODELFILENAME} dill file saved in the folder {CNAME}\n')
##########################################################################


# save created spop_predict function to disk
