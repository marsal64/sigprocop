# analysis of measurements data
#
# - switches to ~/optiguard/models
# - takes data from DataRaw.csv
# - analyzes data and prints models results for mm and offsetx
# - creates final mm model and saves to model_mm.ml
# - creates final offsetx model and saves to model_offsetx.ml

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
import sys
import pickle
import argparse
import re
from sklearn.tree import _tree
import glob
import matplotlib

# arguments
parser = argparse.ArgumentParser(description='OptgiGuartdML create model')

# create DataRaw.csv from non-mm raw data
parser.add_argument('--fromraw', action='store_true', default=True, help='Create RawData.csv from non-mm raw data subdirectories')
parser.add_argument('--rawsubdir', default='indata', help='Subdirectory under rundir with raw data subfolders')
parser.add_argument('--offsubdir', default="^(.+?) .*", help='Regex to match subdirectory name')
parser.add_argument('--chanfname', default=".*os_[0|](.*?)-.*?", help='Regex to match channel in file name in subdirectory')
parser.add_argument('--startmm', type=float, default=-35, help='Starting value of mm for the first position')
parser.add_argument('--speed', type=float, default=0.5, help='Speed in mm/s')


parser.add_argument('--testsplit', type=float, default=0.3, help='Percentage of test, 0.3 means 30 percent of data used for testing')
parser.add_argument('--randomseed', default=1, help='Random seed')
parser.add_argument('--debug', action='store_true', default=False, help='If present, does some more detailed profiling and graphs')
parser.add_argument('--inputdata', default='DataRaw.csv', help='File name of input data')
parser.add_argument('--rundir', default='/home/ml/optiguardml/models', help='If present, does cd to the given directory. If not present, it runs in the directory when the script is present. ')
parser.add_argument('--modeltype', default='DecisionTreeRegressor', help='Model type. Try slower but more precise alternative: ExtraTreesRegressor')
parser.add_argument('--from_mm', default=-35, help='Drop from mm from input data')
parser.add_argument('--to_mm', default=35, help='Drop to mm from input data')

parser.add_argument('--filemodel_1', default='model_mm.ml', help='Filename for model 1 (mm)')
parser.add_argument('--filemodel_2', default='model_offsetx_abs.ml', help='Filename for model 2 (offsetx_abs)')

# development related only
parser.add_argument("--mode", default='client', help='Internal parameter used only by pydev')
parser.add_argument("--port", default=36829, help='Internal parameter used only by pydev')

# collect parameters from line arguments to F namespace object
F = parser.parse_args()

### cd to the working directory
if F.rundir == '':
    # switch to the directory where the file is present
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
else:
    # switch to the directory as given in parameter '--rundir'
    try:
        os.chdir(F.rundir)
    except Exception as e:
        print(f"cannot switch to '{F.rundir}', exiting. Exception:\n{e}")
        sys.exit(1)

# helper function to print rules
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)


# initial convenience settings
np.seterr('raise')                              # avoid warnings (raise error)
pd.options.mode.chained_assignment = None       # avoid pandas warning
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
np.set_printoptions(threshold=sys.maxsize) # tune numpy print

np.random.seed(F.randomseed)     # reset seed

### prepare input file from subdirectories?
if F.fromraw:

    # go to subdir with input data
    try:
        os.chdir(F.rawsubdir)
    except Exception as e:
        print(f"Cannot move to subdirectory {F.rawsubdir}, exiting. Exception:\n{e}")
        sys.exit(1)

    # list subdirectories subdirectories with offset in the subdirectory name
    ofdirs = next(os.walk('.'))[1]

    # browse all subdirectories and read files
    for subdir in ofdirs:
        try:
            offset = re.search(F.offsubdir, subdir).group(1)
        except Exception as e:
            print (f'Not valid name of subdirectory for measurements with given offset: {subdir}')
            # not valid subdir
            continue

        # offset to number
        offset = eval(offset)

        # go to subdirectory with data
        os.chdir(subdir)

        # files with data
        filelist = glob.glob('*')

        # browse subdirectories with offset files
        listfiles = glob.glob('*.csv')

        # initialize dataframe for one directory
        oneofdf = pd.DataFrame()

        for numfilename, filename in enumerate(listfiles):
            try:
                channel = re.search(F.chanfname, filename).group(1)
            except Exception as e:
                print(f'Not valid channel name of file with measurements for given channel: {filename}')
                # not valid subdir
                continue

            # process one file
            rf = pd.read_csv(filename, skiprows=[1], sep=';')
            print(f"processing offset {offset}, channel {channel}")

            # merge to oneofdf
            oneofdf[f'ch{channel}'] = rf[' avg']

            # when first, process the rest
            if numfilename == 0:
                rf['timestamp'] = pd.to_datetime(rf.time, format='%d-%m-%Y %H:%M:%S.%f ')
                rf['timediff'] = rf.timestamp - rf.timestamp.shift(1)
                rf['mm'] = F.startmm
                for i in range(1, rf.index.size):
                    rf.loc[i, 'mm'] = rf.loc[i - 1, 'mm'] + rf.loc[i].timediff.total_seconds() * F.speed
                oneofdf['mm'] = rf['mm']

                print(f"max mm: {oneofdf.iloc[-1:]}")

        oneofdf['offsetx'] = offset
        # add offset to rawdata
        oneofdf = oneofdf.filter(['mm', 'ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'offsetx'], axis=1)

        try:
            rawdata = rawdata.append(oneofdf, ignore_index=True)
        except Exception as e:
            rawdata = oneofdf

        # to back to indata root directore
        os.chdir('..')

    # verification plot
    rawdata.loc[rawdata.offsetx==0].sort_values(by='mm').reset_index(drop=True).plot(x='mm')

# go back to run dir
os.chdir(F.rundir)

### read and prepare input file or use prepared rawdata dataframe
if F.fromraw:
    df = rawdata
else:
    try:
        df = pd.read_csv(F.inputdata)
    except Exception as e:
        print(f'Could not open input data {F.inputdata}, exiting. Exception:\n{e}')

if F.debug:
    df.describe()

# delete empty lines
print("dropping empty lines")
df.dropna(inplace=True)

# select range with valid values
df = df.loc[(df.mm >= F.from_mm) & (df.mm <= F.to_mm)]
df.reset_index(inplace=True, drop=True)

### absolute value of the offset
df['offsetx_abs'] = df.offsetx.abs()
# df['mm_abs'] = df.mm.abs()

# helper columns
# calculate help columns
"""
helpcols = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']
for i in range(0, len(helpcols)-2):
    for j in range(i+1, len(helpcols)-1):
        ncol = f"{helpcols[i]}_minus_{helpcols[j]}"
        df[ncol] = df[helpcols[i]] - df[helpcols[j]]
        ncol = f"{helpcols[i]}_div_{helpcols[j]}"
        df[ncol] = df[helpcols[i]] / df[helpcols[j]]
        ncol = f"{helpcols[i]}_mult_{helpcols[j]}"
        df[ncol] = df[helpcols[i]] * df[helpcols[j]]
"""

### model mm  ##############################################################

print("******** analyze model for mm ****************")

# copy of original data
dfp = df.copy()

# predicted column
predcols = dfp.columns.to_list()
ycol = 'mm'
predcols.remove(ycol)
predcols.remove('offsetx')
predcols.remove('offsetx_abs')

### train and test sets
dfp['rc'] = np.random.rand(dfp.index.size)
dfp['test'] = False
dfp.loc[dfp.rc > F.testsplit, 'test'] = True    # random
# dfp.loc[dfp.index.size - dfp.index.size * F.testsplit < dfp.index , 'test'] = True         # sequential
xtrain = dfp.loc[~dfp.test, predcols].values
ytrain = dfp.loc[~dfp.test, ycol].values

# amend prediction columns?
# predcols = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'offsetx']
# predcols = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'mm']
# predcols = ['ch1', 'ch4', 'ch5', 'ch8']
# predcols.remove('ch7')

# model
modelstring = f"{F.modeltype}()"
md = eval(modelstring)

md.fit(xtrain, ytrain)  # fit prediction

# tree_to_code(md, predcols)

# build df with feature importances
dffe = pd.DataFrame({'feature': predcols, 'importance': md.feature_importances_})
dffe = dffe.sort_values('importance', ascending=False).reset_index(drop=True)
print(dffe)

# predict on test data
dfp['pred'] = md.predict(dfp.loc[:, predcols])  # make predictions
dfp['abserr'] = abs(dfp.pred - dfp[ycol])

# evaluate
mae_test_mean = dfp.loc[dfp.test].abserr.mean()
mae_test_max = dfp.loc[dfp.test].abserr.max()
print(f'Mean Absolute Error on test: {mae_test_mean}')
print(f'Max Absolute Error on test: {mae_test_max}')

### graphs
if F.debug:
    # sort appropriately (for graphs only)
    dfp.sort_values(by='mm', inplace=True)
    dfp.reset_index(inplace=True, drop=True)

    dfp.plot(x='mm', y='abserr', kind='scatter', title='Absolute error, model mm')

    # plot results
    dfp.plot(y=[ycol, 'pred'], title=f'Calculated {ycol} value, model mm')

# save model
pickle.dump(md, open(F.filemodel_1, 'wb'))
print(f"Model 1 for mm saved to file '{F.filemodel_1}'")


### model offsetx_abs  ##############################################################

print("******** analyze model for offsetx_abs ****************")

# copy of original data
dfp = df.copy()

# predicted column
predcols = dfp.columns.to_list()
ycol = 'offsetx_abs'
predcols.remove(ycol)
predcols.remove('offsetx')
predcols.remove('mm')

### train and test sets
dfp['rc'] = np.random.rand(dfp.index.size)
dfp['test'] = False
dfp.loc[dfp.rc > F.testsplit, 'test'] = True    # random
# dfp.loc[dfp.index.size - dfp.index.size * F.testsplit < dfp.index , 'test'] = True         # sequential
xtrain = dfp.loc[~dfp.test, predcols].values
ytrain = dfp.loc[~dfp.test, ycol].values

# amend prediction columns?
# predcols = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'offsetx']
# predcols = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'mm']
# predcols = ['ch1', 'ch4', 'ch5', 'ch8']
# predcols.remove('ch7')

# model
modelstring = f"{F.modeltype}()"
md = eval(modelstring)

md.fit(xtrain, ytrain)  # fit prediction

# tree_to_code(md, predcols)

# build df with feature importances
dffe = pd.DataFrame({'feature': predcols, 'importance': md.feature_importances_})
dffe = dffe.sort_values('importance', ascending=False).reset_index(drop=True)
print(dffe)

# predict on test data
dfp['pred'] = md.predict(dfp.loc[:, predcols])  # make predictions
dfp['abserr'] = abs(dfp.pred - dfp[ycol])

# evaluate
mae_test_mean = dfp.loc[dfp.test].abserr.mean()
mae_test_max = dfp.loc[dfp.test].abserr.max()
print(f'Mean Absolute Error on test: {mae_test_mean}')
print(f'Max Absolute Error on test: {mae_test_max}')

### graphs
if F.debug:
    # sort appropriately (for graphs only)
    dfp.sort_values(by='mm', inplace=True)
    dfp.reset_index(inplace=True, drop=True)

    dfp.plot(x='mm', y='abserr', kind='scatter', title='Absolute error, model offsetx_abs')

    # plot results
    dfp.plot(x='offsetx_abs', y='pred', kind='scatter', title=f'Calculated {ycol} value, model offsetx_abs')

# save model
pickle.dump(md, open(F.filemodel_2, 'wb'))
print(f"Model 2 for offset_x_abs saved to file '{F.filemodel_2}'")
