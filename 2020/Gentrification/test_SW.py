import tqdm
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy import linalg as LA
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import hamming_loss
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import json
import warnings
warnings.filterwarnings("ignore")


import scipy, sklearn, numpy, pandas
print('dependency library versions')
print('scipy', scipy.__version__)
print('sklearn', sklearn.__version__)
print('numpy', numpy.__version__)
print('pandas', pandas.__version__)


# # Ground Truth Done
#b+f
with open('Dataset/groundtruth.json') as fp:
    guide = json.load(fp)

yn_df = pd.read_excel('Dataset/Physical/대상지별_물리데이터_종합_전체연도.xlsx', sheet_name='연남동 공시지가')
ss_df = pd.read_excel('Dataset/Physical/대상지별_물리데이터_종합_전체연도.xlsx', sheet_name='성수동 공시지가')
gl_df = pd.read_excel('Dataset/Physical/대상지별_물리데이터_종합_전체연도.xlsx', sheet_name='경리단길 공시지가')


# In[3]:
price_dict = dict()
for bac, df in [('YN', yn_df), ('GL', gl_df), ('SS', ss_df)]:
    x = list(range(2014, 2021))
    for ac in df.columns[1:]:
        y = df[df['연도'] >= 2013][ac].tolist()
        f = interp1d(x, y)
        xnew = np.linspace(2014, 2020, num=12*6, endpoint=False)
        price_dict['{}-{}'.format(bac, ac)] = f(xnew)


# In[4]:


#########################
### Prepare Data Load ###
#########################

def load_gtdf():
    return pd.read_csv('Dataset/Physical/20200804_4_gis_gtmonthly_2010_final.csv', parse_dates=['date'], index_col=None)


gtdf = load_gtdf()

def load_igdf():
    igdf = pd.read_csv('Dataset/Social/20200725_instagram_the_final.csv', parse_dates=['date'])
    return igdf

igdf = load_igdf()


# place365 load the class label
file_name = 'Dataset/Social/categories_places365.txt'
place365_classes = list()
with open(file_name) as class_file:
    for line in class_file:
        place365_classes.append(line.strip().split(' ')[0][3:])
place365_classes = tuple(place365_classes)

########################
### Side Bar Options ###
########################

ac_unique = sorted(gtdf['area'].unique().tolist())
#ac = 'YN-T'

gtdf_colnames = gtdf.columns.tolist()
mode_dict = {
    'lbf': ['l', 'b', 'f'],
    'op':  [name for name in gtdf_colnames if name[:3] == 'op_'],
    'opup':[name for name in gtdf_colnames if name[:5] == 'opup_']
}
gtdf_coi = mode_dict['lbf'] + mode_dict['op'] + mode_dict['opup']

window = 1

def sig_convolve(arr, window):
    if window == 1:
        return arr
    win = signal.hann(window*2)
    farr = np.zeros(arr.shape)
    farr[window:-window] = (signal.convolve(arr[:], win, mode='same') / sum(win))[window:-window]
    farr[:window] = farr[window]
    farr[-window:] = farr[-window-1]
    return farr

def get_conv_derv_results(fgtdf, window=1):
    ffgtdf = pd.DataFrame(index = fgtdf.index)
    for c in fgtdf.columns:
        if c == 'date':
            ffgtdf[c] = fgtdf[c]
        else:
            ffgtdf[c] = sig_convolve(np.array(fgtdf[c]), window)

    dfgtdf = pd.DataFrame(index = ffgtdf.index)
    for c in ffgtdf.columns:
        if c == 'date':
            dfgtdf[c] = ffgtdf[c]
        else:
            dfgtdf[c] = np.concatenate(([0],                 np.array(ffgtdf[c].iloc[1:]) - np.array(ffgtdf[c].iloc[:-1])))

    return ffgtdf, dfgtdf


#def get_dataY(ac):
    ################# Ground Truth #################

y1, m1, y2, m2 = 2014, 1, 2020, 1

actestdf = dict()
for ac in ac_unique:
    #if ac[:2] != 'YN':
    #    continue
    fgtdf = gtdf[gtdf['area'] == ac][['date'] + gtdf_coi]
    ffgtdf, dfgtdf = get_conv_derv_results(fgtdf)


    figdf = igdf[igdf['ac'] == ac]

    w2v_tag = ['w2v_{}'.format(i) for i in range(100)]
    w2v_ft = [w2v_tag[i] for i in np.argsort(figdf[w2v_tag].values.sum(axis=0))]
    w2v_ft_norm = ['norm_{}'.format(s) for s in w2v_ft]
    ag_tag = ['M0', 'M10', 'M20', 'M30', 'M40', 'M50', 'M50+',
              'F0', 'F10', 'F20', 'F30', 'F40', 'F50', 'F50+']
    ag_norm = ['norm_{}'.format(s) for s in ag_tag]

    cls_tag = list(place365_classes)
    cls_ft = [cls_tag[i] for i in np.argsort(figdf[cls_tag].values.sum(axis=0))]
    cls_ft_norm = ['norm_{}'.format(s) for s in cls_ft]

    cddict = {'TN':['PostN', 'PicN', 'PersonPicN', 'PersonN', 'Male', 'Female'], 
              'TN_norm':['PersonPicRatio', 'MeanPersonEach', 'norm_Male', 'norm_Female'],
              'ag':ag_tag,
              'ag_norm':ag_norm,
              'w2v':w2v_ft,
              'w2v_norm':w2v_ft_norm,
              'place365':cls_ft,
              'place365_norm':cls_ft_norm}


    allarr = []
    for k in cddict:
        allarr.extend(cddict[k])


    figdf = figdf[['date'] + allarr]
    ffigdf, dfigdf = get_conv_derv_results(figdf)

    start_date = pd.Timestamp(y1, m1, 1)
    end_date = pd.Timestamp(y2, m2, 1)

    ffgtdf = ffgtdf.set_index('date')
    ffigdf = ffigdf.set_index('date')


    dataY = ffgtdf[(ffgtdf.index >= start_date) & (ffgtdf.index < end_date)]
    y_cols = dataY.columns = ['{}'.format(s) for s in dataY.columns]

    dataX = ffigdf[(ffigdf.index >= start_date) & (ffigdf.index < end_date)]
    #st.write('dataX', dataX)
    x_cols = dataX.columns = ['{}'.format(s) for s in dataX.columns]

    testdf = pd.concat((dataY, dataX), axis=1)
    actestdf[ac] = testdf




# In[6]:


ymlist2010 = []
for y in range(2010, 2020):
    for m in range(1, 13):
        ymlist2010.append(y*100 + m)


# In[7]:



def phase_interpolate(mlist):
    xs = []
    ys = []
    for item in mlist:
        if item[0] == 202001:
            xs.append(ymlist2010.index(201912))
        else:
            xs.append(ymlist2010.index(item[0]))
        ys.append(item[1])
        
    f = interp1d(xs, ys)
    xnew = np.arange(len(ymlist2010))
    ynew = f(xnew)
    return f(xnew)


# In[8]:


mtestdfs = []
for ac in guide:
    testdf = actestdf[ac]
    all_phase = phase_interpolate(guide[ac])
    phase = all_phase[12*4:]
    prev_phase = all_phase[12*4-1:-1]
    ftestdf = testdf[testdf.index >= pd.Timestamp('2014-01-01')].copy()
    ftestdf['ac'] = ac
    ftestdf['stage'] = 1*(2 <= phase) + 1*(2.5 <= phase)# + 1*(2.5 <= phase) 
    ftestdf['stage10'] = (phase*10 - 10).astype(int)
    #ftestdf['stage'] = 1*(2 <= phase)# + 1*(2 <= phase)# + 1*(2.5 <= phase) 
    ftestdf['stagereg'] = phase
    ftestdf['stageregprev'] = prev_phase
    mtestdfs.append(ftestdf)

mtestdf = pd.concat(mtestdfs)
mtestdf['bac'] = [s[:2] for s in mtestdf['ac']]

N_phase = mtestdf['stage'].nunique()



# Feature Selection

xall = w2v_ft + w2v_ft_norm + ag_tag + ag_norm + cls_ft + cls_ft_norm
#xorig = w2v_ft + ag_tag + cls_ft
xnorm = w2v_ft_norm + ag_norm + cls_ft_norm


# In[12]:


mtestdf['totlbf'] = [a+b+c for a, b, c in zip(mtestdf['l'], mtestdf['b'], mtestdf['f'])]
mtestdf['norm_l'] = [a/t for a, t in zip(mtestdf['l'], mtestdf['totlbf'])]
mtestdf['norm_b'] = [a/t for a, t in zip(mtestdf['b'], mtestdf['totlbf'])]
mtestdf['norm_f'] = [a/t for a, t in zip(mtestdf['f'], mtestdf['totlbf'])]


# In[13]:


from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
#X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
xtest = xnorm + ['norm_l', 'norm_b', 'norm_f']
X = mtestdf[xtest].values
yr = mtestdf['stagereg'].values
yc = mtestdf['stage'].values

print("Runing Recursive Feature Extraction using Support Vector Regressor ...")
svr_estimator = SVR(kernel='linear')
svr_selector = RFE(svr_estimator, n_features_to_select=1, step=1).fit(X, yr)


# In[14]:


class my_regression_model:
    def __init__(self, model_name):
        self.model_name = model_name
        if model_name == 'sv':
            self.model = SVR()
        if model_name == 'rf':
            self.model = RandomForestRegressor()
        if model_name == 'lr':
            self.model = LinearRegression()
        if model_name == 'gbr':
            self.model = GradientBoostingRegressor()
            
    def fit(self, X_train, y_train):
        if model_name in ['lr', 'sv', 'rf', 'gbr']:
            self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        if model_name in ['lr', 'sv', 'rf', 'gbr']:
            return self.model.predict(X_test)


class my_classification_model:
    def __init__(self, model_name, no_class):
        self.model_name = model_name
        self.no_class = no_class
        if model_name == 'lr':
            self.model = LogisticRegression()
        if model_name == 'sv':
            self.model = SVC()
        if model_name == 'rf':
            self.model = RandomForestClassifier()
        if model_name == 'dnn1':
            self.model = keras.Sequential([
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dense(self.no_class, activation='softmax')
            ])
            self.model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
        if model_name == 'dnn2':
            self.model = keras.Sequential([
                keras.layers.Dense(256, activation='relu'),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dense(self.no_class, activation='softmax')
            ])
            self.model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
            
    def fit(self, X_train, y_train):
        if model_name in ['lr', 'sv', 'rf']:
            self.model.fit(X_train, y_train)
        if model_name in ['dnn1', 'dnn2']:
            self.model.fit(X_train, y_train, epochs=150, verbose=0)
    
    def predict(self, X_test):
        if model_name in ['lr', 'sv', 'rf']:
            return self.model.predict(X_test)
        if model_name in ['dnn1', 'dnn2']:
            return np.argmax(self.model.predict(X_test), axis=1)
        
    def predict_s(self, X_test):
        if model_name == 'sv':
            return self.model.predict(X_test)
        if model_name in ['lr', 'rf']:
            return self.model.predict_proba(X_test) @ np.arange(self.no_class)
        if model_name in ['dnn1', 'dnn2']:
            return self.model.predict(X_test) @ np.arange(self.no_class)


# In[15]:


def moving_average(numbers, window):
	numbers_series = pd.Series(numbers)
	windows = numbers_series.rolling(window)
	moving_averages = windows.mean()
	arr = moving_averages_list = moving_averages.tolist()
	
	return arr


# In[16]:


ac_list = sorted(mtestdf.ac.unique().tolist())


# In[17]:


yns = ac_list


# In[18]:


selected_cols = {
                 'result': np.array(xtest)[svr_selector.ranking_ <= 449].tolist(),
                }


# In[19]:


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[20]:


print("Train & Testing Results ...")
#training_ac = ['GL-A', 'GL-B', 'GL-D']#, 'SS-E', 'SS-F']

training_ac = yns#yn_ac_list#mtestdf['ac'].unique().tolist()
selected_ac = yns#yn_ac_list#mtestdf['ac'].unique().tolist()#['GL-B']
report = {ac:dict() for ac in selected_ac}

rtestdf = mtestdf[mtestdf['ac'].isin(training_ac + selected_ac)]
yr = rtestdf['stagereg']
yc = rtestdf['stage10']
for ac in selected_ac:
    for k, selected in selected_cols.items():
        if k not in report[ac]:
            report[ac][k] = dict()
        
        X = rtestdf[selected]
        X_train, X_test = X[rtestdf['ac'] != ac].values.astype(dtype=np.float32),                           X[rtestdf['ac'] == ac].values.astype(dtype=np.float32)
        yr_train, yr_test = yr[rtestdf['ac'] != ac].values.astype(dtype=np.float32),                             yr[rtestdf['ac'] == ac].values.astype(dtype=np.float32)
        yc_train, yc_test = yc[rtestdf['ac'] != ac].values.astype(dtype=np.float32),                             yc[rtestdf['ac'] == ac].values.astype(dtype=np.float32)
        
        
            
        for model_name in ['lr']:#, 'gbr']:
            model = my_regression_model(model_name)
            
            yrs_train = yr_train
            Xs_train = X_train
            
            model.fit(Xs_train, yrs_train)
            yr_pred = model.predict(X_test).tolist()
            report[ac][k]['reg_'+model_name] = (yr_pred, yr_test.tolist())
            
            vX = np.vstack((yr_pred, yr_test))
            CoV = np.var(vX, axis=0, ddof=1).mean()
            mapacc = 100-mean_absolute_percentage_error(yr_pred, yr_test)
            print(ac, 'reg', model_name, mapacc)
            #print(ac, k, 'reg', model_name,                   '{:.4f}'.format(mean_absolute_error(yr_pred, yr_test.tolist())),                  '{:.4f}'.format(pearsonr(yr_pred, yr_test)[0]), sep='\t')
            


print(' ', 'MAPACC (100-MAPE)', sep='\t')

for k in selected_cols:
    model_mapaccs = []
    for m in ['lr']:
        mapaccs = []
        for ac in report:
            yr_pred, yr_test = report[ac][k]['reg_'+m]
            mapacc = 100-mean_absolute_percentage_error(yr_pred, yr_test)
            mapaccs.append(mapacc)

        model_mapaccs.append(np.mean(mapaccs))

    print(k, '\t'.join(['{:.4f}'.format(m) for m in model_mapaccs]), sep='\t')


# In[22]:




