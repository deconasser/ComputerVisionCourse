import os
import numpy as np
from sklearn.metrics import r2_score

def MAE(y_true, y_pred):
    """ Mean Absolute Error """
    # print(f'{y_true.shape = }')
    # print(f'{y_pred.shape = }')
    # print(f'{(y_true - y_pred).shape = }')
    return np.mean(np.abs(y_true - y_pred))

def MSE(y_true, y_pred):
    """ Mean Squared Error """ 
    return np.mean((y_true - y_pred) ** 2)

def RMSE(y_true, y_pred):
    """ Root Mean Squared Error """
    return np.sqrt(np.mean((y_true-y_pred)**2))


metric_dict = {
    'MAE' : MAE, 
    'MSE' : MSE,
    'RMSE' : RMSE, 
}

def score(y, 
          yhat, 
          # path=None, 
          # model='',
          r):
    # print(f'{y.shape = }')
    # print(f'{yhat.shape = }')
    if len(yhat.shape) == 3: 
        nsamples, nx, ny = yhat.shape
        yhat = yhat.reshape((nsamples,nx*ny))

    y = np.squeeze(y)
    yhat = np.squeeze(yhat)
    
    if r != -1:
        results = [str(np.round(np.float64(metric_dict[key](y, yhat)), r)) for key in metric_dict.keys()]
    else:
        results = [str(metric_dict[key](y, yhat)) for key in metric_dict.keys()]
    # if path: 
    #     os.makedirs(os.path.join(path, 'values'), exist_ok=True)
    #     np.save(open(os.path.join(path, 'values', f'yhat-{model}.npy'), 'wb'), yhat)
    return results
