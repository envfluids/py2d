from scipy import stats
from sklearn.neighbors import KernelDensity
import numpy as np

def myKDE(uM, BANDWIDTH=1, Nx=100):

    print('Start: Calculating PDF - KDE ... > ')

    Vecvalues=uM.reshape(-1,1)
    print('Vecvalues shape: ', Vecvalues.shape)
    Vecpoints=np.linspace(Vecvalues.min(),Vecvalues.max(),Nx).reshape(-1,1)
    print('Vecpoints shape: ', Vecpoints.shape)
    kde = KernelDensity(kernel='gaussian', bandwidth=BANDWIDTH).fit(Vecvalues)
    logkde = kde.score_samples(Vecpoints)
    print('logkde shape: ', logkde.shape)

    print('...| End of Calculating PDF - KDE |')

    return Vecpoints.reshape(-1,), np.exp(logkde).reshape(-1,), logkde.reshape(-1,), kde

def mymodelsave(model,filename):
    import joblib

    filename = filename+'.sav'
    joblib.dump(model, filename)
    print('KDE model saved: ', filename )

    return 1

#from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import LeaveOneOut

def mybandwidth_scott(x):
    # https://docs.scipy.org/doc//scipy-1.4.1/reference/generated/scipy.stats.gaussian_kde.html
    d = 1
    n = len(x)
    return n**(-1./(d+4))