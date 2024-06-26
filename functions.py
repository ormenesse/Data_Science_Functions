from pandas.api.types import is_numeric_dtype
from scipy.spatial.distance import cdist
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score, roc_curve 
from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict, KFold, GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ks_2samp
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import ppscore as pps
import random
import seaborn as sns

class CatEncoder():
    """Semelhante ao LabelEncoder do SKLEARN mas funciona com categorias ineditas.
        Uso: ce.fit(df[col]) e ce.transform(df[col])."""
    def __init__(self):
        self.dic = {}
        self.rev_dic = {}
    def fit(self, vet):
        uniques = []
        for c in vet.unique():
            if str(type(c)) == "<class 'str'>":
                uniques.append(c)
        uniques.sort()
        for a, b in enumerate(uniques):
            self.dic[b] = a
            self.rev_dic[a] = b
        return self
    def check(self, vet):
        try:
            if type(vet) == list:
                return pd.Series(vet)
            return vet
        except:
            #print('Error in check function. Are you using list instead of a Pandas Column?')
            return vet
    def transform(self, vet):
        vet = self.check(vet)
        #não quero isso
        #return vet.map(self.dic).replace(np.nan, -1).astype(int)
        return vet.map(self.dic)
    def inverse_transform(self, vet):
        vet = self.check(vet)
        #não quero isso 
        #return vet.map(self.rev_dic).replace(np.nan, 'NaN')
        return vet.map(self.rev_dic)
    def fit_transform(self,vet):
        self.fit(vet)
        return self.transform(vet)
    
def save_to_file(objeto, nome_arquivo):
    with open(nome_arquivo, 'wb') as output:
        pickle.dump(objeto, output, pickle.HIGHEST_PROTOCOL)


def load_file(nome_arquivo):
    with open(nome_arquivo, 'rb') as input:
        objeto = pickle.load(input)
    return objeto

#lidando com variáveis catégoricas
def faz_meu_label(X):
    dict_label = {}
    categoricas = []
    for coluna in X.columns:
        if str(X[coluna].dtypes) == 'object':
            label_encoder = LabelEncoder()
            coluna_prov = label_encoder.fit_transform(X[coluna].astype(str))
            X[coluna] = coluna_prov
            dict_label[coluna] = label_encoder
            categoricas.append(coluna)
    return dict_label


def faz_meu_label2(X):
    dict_label = {}
    for coluna in X.columns:
        if str(X[coluna].dtypes) == 'object':
            ce = CatEncoder()
            coluna_prov = ce.fit_transform(X[coluna])
            X[coluna] = coluna_prov
            dict_label[coluna] = ce
    return dict_label


def min_max_norm(B,categorical_columns):
    X = B.copy()
    norm = {}
    for col in X.columns:
        if col in categorical_columns:
            norm[col] = [X[col].min(),X[col].max(),1,0]
        else:
            norm[col] = [X[col].min(),X[col].max(),X[col].std(),X[col].mean()]
            X[col]=(X[col]-X[col].min())/(X[col].max()-X[col].min())
    return X , norm

def std_norm(B,categorical_columns):
    X = B.copy()
    norm = {}
    for col in X.columns:
        if col in categorical_columns:
            norm[col] = [X[col].min(),X[col].max(),1,0]
        else:
            norm[col] = [X[col].min(),X[col].max(),X[col].std(),X[col].mean()]
            X[col]=(X[col]-X[col].mean())/(X[col].std())
    return X , norm

def cria_matriz_correlacao(df):
    correlations = df.corr()
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations,vmin=-1, vmax=1,cmap='Greys')
    fig.colorbar(cax)
    ticks = np.arange(0,len(df.columns),1)
    ax.set_xticks(ticks-0.5)
    ax.set_yticks(ticks-0.5)
    ax.set_xticklabels(df.columns,rotation = 90,ma='center',size='medium')
    ax.set_yticklabels(df.columns,ma='center',size='medium')
    plt.show()
    
def cria_matriz_pps(df):
    correlations = pps.matrix(df).pivot(columns='x', index='y',  values='ppscore')
    plt.figure(figsize=(20,20))
    sns.heatmap(correlations, annot=True)
    #plt.show()
    
def exclui_vars_correlacionadas(db,y=None,frac=1,corrFactor=0.95):
    if y == None:
        correlacao = db.sample(frac=frac).corr().abs()
        corrs = [] #variaveis correlacionadas
        keep = []
        for i,c in enumerate(correlacao.iterrows()):
            for j,czinho in enumerate(c[1]):
                if czinho >= corrFactor and i != j:
                    if c[1].index[i] not in keep and c[1].index[i] not in corrs:
                        keep.append(c[1].index[i])
                        corrs.append(c[1].index[j])
                        #print(c[1].index[i],'->',c[1].index[j])
        return list(set(keep))
    else:
        correlacao = db.sample(frac=frac).corr().abs()
        corrs = [] #variaveis correlacionadas
        keep = []
        for i,c in enumerate(correlacao.iterrows()):
            currentRow = c[0]
            for j,czinho in enumerate(c[1]):
                if czinho >= corrFactor and i != j:
                    if c[1].index[i] not in keep and c[1].index[i] not in corrs:
                        if correlacao[y].loc[currentRow] > correlacao[y].loc[c[1].index[j]]:
                            keep.append(c[1].index[i])
                            corrs.append(c[1].index[j])
                        else:
                            keep.append(c[1].index[j])
                            corrs.append(c[1].index[i])                 
        return list(set(keep))

def cria_curva_roc_auc(modelo,df_verificacao,df_target):
    predictions = modelo.predict_proba(df_verificacao)
    fpr, tpr, threshold = roc_curve(df_target, predictions[:,1])
    plt.figure(figsize=(8,8))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_score(df_target, predictions[:,1]))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic REAIS')
    plt.legend(loc="lower right")
    plt.show()
    
def limiar_escore(modelo,df_verificacao,df_target):
    #Imprimindo limiar de Escore
    predictions = modelo.predict_proba(df_verificacao)
    fpr, tpr, threshold = roc_curve(df_target, predictions[:,1])
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.loc[(roc.tf-0).abs().argsort()[:1]]
    print('Limiar que maxima especificidade e sensitividade:')
    print(list(roc_t['threshold']))
    #analisando modelo com novo limiar
    tn, fp, fn, tp = confusion_matrix(df_target, [1 if item>=list(roc_t['threshold'])[0] else 0 for item in predictions[:,1]]).ravel()
    Precision = tp/(tp+fp)
    Recall = tp/(tp+fn)
    acuracia = (tp+tn)/(tn+fp+fn+tp)
    F = (2*Precision*Recall)/(Precision+Recall)
    print('Precision',Precision)
    print('Recall',Recall)
    print('Acuracia',acuracia)
    print('F-Score',F)
    print('Roc-AUC', roc_auc_score(df_target, predictions[:,1]))

def df_limiar_escore(df_verificacao,df_target,_print=True):
  #Imprimindo limiar de Escore
  predictions = df_verificacao
    try:
        fpr, tpr, threshold = roc_curve(df_target, predictions)
        i = np.arange(len(tpr)) 
        roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
        roc_t = roc.loc[(roc.tf-0).abs().argsort()[:1]]
    except:
        roc_t = pd.DataFrame([np.nan],columns=['threshold'])
    #analisando modelo com novo limiar
    try:
        tn, fp, fn, tp = confusion_matrix(df_target, [1 if item>=list(roc_t['threshold'])[0] else 0 for item in predictions]).ravel()
        Precision = tp/(tp+fp)
        Recall = tp/(tp+fn)
        acuracia = (tp+tn)/(tn+fp+fn+tp)
        F = (2*Precision*Recall)/(Precision+Recall)
    except:
        Precision = np.nan
        Recall = np.nan
        acuracia = np.nan
        F = np.nan
    try:
        roc = roc_auc_score(df_target, predictions)
    except:
        roc = np.nan
    try:
        ks = ks_2samp(df_target, predictions)
    except:
        ks = (np.nan, np.nan)
    if _print:
        print('Limiar que maxima especificidade e sensitividade:')
        print(list(roc_t['threshold']))
        print('Precision',Precision)
        print('Recall',Recall)
        print('Acuracia',acuracia)
        print('F-Score',F)
        print('Roc-AUC', roc)
        print('KS', ks)
    return {
        'bestThreshold': list(roc_t['threshold'])[0],
        'precision': Precision,
        'recall': Recall,
        'accuracy': acuracia,
        'fscore': F,
        'rocauc': roc,
        'gini': 2*roc-1,
        'ks': ks,
        'classDist': df_target.value_counts(normalize=True).to_dict()
      }


def retorna_limiar_escore(modelo,df_verificacao,df_target):
    #Imprimindo limiar de Escore
    predictions = modelo.predict_proba(df_verificacao)
    fpr, tpr, threshold = roc_curve(df_target, predictions[:,1])
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.loc[(roc.tf-0).abs().argsort()[:1]]
    return list(roc_t['threshold'])[0]
    
def kfoldcv(indices, k = 5, seed = 4242, how='tuple'):
    
    size = len(indices)
    subset_size = round(size / k)
    random.Random(seed).shuffle(indices)
    subsets = [indices[x:x+subset_size] for x in range(0, len(indices), subset_size)]
    test = []
    train = []
    tupleTrainTest = []
    for i in range(k):
        test.append(subsets[i])
        trainz = np.array([])
        for j,subset in enumerate(subsets):
            if i != j:
                trainz = np.concatenate((trainz,subset),axis=0)
        train.append(list(trainz))
        
        tupleTrainTest.append((train[-1],test[-1]))
        
    if how == 'tuple':
        
        return tupleTrainTest
    
    elif how == 'array':
        
        return train,test
    
    else:
        
        return tupleTrainTest
