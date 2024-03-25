from functools import reduce
from typing import Union, Any
from tqdm import tqdm


# Manipulação de Dados
import pandas as pd
import numpy as np

# Modelos e Preprocessamento
from sklearn import tree
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, \
    roc_auc_score, \
        accuracy_score, \
            roc_curve, \
            confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import mutual_info_classif
from catboost import CatBoostClassifier, metrics
import optuna


# Visualização dos dados
import matplotlib.pyplot as plt
import seaborn as sns


### Definindo Funções Utilizadas
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
        self.fit(vet.astype(str))
        return self.transform(vet.astype(str))

def ppscore_num(df: pd.DataFrame, x: Union[list, np.ndarray], y: str) -> pd.DataFrame:
    """
    Calcula o Predictive Power Score para cada variável da base de dados escolhida, para uma variável alvo numérica.

    Paramêtros
    ----------
    df: Base de dados selecionada.

    x: Lista com as variáveis cujo o Predict Power Score será calculado.

    y: Nome da variável alvo.

    Retorna
    -------
    pd.DataFrame -> DataFrame com uma linah populada por todas as variáveis e com todas
    as outras colunas referentes ao pps de cada variável informada.
    """

    naive = df[y].median()
    median = np.ones(df[y].shape)*naive
    arrayAnalysis = []
    for col in tqdm(x):
        if df[col].dtype == 'O':
            catEnc = CatEncoder()
            aux = catEnc.fit_transform(df[col])
            oneHot = preprocessing.OneHotEncoder()
            aux =  oneHot.fit_transform(aux.values.reshape(-1,1))
            model = tree.DecisionTreeRegressor()
            model.fit(aux,df[y].fillna(0))
            score = 1 - (mean_absolute_error(df[y],model.predict(aux))/\
                            mean_absolute_error(df[y],median))
            arrayAnalysis.append({'column':col,'score':score})
        elif len(df[col].unique()) <= 20:
            catEnc = CatEncoder()
            aux = catEnc.fit_transform(df[col])
            oneHot = preprocessing.OneHotEncoder()
            aux =  oneHot.fit_transform(aux.values.reshape(-1,1))
            model = tree.DecisionTreeRegressor()
            model.fit(aux,df[y].fillna(0))
            score = 1 - (mean_absolute_error(df[y],model.predict(aux))/\
                            mean_absolute_error(df[y],median))
            arrayAnalysis.append({'column':col,'score':score})
        elif df[col].dtype == 'float64' or df[col].dtype == 'int64':
            fillna = 0 if np.isnan(df[col].mean()) or np.isinf(df[col].mean()) else df[col].mean()
            model = tree.DecisionTreeRegressor()
            model.fit(df[col].fillna(fillna).values.reshape(-1,1),df[y].fillna(0))
            score = 1 - (mean_absolute_error(df[y],model.predict(df[col].fillna(fillna).values.reshape(-1,1)))/\
                            mean_absolute_error(df[y],median))
            arrayAnalysis.append({'column':col,'score':score})
        else:
            arrayAnalysis.append({'column':col,'score':-10})

    return pd.DataFrame(arrayAnalysis)


def binary_score(target: Union[list, np.ndarray],
                variable: Union[list, np.ndarray],
                naive_pred: Union[list, np.ndarray]) -> float:
    """
    Obtém o ppscore para uma classificação binária.
    """
    return (roc_auc_score(target, variable) - roc_auc_score(target, naive_pred))/\
        (1-roc_auc_score(target, naive_pred))


def multiclass_score(target: Union[list, np.ndarray],
                    variable: Union[list, np.ndarray],
                    naive_pred: Union[list, np.ndarray]) -> float:
    """
    Obtém o ppscore para uma classificação multiclasse.
    """
    return (accuracy_score(target, variable) - accuracy_score(target, naive_pred))/\
        (1-accuracy_score(target, naive_pred))


# def multiclass_score(target, variable, naive_pred):
#     return (roc_auc_score(target, variable, multi_class='ovr', average='macro') - roc_auc_score(target, naive_pred, multi_class='ovr', average='macro'))/\
#         (1-roc_auc_score(target, naive_pred, multi_class='ovr', average='macro'))



def ppscore_cat(df: pd.DataFrame, x: Union[list, np.ndarray], y: str, classification='binary'):
    """
    Calcula o Predictive Power Score para cada variável da base de dados escolhida, para uma variável alvo categórica.

    Paramêtros
    ----------
    df: Base de dados selecionada.

    x: Lista com as variáveis cujo o Predict Power Score será calculado.

    y: Nome da variável alvo.

    Retorna
    -------
    pd.DataFrame -> DataFrame com uma linah populada por todas as variáveis e com todas
    as outras colunas referentes ao pps de cada variável informada.
    """
    dc = DummyClassifier(strategy='uniform')

    dc.fit(df[x].fillna(0), df[y].fillna(0))

    naive = dc.predict(df[x].fillna(0))

    arrayAnalysis = []

    for col in tqdm(x):
                
        if df[col].dtype == 'O':
            catEnc = CatEncoder()
            aux = catEnc.fit_transform(df[col])
            oneHot = preprocessing.OneHotEncoder()
            aux =  oneHot.fit_transform(aux.values.reshape(-1,1))
            model = tree.DecisionTreeClassifier()
            model.fit(aux,df[y].fillna(0))

            if classification == 'binary':
                score = binary_score(target=df[y].fillna(0), variable=model.predict(aux), naive_pred=naive)
            elif classification == 'multiclass':
                score = multiclass_score(target=df[y].fillna(0), variable=model.predict(aux), naive_pred=naive)
            else:
                raise ValueError("Classificação escolhida é inválida")
            arrayAnalysis.append({'column':col,'score':score})
        elif len(df[col].unique()) <= 20:
            catEnc = CatEncoder()
            aux = catEnc.fit_transform(df[col])
            oneHot = preprocessing.OneHotEncoder()
            aux =  oneHot.fit_transform(aux.values.reshape(-1,1))
            model = tree.DecisionTreeClassifier()
            model.fit(aux,df[y].fillna(0))

            if classification == 'binary':
                score = binary_score(target=df[y].fillna(0), variable=model.predict(aux), naive_pred=naive)
            elif classification == 'multiclass':
                score = multiclass_score(target=df[y].fillna(0), variable=model.predict(aux), naive_pred=naive)
            else:
                raise ValueError("Classificação escolhida é inválida")
            arrayAnalysis.append({'column':col,'score':score})
        elif df[col].dtype == 'float64' or df[col].dtype == 'int64':
            fillna = 0 if np.isnan(df[col].mean()) or np.isinf(df[col].mean()) else df[col].mean()
            model = tree.DecisionTreeClassifier()
            model.fit(df[col].fillna(fillna).values.reshape(-1,1),df[y].fillna(0))


            if classification=='binary':
                score = binary_score(target=df[y].fillna(0), variable=model.predict(df[col].fillna(fillna).values.reshape(-1,1)), naive_pred=naive)
            elif classification=='multiclass':
                score = multiclass_score(target=df[y].fillna(0), variable=model.predict(df[col].fillna(fillna).values.reshape(-1,1)), naive_pred=naive)
            else:
                raise ValueError("Classificação escolhida é inválida")
            arrayAnalysis.append({'column':col,'score':score})
        else:
            arrayAnalysis.append({'column':col,'score':-10})

    return pd.DataFrame(arrayAnalysis)

def data_ppscore(data: pd.DataFrame, variaveis: Union[list, np.ndarray], alvos: str) -> pd.DataFrame:
    """
    Calcula o Predictive Power Score para cada variável da base de dados escolhida, para mais de uma variável alvo.

    Paramêtros
    ----------
    df: Base de dados selecionada.

    x: Lista com as variáveis cujo o Predict Power Score será calculado.

    y: Nome da variável alvo.

    Retorna
    -------
    pd.DataFrame -> DataFrame com uma linha populada por todas as variáveis e com todas
    as outras colunas referentes ao pps de cada variável informada.
    """
    arr = []
    for alvo in alvos:
        score = 'pps_score_' + alvo
        
        arr.append(ppscore_cat(data, x=variaveis, y=alvo).rename(columns={'score': score}))
        
    return pd.concat(arr, axis=1).T.drop_duplicates().T


def objective(trial, X, y, cvs):
    """
    Função de estudo do optuna adaptada para Árvore de Decisão.

    Paramêtros
    ----------
    trial: Função processo de avaliação optuna

    x: Variáveis Explicativa

    y: Nome da variável alvo.

    Retorna
    -------
    tuple -> roc-auc média do modelo base de dados de teste, diferença da roc-auc de teste e
    do treinamento. 
    """
    param_grid = {
        "criterion": trial.suggest_categorical("criterion", ['gini', 'entropy']),
        "splitter": trial.suggest_categorical("splitter", ['random', 'best']),
        "min_samples_leaf": trial.suggest_int('min_samples_leaf', 1, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 5, 40),
        'ccp_alpha': trial.suggest_loguniform('ccp_alpha', 0.0001, 0.05),
        'max_depth': trial.suggest_int('max_depth', 2, 50)

        
    }

    cv = StratifiedKFold(n_splits=cvs, shuffle=True, random_state=42)

    cv_scores = np.empty(cvs)
    cv_scores_train = np.empty(cvs)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = tree.DecisionTreeClassifier(
            **param_grid
        )

        model.fit(
            X_train,
            y_train
        )

        preds = model.predict_proba(X_test)
        cv_scores[idx] = roc_auc_score(y_test, preds[:,1])
        cv_scores_train[idx] = roc_auc_score(y_train,model.predict_proba(X_train)[:,1]) - cv_scores[idx]
        
    return np.mean(cv_scores), np.mean(cv_scores_train)
def estudo_optuna(x: pd.DataFrame, y: Union[list, np.ndarray, pd.Series], thres = 0.1) -> dict:
    """
    Realização do estudo optuna

    Paramêtros
    ----------
    x: Dataframe com as variáveis explicativas

    y: Nome da variável alvo.

    Retorna
    -------
    pd.DataFrame -> DataFrame com uma linah populada por todas as variáveis
    """

    study_pps = optuna.create_study(directions=['maximize','minimize'])
    func_pps = lambda trial: objective(trial, x, y, 5)
    study_pps.optimize(func_pps, n_trials=100)

    gdsOptuna_pps = study_pps.trials_dataframe()
    dicts_pps = gdsOptuna_pps[(gdsOptuna_pps['values_1'] < thres)].sort_values(['values_0','values_1'],ascending=[False, False]).head(20).to_dict(orient='records')
    return {
    'criterion' : dicts_pps[0]['params_criterion'],
    'splitter' : dicts_pps[0]['params_splitter'],
    'min_samples_leaf' : dicts_pps[0]['params_min_samples_leaf'],
    'min_samples_split' : dicts_pps[0]['params_min_samples_split'],
    'max_leaf_nodes' : dicts_pps[0]['params_max_leaf_nodes'],
    'ccp_alpha' : dicts_pps[0]['params_ccp_alpha'],
    'max_depth': dicts_pps[0]['params_max_depth']
    }

def ppscore_cat_opt(df, x, y, classification='binary'):
    """
    Calcula o Predictive Power Score para cada variável da base de dados escolhida, para mais de uma variável alvo.
    O modelo utilizado para calcular o Predictive Power Score passa por um estudo optuna, de modo que os melhores
    hiperparâmetros sejam escolhidos. Retornando um PPS mais preciso.

    Paramêtros
    ----------
    df: Base de dados selecionada.

    x: Lista com as variáveis cujo o Predict Power Score será calculado.

    y: Nome da variável alvo.

    Retorna
    -------
    pd.DataFrame -> DataFrame com uma linah populada por todas as variáveis e com todas
    as outras colunas referentes ao pps de cada variável informada.
    """
    dc = DummyClassifier(strategy='uniform')

    dc.fit(df[x].fillna(0), df[y].fillna(0))

    naive = dc.predict(df[x].fillna(0))

    arrayAnalysis = []

    for col in tqdm(x):
                
        if df[col].dtype == 'O':
            catEnc = CatEncoder()
            aux = catEnc.fit_transform(df[col])
            oneHot = preprocessing.OneHotEncoder(sparse=False)
            aux =  oneHot.fit_transform(aux.values.reshape(-1,1))
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            param = estudo_optuna(pd.DataFrame(aux).fillna(0), df[y].fillna(0))
            model = tree.DecisionTreeClassifier(**param)
            model.fit(aux,df[y].fillna(0))

            if classification == 'binary':
                score = binary_score(target=df[y].fillna(0), variable=model.predict(aux), naive_pred=naive)
            elif classification == 'multiclass':
                score = multiclass_score(target=df[y].fillna(0), variable=model.predict(aux), naive_pred=naive)
            else:
                raise ValueError("Classificação escolhida é inválida")
            arrayAnalysis.append({'column':col,'score':score})
        elif len(df[col].unique()) <= 20:
            catEnc = CatEncoder()
            aux = catEnc.fit_transform(df[col])
            oneHot = preprocessing.OneHotEncoder(sparse=False)
            aux =  oneHot.fit_transform(aux.values.reshape(-1,1))
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            param = estudo_optuna(pd.DataFrame(aux).fillna(0), df[y].fillna(0))
            model = tree.DecisionTreeClassifier(**param)
            model.fit(aux,df[y].fillna(0))


            if classification == 'binary':
                score = binary_score(target=df[y].fillna(0), variable=model.predict(aux), naive_pred=naive)
            elif classification == 'multiclass':
                score = multiclass_score(target=df[y].fillna(0), variable=model.predict(aux), naive_pred=naive)
            else:
                raise ValueError("Classificação escolhida é inválida")
            arrayAnalysis.append({'column':col,'score':score})
        elif df[col].dtype == 'float64' or df[col].dtype == 'int64':
            fillna = 0 if np.isnan(df[col].mean()) or np.isinf(df[col].mean()) else df[col].mean()
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            param = estudo_optuna(pd.DataFrame(df[col]).fillna(fillna), df[y].fillna(0))
            model = tree.DecisionTreeClassifier(**param)
            model.fit(df[col].fillna(fillna).values.reshape(-1,1),df[y].fillna(0))


            if classification=='binary':
                score = binary_score(target=df[y].fillna(0), variable=model.predict(df[col].fillna(fillna).values.reshape(-1,1)), naive_pred=naive)
            elif classification=='multiclass':
                score = multiclass_score(target=df[y].fillna(0), variable=model.predict(df[col].fillna(fillna).values.reshape(-1,1)), naive_pred=naive)
            else:
                raise ValueError("Classificação escolhida é inválida")
            arrayAnalysis.append({'column':col,'score':score})

    return pd.DataFrame(arrayAnalysis)

class FeatureImportance:
    def __init__(self):
        self.data_most_important = pd.DataFrame()
        self.catModel = None

    def load(self, data: pd.DataFrame, variaveis: Union[str, list], alvo: str) -> None:
        self.data = data
        self.variaveis = variaveis
        self.alvo = alvo
        

    def objective_lr(self, trial: Any, X: list, y: str, cvs: int) -> tuple:
        """
        Função utilizada para o estudo do optuna dos hiperparâmetros
        da Regressão Logística
        """

        # Definição dos hiperparâmetros que serão foco do estudo
        param_grid = {
            "tol": trial.suggest_loguniform("tol", 0.00001, 1.0),
            "max_iter": trial.suggest_int("max_iter", 50, 200),
            "l1_ratio": trial.suggest_loguniform("l1_ratio", 0.0, 1.0)
        }

        # Estabelecendo o KFold estratificado
        cv = StratifiedKFold(n_splits=cvs, shuffle=True, random_state=42)

        cv_scores = np.empty(cvs)
        cv_scores_train = np.empty(cvs)
        for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = LogisticRegression(class_weight='balanced',
                                        penalty='elasticnet',
                                        solver='saga',
                                        random_state=13,
                                        **param_grid
            )
            model.fit(
                X_train,
                y_train,
            )
            preds = model.predict_proba(X_test)
            
            try:
                cv_scores[idx] = roc_auc_score(y_test, preds[:,1])
                cv_scores_train[idx] = roc_auc_score(y_train,model.predict_proba(X_train)[:,1]) - cv_scores[idx]
            except:
                cv_scores[idx] = -1
                cv_scores_train[idx] = 1
        return np.mean(cv_scores), np.mean(cv_scores_train)


    def objective_cat(self, trial: Any, X: list, y: str, cvs: int) -> tuple:
        """
        Função para estudo Optuna do modelo de CatBoost
        """
        param_grid = {
            "n_estimators": trial.suggest_int("n_estimators", 5,50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_categorical("max_depth", [2,3,5]),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel",  0.05, 1)
        }

        cv = StratifiedKFold(n_splits=cvs, shuffle=True, random_state=42)

        cv_scores = np.empty(cvs)
        cv_scores_train = np.empty(cvs)
        for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model = CatBoostClassifier(custom_loss=[metrics.AUC()],
                        **param_grid)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=100,
                verbose_eval=False
            )
            preds = model.predict_proba(X_test)
            try:
                cv_scores[idx] = roc_auc_score(y_test, preds[:,1])
                cv_scores_train[idx] = roc_auc_score(y_train,model.predict_proba(X_train)[:,1]) - cv_scores[idx]
            except:
                cv_scores[idx] = -1
                cv_scores_train[idx] = 1
        return np.mean(cv_scores), np.mean(cv_scores_train)

    def mutual_info(self, data: pd.DataFrame, X: Union[str, list], y: str) -> list:
        """
        Função para o Cálculo da Informação Mútua entre as variáveis explicativas
        e a variável alvo.
        """
        if isinstance(X, str):
            df_ = data.dropna(subset=[X, y], how='any', axis=0)
            df_ = df_[[X, y]].copy()
        elif isinstance(X, list):
            df_ = data.dropna(subset=X + [y], how='any', axis=0)
            df_ = df_[X + [y]].copy()
        df_ = df_.reset_index(drop=True)

        return mutual_info_classif(df_.drop(columns=[y]), df_[y], random_state=13)
        




    def logistic_roc(self, data: pd.DataFrame, X: Union[str, list], y: str) -> float:
        """
        Função que obtém a roc de um modelo de regressão logística treinado 
        a partir de um conjunto de variáveis especificado.

        O modelo é otimizado via optuna.
        """
        
        if isinstance(X, str):
            df_ = data.dropna(subset=y, how='any', axis=0)
            df_ = df_[[X] + [y]].copy()

        elif isinstance(X, list):
            df_ = data.dropna(subset=[y], how='any', axis=0)
            df_ = df_[X + [y]].copy()
        else:
            raise ValueError("Variaveis deve ser uma instância list ou str")

        df_ = df_.reset_index(drop=True)

        try:
            study = optuna.create_study(directions=['maximize','minimize'])
            func = lambda trial: self.objective_lr(trial, df_.drop(columns=y).fillna(0), df_[y], 5)
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study.optimize(func, n_trials=50)

            gdsOptuna = study.trials_dataframe()
            dicts = gdsOptuna.sort_values(['values_0','values_1'],ascending=[False, False]).head(20).to_dict(orient='records')

                
            return dicts[0]['values_0']
        except:
            return -1




    def catBoost_roc(self, data: pd.DataFrame, X: Union[str, list], y: str) -> float:
        """
        Função que obtém a roc de um CatBoost treinado 
        a partir de um conjunto de variáveis especificado.

        O modelo é otimizado via optuna.
        """

        if isinstance(X, str):
            df = data.dropna(subset=y, how='any', axis=0)
            df = df[[X] + [y]].copy()

        elif isinstance(X, list):
            df = data.dropna(subset=y, how='any', axis=0)
            df = df[X + [y]].copy()

        df = df.reset_index(drop=True)

        try:
            study = optuna.create_study(directions=['maximize','minimize'])
            func = lambda trial: self.objective_cat(trial, df.drop(columns=[y]), df[y], 5)
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study.optimize(func, n_trials=50)

            gdsOptuna = study.trials_dataframe()
            dicts = gdsOptuna.sort_values(['values_0','values_1'],ascending=[False, False]).head(20).to_dict(orient='records')

                
            return dicts[0]['values_0']
        except:
            return -1


    def fit_catBoost(self, data: pd.DataFrame, X: Union[str, list], y: str) -> tuple:

        if isinstance(X, str):

            df = data.dropna(subset=y, how='any', axis=0)
            df = df[[X] + [y]].copy()
        elif isinstance(X, list):
            df = data.dropna(subset=y, how='any', axis=0)
            df = df[X + [y]].copy()
        df = df.reset_index(drop=True)


        
        study = optuna.create_study(directions=['maximize','minimize'])
        func = lambda trial: self.objective_cat(trial, df.drop(columns=[y]), df[y], 5)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(func, n_trials=50)

        gdsOptuna = study.trials_dataframe()
        dicts = gdsOptuna.sort_values(['values_0','values_1'],ascending=[False, False]).head(20).to_dict(orient='records')

        # return dicts

        # Definindo o modelo de catBoost

        params = {
        'colsample_bylevel' : dicts[0]['params_colsample_bylevel'],
        'learning_rate' : dicts[0]['params_learning_rate'],
        'max_depth' :dicts[0]['params_max_depth'],
        'n_estimators' : dicts[0]['params_n_estimators'],
        'reg_lambda' : dicts[0]['params_reg_lambda'],
        }
        self.catModel = CatBoostClassifier(custom_loss=[metrics.AUC()],
                verbose=False,
                **params)
        self.catModel.fit(
            df.drop(columns=y), df[y],
            verbose=False
        )



    def catBoost_gain(self, data: pd.DataFrame, X: Union[str, list], y: str) -> tuple:
        """
        Função que obtém o ganho de emtropia de um CatBoost treinado 
        a partir de um conjunto de variáveis especificado.

        O modelo é otimizado via optuna.
        """


        self.fit_catBoost(data, X, y)

        return self.catModel.feature_importances_, self.catModel.feature_names_

    
    def data_logistic_roc(self) -> pd.DataFrame:
        """
        Gera uma tabela ranqueando as variáveis com maior roc na regressão logística
        """
        metrica = 'logistic_roc_'+ self.alvo
        dic_ = {
            'variavel': [],
            metrica: []
        }

        for variavel in self.variaveis:
            dic_['variavel'].append(variavel)
            dic_[metrica].append(self.logistic_roc(self.data, variavel, self.alvo))


        return pd.DataFrame(dic_).sort_values(by=metrica, ascending=False)

    def data_catboost_roc(self) -> pd.DataFrame:
        """
        Gera uma tabela ranqueando as variáveis com maior roc no CatBoost
        """
        metrica = 'catboost_roc_' + self.alvo
        dic_ = {
            'variavel': [],
            metrica: []
        }

        for variavel in self.variaveis:
            dic_['variavel'].append(variavel)
            dic_[metrica].append(self.catBoost_roc(self.data, variavel, self.alvo))


        return pd.DataFrame(dic_).sort_values(by=metrica, ascending=False)


    def data_mutual_info(self) -> pd.DataFrame:
        """
        Gera uma tabela ranqueando as variáveis com maior informação mútua
        """
        metrica = 'mutual_info_' + self.alvo
        dic_ = {
            'variavel': [],
            metrica: []
        }

        for variavel in self.variaveis:
            dic_['variavel'].append(variavel)
            dic_[metrica].append(self.mutual_info(self.data, variavel, self.alvo)[0])


        return pd.DataFrame(dic_).sort_values(by=metrica, ascending=False)



    def data_catboost_gain(self) -> pd.DataFrame:
        """
        Gera uma tabela ranqueando as variáveis com maior ganho de entropia
        no CatBoost
        """
        metrica = 'catboost_gain_' + self.alvo
        feats_ = self.catBoost_gain(self.data, self.variaveis, self.alvo)
        dic_ = {
            'variavel': feats_[1],
            metrica: feats_[0]
        }

        return pd.DataFrame(dic_).sort_values(by=metrica, ascending=False)


    def data_model_metrics(self) -> pd.DataFrame:
        """
        Gera uma tabela ranqueando as variáveis com maior ganho de entropia
        no CatBoost
        """
        metrica = 'catboost_gain_' + self.alvo
        feats_ = self.catBoost_gain(self.data, self.variaveis, self.alvo)
        dic_ = {
            'variavel': feats_[1],
            metrica: feats_[0]
        }

        return pd.DataFrame(dic_).sort_values(by=metrica, ascending=False)


    

    def get_feature_importances(self, cache: bool=True) -> pd.DataFrame:
        """
        Gera uma tabela compilando os resultados da roc logística, roc catboost,
        ganho de entropia catboost, e informação mútua
        """
        if cache and len(self.data_most_important):
            return self.data_most_important

        else:
            
            temp_ = [
                        self.data_logistic_roc(),
                        self.data_mutual_info(),
                        self.data_catboost_gain(),
                        self.data_catboost_roc(),

            ]

            self.data_most_important = reduce(lambda right, left: pd.merge(right, left, on='variavel'), temp_)

            return self.data_most_important

def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None,
                          ax=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories,ax=ax)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)

        
def limiar_escore(predictions,df_target):
    #Imprimindo limiar de Escore
    fpr, tpr, threshold = roc_curve(df_target, predictions)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.loc[(roc.tf-0).abs().argsort()[:1]]
    # print('Limiar que maxima especificidade e sensitividade:')
    # print(list(roc_t['threshold']))
    #analisando modelo com novo limiar
    tn, fp, fn, tp = confusion_matrix(df_target, [1 if item>=list(roc_t['threshold'])[0] else 0 for item in predictions]).ravel()
    Precision = tp/(tp+fp)
    Recall = tp/(tp+fn)
    acuracia = (tp+tn)/(tn+fp+fn+tp)
    F = (2*Precision*Recall)/(Precision+Recall)
    # print('Precision',Precision)
    # print('Recall',Recall)
    # print('Acuracia',acuracia)
    # print('F-Score',F)
    # print('Roc-AUC', roc_auc_score(df_target, predictions))
    return{'precision': Precision,
            'recall': Recall,
            'acuracia': acuracia,
            'f-score': F,
            'roc-auc': roc_auc_score(df_target, predictions),
            'limiar': float(roc_t['threshold'])
            }

def merge_all(array_df: list, variavel_merge: str) -> pd.DataFrame:
    """
    Mergea uma lista de dataframes de acordo com a variável escolhida
    """
    return reduce(lambda right, left: pd.merge(right, left, on=variavel_merge), array_df)



def calculate_score(df: pd.DataFrame, metrics_column: list):
    """
    Dado o dataframe, e as colunas com as métricas. Calcula o score por ranqueamento das métricas.
    A primeira coluna do dataframe deve ser a coluna de variáveis. Todas as demais devem ser numéricas.
    """
    df_ = df.sort_values(by=metrics_column[0], ascending=False)
    df_['score'] = [x for x in range(len(df)-1, -1, -1)]
    
    for metric in metrics_column[1:]:
        df_ = df_.sort_values(by=metric, ascending=False)
        df_['temp_score'] = [x for x in range(len(df)-1, -1, -1)]

        df_['score'] = df_[['score', 'temp_score']].sum(axis=1)

        df_ = df_.drop(columns='temp_score')

    return df_.sort_values(by='score', ascending=False)
