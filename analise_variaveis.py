import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import warnings
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer

def analise_temporal_base(X,X_perc=0.25,coluna_target=None,coluna_mes='NUM_MES_REF',
                          dsvpd_thr=0.25,nome_arquivo="analise_dados_mes",MI=False,
                          Fast_Analysis=False,log_cat=False,log_fig=False,
                          transpor=False,subs_miss_espec=True,vetor_esp_missing=[-1,-9,-99,-999,-999999999]):
    """
    #Análise Temporal de Bases#
    Args:
        X: Base para analise (requerido pandas DataFrame).
        X_perc: Amostragem da base inicial que será utilizado (0 < X_perc <= 1). Análise fica mais rápida, caso realizado desta maneira.
        coluna_target: Coluna da base onde está o target, se None, análise por target não será feita.
        coluna_mes:  coluna onde os dados são separados mensalmente, padrão "NUM_MES_REF".
        dsvpd_thr: Limiar mensal (threshold) da variação de coeficiente (detector de problemas) 
        nome_arquivo: Nome do arquivo Excel que será salvo.
        MI: Mutual Information (true - retorna, False - não retorna)
        Fast_Analysis: Trabalhando com uma análise pontual e mais rápida sem tantos detalhes.
        log_cat: Logs em texto no Jupyter/terminal.
        log_fig: Plot de figuras no Jupyter ou não.
        transpor: Transpor informações nas planilhas
        subs_miss_espec: Substituição de missing especial por np.nan, sim ou não
        vetor_esp_missing: Vetor de Substituição de Missings Especiais, se subs_miss_espec == True
        
    Returns:
        Arquivo planilha Excel com dados consolidados.
    
    Author:
        Vinícius Ormenesse
    """
    warnings.filterwarnings('ignore')
    if str(type(X)) == "<class 'pandas.core.frame.DataFrame'>":
        writer = pd.ExcelWriter(nome_arquivo+'.xlsx', engine='xlsxwriter')
        if X_perc > 1:
            X_perc = 1
            print('Percentual de amostragem incorreto, o percentual utilizado nesta análise será de 100% da amostra.')
        elif X_perc <= 0:
            X_perc = 0.1
            print('Percentual de amostragem incorreto, o percentual utilizado nesta análise será de 10% da amostra.')
        if dsvpd_thr > 1:
            dsvpd_thr = 1
            print('dsvpd_thr incorreto, o percentual utilizado nesta análise será de 100% da amostra.')
        elif dsvpd_thr <= 0:
            dsvpd_thr = 0.1
            print('dsvpd_thr incorreto, o percentual utilizado nesta análise será de 10% da amostra.')
        X = X.sample(frac=X_perc) #fracionamento da entrada, para que análise fique mais rápida.
        meses = X[coluna_mes].fillna(method='ffill').unique()
        meses.sort()
        #Transformando missing em verdadeiros MISSINGS.
        if subs_miss_espec:
            print('Transformando Missings Especiais')
            X = X.replace(vetor_esp_missing, np.full(len(vetor_esp_missing), np.nan)) 
        #tratando colunas com categoricas
        categoricas = []
        for cols in X.columns:
            if cols is not coluna_mes:
                if str(X[cols].dtypes) == 'object':
                    if log_cat == True:
                        print("Ordem das classes de label para a coluna: "+cols)
                        print('Antes:')
                        print(X[cols].unique())
                    categoricas.append(cols)
                    le = LabelEncoder()
                    X[cols] = le.fit_transform(X[cols].astype(str))
                    if log_cat == True:
                        print('Depois')
                        print(list(le.classes_))
        #Começando a análise para quem possui variável resposta.
        if coluna_target is not None:
            colunas_var = X.columns.tolist()
            colunas_var.remove(coluna_mes)
            colunas_var.remove(coluna_target)
            #fuer die variabeln
            """
                0 - desvpad
                1 - mean 
                2 - missing
                3 - outliers
                4 - Coefficient Variation
                5 - KS
                6 - MI
                7 - PSI
            """
            analises = []
            for i in range(0,8):
                analises.append(pd.DataFrame([],index=colunas_var))
            #valores targets
            valores_targets = X[coluna_target].unique().tolist()
            
            #olhando para missing media e desvio padrao
            for i,mes in enumerate(meses):
                print("Calculando dados no mês: " + str(int(mes)))
                for valor in valores_targets:
                    valor = int(valor)
                    outlier = []
                    apsi = []
                    describe = X[colunas_var][(X[coluna_mes] == mes) & (X[coluna_target] == valor)].describe().transpose()
                    pdstd = pd.DataFrame(describe['std'].values[:,None],columns=[str(mes)+"_"+str(valor)],index=describe.index)
                    pdmedia = pd.DataFrame(describe['mean'].values[:,None],columns=[str(mes)+"_"+str(valor)],index=describe.index)
                    pdcv = pdstd/pdmedia #coefficient variation 
                    c_shape = X[colunas_var][(X[coluna_mes] == mes) & (X[coluna_target] == valor)].shape[0]
                    pdmissing = pd.DataFrame((describe['count'].values/c_shape),columns=[str(mes)+"_"+str(valor)],index=describe.index)
                    for coluna in colunas_var:
                        out = X[coluna][(X[coluna_mes] == mes) & (np.abs(X[coluna]-describe['mean'].loc[coluna]) > (3*X[coluna].mean()-describe['std'].loc[coluna]))].count()
                        outlier.append(out)
                        if i != 0 and not Fast_Analysis:
                            apsi.append(calculate_psi(X[coluna][(X[coluna_mes] == mes) & (X[coluna_target] == valor)].dropna(),X[coluna][(X[coluna_mes] == meses[i-1]) & (X[coluna_target] == valor)].dropna()))
                    pdpsi = pd.DataFrame(apsi,columns=[str(mes)+"_"+str(valor)],index=colunas_var)
                    pdoutlier = pd.DataFrame(outlier,columns=[str(mes)+"_"+str(valor)],index=colunas_var)
                    analises[0] = pd.concat([analises[0],pdstd],axis=1)
                    analises[1] = pd.concat([analises[1],pdmedia],axis=1)
                    analises[2] = pd.concat([analises[2],pdmissing],axis=1)
                    analises[3] = pd.concat([analises[3],pdoutlier],axis=1)
                    analises[4] = pd.concat([analises[4],pdcv],axis=1)
                    if not Fast_Analysis:
                        analises[7] = pd.concat([analises[7],pdpsi],axis=1)
                #fazendo teste KS - Trabalhando com 1 ou mais classes.
                if not Fast_Analysis:
                    combinacoes_target = list(itertools.combinations_with_replacement(valores_targets,2))
                    for c_t in combinacoes_target:
                        if c_t[0] != c_t[1]:
                            for mes in meses:
                                ksarray = []
                                for coluna in colunas_var:
                                    ks,pvalor = stats.ks_2samp(X[coluna][(X[coluna_mes] == mes) & (X[coluna_target] == c_t[0])], X[coluna][(X[coluna_mes] == mes) & (X[coluna_target] == c_t[1])])
                                    ksarray.append(ks)
                                pdks = pd.DataFrame(ksarray,columns=[str(mes)+"_"+str(str(c_t))],index=colunas_var)
                            analises[5] = pd.concat([analises[5],pdks],axis=1)
            #Informação Mútua
            if MI:
                table_mi = X.copy()
                KB = KBinsDiscretizer(n_bins=10,encode='ordinal')
                colunas = []
                for coluna in X.columns:
                    if coluna not in categoricas and (coluna not in [coluna_mes,coluna_target]):
                        colunas.append(coluna)
                table_mi[colunas] = KB.fit_transform(table_mi[colunas].fillna(0))
                for mes in meses:
                    mi = mutual_info_classif(table_mi[(table_mi[coluna_mes] == mes)].drop([coluna_mes,coluna_target],axis=1).fillna(0), table_mi[coluna_target][(table_mi[coluna_mes] == mes)])
                    #mi /= np.max(mi) #melhor nao fazer essa análise
                    pdmi = pd.DataFrame(mi,columns=[str(mes)],index=X.drop([coluna_mes,coluna_target],axis=1).columns)
                    analises[6] = pd.concat([analises[6],pdmi],axis=1)
                del table_mi, colunas, KB
            #Analisando Estatisticas no Extraídas
            lista_histogramas = set() #não vou querer imprimir isso várias vezes
            for i,analise in enumerate(analises):
                dsvpd = pd.DataFrame([],columns=['DESVPAD_FEATURE'])
                meanpd = pd.DataFrame([],columns=['MEAN_FEATURE'])
                for ind in analise.index:
                    dsvpd.loc[ind] = analise.loc[ind].std()
                    meanpd.loc[ind] = analise.loc[ind].mean()
                    if (dsvpd.loc[ind].values/meanpd.loc[ind].values >= dsvpd_thr or dsvpd.loc[ind].values/meanpd.loc[ind].values <= 0.03) and (i != 2 or i != 3):
                        lista_histogramas.add(ind)
                analises[i] = pd.concat([analise,dsvpd,meanpd],axis=1)
            #analisar dados no excel
            sheets = ['DESVPAD','MEDIA','MISSING','OUTLIERS','COEFF VAR','KS','MI','PSI']
            if not transpor:
                for i in range(0,8):
                    analises[i].to_excel(writer,sheets[i])
                for sh in sheets:
                    worksheet = writer.sheets[sh]
                    for i in range(2,analises[0].shape[0]+2):
                        coluna_excel = num_to_col_letters(i)
                        worksheet.conditional_format('B'+str(i)+':'+num_to_col_letters(analises[0].shape[1]-1)+str(i), {'type': '3_color_scale','min_type': 'percent','mid_type':'percent','max_type': 'percent'})
            else:
                for i in range(0,8):
                    analises[i].transpose().to_excel(writer,sheets[i])
                for sh in sheets:
                    worksheet = writer.sheets[sh]
                    for i in range(2,analises[0].shape[0]+2):
                        coluna_excel = num_to_col_letters(i)
                        worksheet.conditional_format(coluna_excel+'2'+':'+coluna_excel+str(1+len(meses)*len(valores_targets)), {'type': '3_color_scale','min_type': 'percent','mid_type':'percent','max_type': 'percent'})
            #Salvando Imagens
            if not Fast_Analysis:
                worksheet = writer.book.add_worksheet(name="Análises_Imagens")
                row = 0
                for ind in lista_histogramas:
                    col = 0
                    #histogramas
                    plt.figure(figsize=(10,5))
                    for mes in meses:
                        try:
                            for valor in valores_targets:
                                sns.distplot(X[ind][(X[coluna_target] == valor) & (X[coluna_mes] == mes)].dropna())
                            plt.legend(valores_targets)
                            plt.title("Distribuição de "+str(ind)+" "+str(mes))
                            imgdata = BytesIO()
                            plt.savefig(imgdata, format="png")
                            imgdata.seek(0)
                            worksheet.insert_image(row, col, "", {'image_data': imgdata})
                            col += 17
                            if not log_fig:
                                plt.close()
                            else:
                                plt.show()
                                plt.close()
                        except:
                            pass
                    row += 25
                col = 0
                #Matriz de covariancia
                worksheet = writer.book.add_worksheet(name="Covariância")
                correlations = X.corr()
                fig = plt.figure(figsize=(1+int(len(X.columns)*0.2401+0.8911),1+int(len(X.columns)*0.2401+0.8911)))
                ax = fig.add_subplot(111)
                #cax = ax.matshow(correlations,extent=[0,len(colunas_var),0,len(colunas_var)],vmin=-1, vmax=1)
                cax = ax.matshow(correlations,vmin=-1, vmax=1)
                fig.colorbar(cax)
                ticks = np.arange(0,len(X.columns),1)
                ax.set_xticks(ticks-0.5)
                ax.set_yticks(ticks-0.5)
                ax.set_xticklabels(X.columns,rotation = 90,ma='center',size='medium')
                ax.set_yticklabels(X.columns,ma='center',size='medium')
                imgdata = BytesIO()
                fig.savefig(imgdata, format="png")
                imgdata.seek(0)
                worksheet.insert_image(0, 0, "", {'image_data': imgdata})
                if not log_fig:
                    plt.close()
                else:
                    plt.show()
                    plt.close()
            #Finalmente salvando o arquivo e testando.
            print("Salvando os dados como:"+nome_arquivo+".xlsx")
            writer.save()
            
        else:
            colunas_var = X.columns.tolist()
            colunas_var.remove(coluna_mes)
            #fuer die variabeln
            """
                0 - desvpad
                1 - mean 
                2 - missing
                3 - outliers
                4.- PSI
                5 - Coefficient Variation
            """
            analises = []
            for i in range(0,6):
                analises.append(pd.DataFrame([],index=colunas_var))
            #olhando para missing media e desvio padrao
            for i,mes in enumerate(meses):
                print("Calculando dados no mês: " + str(int(mes)))
                outlier = []
                apsi = []
                describe = X[colunas_var][(X[coluna_mes] == mes)].describe().transpose()
                pdstd = pd.DataFrame(describe['std'].values[:,None],columns=[str(mes)],index=describe.index)
                pdmedia = pd.DataFrame(describe['mean'].values[:,None],columns=[str(mes)],index=describe.index)
                pdcv = pdstd/pdmedia
                c_shape = X[colunas_var][(X[coluna_mes] == mes)].shape[0]
                pdmissing = pd.DataFrame((describe['count'].values/c_shape),columns=[str(mes)],index=describe.index)
                for coluna in colunas_var:
                    out = X[coluna][(X[coluna_mes] == mes) & (np.abs(X[coluna]-describe['mean'].loc[coluna]) > (3*X[coluna].mean()-describe['std'].loc[coluna]))].count()
                    outlier.append(out)
                    if i != 0:
                        apsi.append(calculate_psi(X[coluna][(X[coluna_mes] == mes)].dropna(),X[coluna][(X[coluna_mes] == meses[i-1])].dropna()))
                pdoutlier = pd.DataFrame(outlier,columns=[str(mes)],index=colunas_var)
                pdpsi = pd.DataFrame(apsi,columns=[str(mes)],index=colunas_var)
                analises[0] = pd.concat([analises[0],pdstd],axis=1)
                analises[1] = pd.concat([analises[1],pdmedia],axis=1)
                analises[2] = pd.concat([analises[2],pdmissing],axis=1)
                analises[3] = pd.concat([analises[3],pdoutlier],axis=1)
                analises[4] = pd.concat([analises[4],pdpsi],axis=1)
                analises[5] = pd.concat([analises[5],pdcv],axis=1)
            #Analisando Estatisticas no Extraídas
            lista_histogramas = set() #não vou querer imprimir isso várias vezes
            for i,analise in enumerate(analises):
                dsvpd = pd.DataFrame([],columns=['DESVPAD_FEATURE'])
                meanpd = pd.DataFrame([],columns=['MEAN_FEATURE'])
                for ind in analise.index:
                    dsvpd.loc[ind] = analise.loc[ind].std()
                    meanpd.loc[ind] = analise.loc[ind].mean()
                    if (dsvpd.loc[ind].values/meanpd.loc[ind].values >= dsvpd_thr or dsvpd.loc[ind].values/meanpd.loc[ind].values <= 0.03) and (i != 2 or i != 3):
                        lista_histogramas.add(ind)
                analises[i] = pd.concat([analise,dsvpd,meanpd],axis=1)
            #analisar dados no excel    
            sheets = ['DESVPAD','MEDIA','MISSING','OUTLIERS','PSI','COEFF VAR']
            if not transpor:
                for i in range(0,6):
                    analises[i].to_excel(writer,sheets[i])
                for sh in sheets:
                    worksheet = writer.sheets[sh]
                    for i in range(2,analises[0].shape[0]+2):
                        coluna_excel = num_to_col_letters(i)
                        worksheet.conditional_format('B'+str(i)+':'+num_to_col_letters(analises[0].shape[1]-1)+str(i), {'type': '3_color_scale','min_type': 'percent','mid_type':'percent','max_type': 'percent'})
            else:
                for i in range(0,6):
                    analises[i].transpose().to_excel(writer,sheets[i])
                for sh in sheets:
                    worksheet = writer.sheets[sh]
                    for i in range(2,analises[0].shape[0]+2):
                        coluna_excel = num_to_col_letters(i)
                        worksheet.conditional_format(coluna_excel+'2'+':'+coluna_excel+str(1+len(meses)), {'type': '3_color_scale','min_type': 'percent','mid_type':'percent','max_type': 'percent'})
            #Salvando Imagens
            if not Fast_Analysis:
                worksheet = writer.book.add_worksheet(name="Análises_Imagens")
                row = 0
                for ind in lista_histogramas:
                    col = 0
                    #histogramas
                    plt.figure(figsize=(10,5))
                    try:
                        for mes in meses:
                            sns.distplot(X[ind][(X[coluna_mes] == mes)].dropna())
                    except:
                        print('Erro plotar gráfico no índice',str(ind),'no mes',str(mes))
                    plt.title("Distribuição de "+str(ind))
                    imgdata = BytesIO()
                    plt.savefig(imgdata, format="png")
                    imgdata.seek(0)
                    worksheet.insert_image(row, col, "", {'image_data': imgdata})
                    col += 17
                    if not log_fig:
                        plt.close()
                    else:
                        plt.show()
                        plt.close()
                    row += 25
                col = 0
                #Matriz de covariancia
                worksheet = writer.book.add_worksheet(name="Covariância")
                correlations = X.corr()
                fig = plt.figure(figsize=(1+int(len(X.columns)*0.2401+0.8911),1+int(len(X.columns)*0.2401+0.8911)))
                ax = fig.add_subplot(111)
                cax = ax.matshow(correlations,vmin=-1, vmax=1)
                fig.colorbar(cax)
                ticks = np.arange(0,len(X.columns),1)
                ax.set_xticks(ticks-0.5)
                ax.set_yticks(ticks-0.5)
                ax.set_xticklabels(X.columns,rotation = 90,ma='center',size='medium')
                ax.set_yticklabels(X.columns,ma='center',size='medium')
                imgdata = BytesIO()
                fig.savefig(imgdata, format="png")
                imgdata.seek(0)
                worksheet.insert_image(0, 0, "", {'image_data': imgdata})
                if not log_fig:
                    plt.close()
                else:
                    plt.show()
                    plt.close()
            #Salvando o resultado em uma planilha de excel para análise.
            print("Salvando os dados como:"+nome_arquivo+".xlsx")
            writer.save()
    else:
        print("Por favor, utilizar um dataframe pandas para utilizar essa função.\nNada foi feito.")
        
#função para trabalhar com o Excel
def num_to_col_letters(num):
    letters = ''
    while num:
        mod = (num - 1) % 26
        letters += chr(mod + 65)
        num = (num - 1) // 26
    return ''.join(reversed(letters))
'''
Interpretation
PSI < 0.1: no significant population change
PSI < 0.2: moderate population change
PSI >= 0.2: significant population change
def psi(coluna_base,coluna_comparacao):
    base = np.histogram(coluna_comparacao)[0]/len(coluna_comparacao)
    target = np.histogram(coluna_base)[0]/len(coluna_base)
    return np.sum((base-target)*np.log(1e-9 + base/target))
'''
#PSI thanks to this guy:
#https://github.com/mwburke/population-stability-index/blob/master/psi.py
def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0):
    '''Calculate the 
    (population stability index) across all variables
    Args:
       expected: numpy matrix of original values
       actual: numpy matrix of new values, same size as expected
       buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
       buckets: number of quantiles to use in bucketing variables
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal
    Returns:
       psi_values: ndarray of psi values for each variable
    Author:
       Matthew Burke
       github.com/mwburke
       worksofchart.com
    '''

    def psi(expected_array, actual_array, buckets):
        '''Calculate the PSI for a single variable
        Args:
           expected_array: numpy array of original values
           actual_array: numpy array of new values, same size as expected
           buckets: number of percentile ranges to bucket the values into
        Returns:
           psi_value: calculated PSI value
        '''

        def scale_range (input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input
        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        elif buckettype == 'quantiles':
            breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])
        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)
        def sub_psi(e_perc, a_perc):
            '''Calculate the actual PSI value from comparing the values.
               Update the actual value to a very small number if equal to zero
            '''
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001
            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return(value)
        psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))
        return(psi_value)
    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[axis])
    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, actual, buckets)
        elif axis == 0:
            psi_values[i] = psi(expected[:,i], actual[:,i], buckets)
        elif axis == 1:
            psi_values[i] = psi(expected[i,:], actual[i,:], buckets)
    return(psi_values)
