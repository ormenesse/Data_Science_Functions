from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import warnings

def analise_temporal_base(X,X_perc=0.25,coluna_target=None,coluna_mes='NUM_MES_REF',dsvpd_thr=5,nome_arquivo="analise_dados_mes",loggy_cat=True,loggy_fig=False):
	"""	
	#Análise Temporal de Bases#
	X - Base para analise (requerido pandas DataFrame).
	X_perc - Percentual de amostragem da base inicial que será utilizado (0 < X_perc <= 1).
	coluna_target - Coluna da base onde está o target, se None, análise por target não será feita.
	coluna_mes -  coluna onde os dados são separados mensalmente, padrão "NUM_MES_REF".
	dsvpd_thr - Limiar mensal (threshold) do desvio padrão mensal 
	nome_arquivo - Nome do arquivo Excel que será salvo.
	loggy_cat - Logs em texto no Jupyter/terminal.
	loggy_fig - Plot de figuras no Jupyter ou não.
	
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
		X = X.sample(frac=X_perc) #samppleando o input
		meses = X[coluna_mes].fillna(method='ffill').unique()
		meses.sort()
		
		#tratando colunas com categoricas
		for cols in X.columns:
			if str(X[cols].dtypes) == 'object':
				if loggy_cat == True:
					print("Ordem das classes de label para a coluna: "+cols)
					print('Antes:')
					print(X[cols].unique())
				le = LabelEncoder()
				X[cols] = le.fit_transform(X[cols].astype(str))
				if loggy_cat == True:
					print('Depois')
					print(list(le.classes_))
						
		if coluna_target is not None:
			
			#fuer die variabeln
			"""
				0 - desvpad
				1 - mean 
				2 - missing
				3 - outliers
				4 - KS
			"""
			analises = []
			for i in range(0,5):
				analises.append(pd.DataFrame([],index=X.columns))
			#valores targets
			valores_targets = X[coluna_target].unique().tolist()
			
			#olhando para missing media e desvio padrao
			for mes in meses:
				print("Trabalhando no mes: " + str(int(mes)))
				for valor in valores_targets:
					valor = int(valor)
					mean = []
					std = []
					missing = []
					outlier = []
					for coluna in X.columns:
					#media e std sem na
						percentil = 0
						desvpad = 0
						media = 0
						out = 0
						numnans = X[coluna][(X[coluna_mes] == mes) & (X[coluna_target] == valor)].isnull().sum() #Qntd de missing
						percentil = numnans/X.shape[0]
						desvpad = X[coluna][(X[coluna_mes] == mes) & (X[coluna_target] == valor)].dropna().std()#eu tenho que dar um dropna
						media = X[coluna][(X[coluna_mes] == mes) & (X[coluna_target] == valor)].dropna().mean()#eu tenho que dar um dropna
						out = X[coluna][(X[coluna_mes] == mes) & (X[coluna_target] == valor) & (np.abs(X[coluna]-media) > (3*desvpad))].count()#outliers
						mean.append(media)
						std.append(desvpad)
						missing.append(percentil)
						outlier.append(out)
					pdstd = pd.DataFrame(std,columns=[str(mes)+"_"+str(valor)],index=X.columns)
					pdmedia = pd.DataFrame(mean,columns=[str(mes)+"_"+str(valor)],index=X.columns)
					pdmissing = pd.DataFrame(missing,columns=[str(mes)+"_"+str(valor)],index=X.columns)
					pdoutlier = pd.DataFrame(outlier,columns=[str(mes)+"_"+str(valor)],index=X.columns)
					analises[0] = pd.concat([analises[0],pdstd],axis=1)
					analises[1] = pd.concat([analises[1],pdmedia],axis=1)
					analises[2] = pd.concat([analises[2],pdmissing],axis=1)
					analises[3] = pd.concat([analises[3],pdoutlier],axis=1)
				#fazendo teste KS - Trabalhando com 1 ou mais classes.
				combinacoes_target = list(itertools.combinations_with_replacement(valores_targets,2))
				for c_t in combinacoes_target:
					if c_t[0] != c_t[1]:
						ks,pvalor = stats.ks_2samp(X[coluna][(X[coluna_mes] == mes) & (X[coluna_target] == c_t[0])], X[coluna][(X[coluna_mes] == mes) & (X[coluna_target] == c_t[1])])
						pdks = pd.DataFrame(ks,columns=[str(mes)+"_"+str(str(c_t))],index=X.columns)
						analises[4] = pd.concat([analises[4],pdks],axis=1)
			#Analisando Estatisticas no Extraídas
			lista_histogramas = set() #não vou querer imprimir isso várias vezes
			for i,analise in enumerate(analises):
				dsvpd = pd.DataFrame([],columns=['DESVPAD_FEATURE'])
				meanpd = pd.DataFrame([],columns=['MEAN_FEATURE'])
				for ind in analise.index:
					dsvpd.loc[ind] = analise.loc[ind].std()
					meanpd.loc[ind] = analise.loc[ind].mean()
					if dsvpd.loc[ind].values > dsvpd_thr:
						lista_histogramas.add(ind)
				analises[i] = pd.concat([analise,dsvpd,meanpd],axis=1)
			#analisar dados no excel	
			sheets = ['DESVPAD','MEDIA','MISSING','OUTLIERS','KS']
			analises[0].to_excel(writer,sheets[0])
			analises[1].to_excel(writer,sheets[1])
			analises[2].to_excel(writer,sheets[2])
			analises[3].to_excel(writer,sheets[3])
			analises[4].to_excel(writer,sheets[4])
			
			for sh in sheets:
				worksheet = writer.sheets[sh]
				for i in range(2,analises[0].shape[0]+2):
					coluna_excel = num_to_col_letters(i)
					worksheet.conditional_format('B'+str(i)+':'+num_to_col_letters(analises[0].shape[1]-1)+str(i), {'type': '3_color_scale','min_type': 'percent','mid_type':'percent','max_type': 'percent'})
			#Salvando Imagens
			worksheet = writer.book.add_worksheet(name="Análises_Imagens")
			row = 0
			for ind in lista_histogramas:
				plt.figure(figsize=(10,5))
				for valor in valores_targets:
					sns.distplot(X[ind][(X[coluna_target] == valor)].dropna())
				plt.legend(valores_targets)
				plt.title("Distribuição de "+str(ind))
				imgdata = BytesIO()
				plt.savefig(imgdata, format="png")
				imgdata.seek(0)
				worksheet.insert_image(row, 0, "", {'image_data': imgdata})
				row = row+25
				if not loggy_fig:
					plt.close()
				else:
					plt.show()
					plt.close()
								
			#Matriz de covariancia
			correlations = X.corr()
			fig = plt.figure(figsize=(20,20))
			ax = fig.add_subplot(111)
			#cax = ax.matshow(correlations,extent=[0,len(X.columns),0,len(X.columns)],vmin=-1, vmax=1)
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
			worksheet.insert_image(row, 0, "", {'image_data': imgdata})
			if not loggy_fig:
				plt.close()
			else:
				plt.show()
				plt.close()
			#
			print("Salvando os dados como:"+nome_arquivo+".xlsx")
			writer.save()
			
		else:
			#fuer die variabeln
			"""
				0 - desvpad
				1 - mean 
				2 - missing
				3 - outliers
			"""
			analises = []
			for i in range(0,4):
				analises.append(pd.DataFrame([],index=X.columns))
				
			#olhando para missing media e desvio padrao
			for mes in meses:
				print("Trabalhando no mes: " + str(int(mes)))
				mean = []
				std = []
				missing = []
				outlier = []
				for coluna in X.columns:
				#media e std sem na
					percentil = 0
					desvpad = 0
					media = 0
					out = 0
					numnans = X[coluna][(X[coluna_mes] == mes)].isnull().sum() #Qntd de missing
					percentil = numnans/X.shape[0]
					desvpad = X[coluna][(X[coluna_mes] == mes)].dropna().std()#eu tenho que dar um dropna
					media = X[coluna][(X[coluna_mes] == mes)].dropna().mean()#eu tenho que dar um dropna
					out = X[coluna][(X[coluna_mes] == mes) & (np.abs(X[coluna]-media) > (3*desvpad))].count()#outliers
					mean.append(media)
					std.append(desvpad)
					missing.append(percentil)
					outlier.append(out)
				pdstd = pd.DataFrame(std,columns=[str(mes)],index=X.columns)
				pdmedia = pd.DataFrame(mean,columns=[str(mes)],index=X.columns)
				pdmissing = pd.DataFrame(missing,columns=[str(mes)],index=X.columns)
				pdoutlier = pd.DataFrame(outlier,columns=[str(mes)],index=X.columns)
				analises[0] = pd.concat([analises[0],pdstd],axis=1)
				analises[1] = pd.concat([analises[1],pdmedia],axis=1)
				analises[2] = pd.concat([analises[2],pdmissing],axis=1)
				analises[3] = pd.concat([analises[3],pdoutlier],axis=1)
			#Analisando Estatisticas no Extraídas
			lista_histogramas = set() #não vou querer imprimir isso várias vezes
			for i,analise in enumerate(analises):
				dsvpd = pd.DataFrame([],columns=['DESVPAD_FEATURE'])
				meanpd = pd.DataFrame([],columns=['MEAN_FEATURE'])
				for ind in analise.index:
					dsvpd.loc[ind] = analise.loc[ind].std()
					meanpd.loc[ind] = analise.loc[ind].mean()
					if dsvpd.loc[ind].values > dsvpd_thr:
						lista_histogramas.add(ind)
				analises[i] = pd.concat([analise,dsvpd,meanpd],axis=1)
			#analisar dados no excel	
			sheets = ['DESVPAD','MEDIA','MISSING','OUTLIERS']
			analises[0].to_excel(writer,sheets[0])
			analises[1].to_excel(writer,sheets[1])
			analises[2].to_excel(writer,sheets[2])
			analises[3].to_excel(writer,sheets[3])
			
			for sh in sheets:
				worksheet = writer.sheets[sh]
				for i in range(2,analises[0].shape[0]+2):
					coluna_excel = num_to_col_letters(i)
					worksheet.conditional_format('B'+str(i)+':'+num_to_col_letters(analises[0].shape[1]-1)+str(i), {'type': '3_color_scale','min_type': 'percent','mid_type':'percent','max_type': 'percent'})
			#Salvando Imagens
			worksheet = writer.book.add_worksheet(name="Análises_Imagens")
			row = 0
			for ind in lista_histogramas:
				plt.figure(figsize=(10,5))
				sns.distplot(X[ind].dropna())
				plt.legend(ind)
				plt.title("Distribuição de "+str(ind))
				imgdata = BytesIO()
				plt.savefig(imgdata, format="png")
				imgdata.seek(0)
				worksheet.insert_image(row, 0, "", {'image_data': imgdata})
				row = row+25
				if not loggy_fig:
					plt.close()
				else:
					plt.show()
					plt.close()
								
			#Matriz de covariancia
			correlations = X.corr()
			fig = plt.figure(figsize=(20,20))
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
			worksheet.insert_image(row, 0, "", {'image_data': imgdata})
			if not loggy_fig:
				plt.close()
			else:
				plt.show()
				plt.close()
			#
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