# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 21:43:12 2021

@author: leonardo
"""

import pandas as pd
import math
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder  #transformar variaveis categoricas em numero
from tqdm import tqdm # verifica o progresso de uma tarefa
from statistics import mean
from sklearn.metrics import mean_squared_error, r2_score


from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, SGDRegressor


# instalar pip install pandas-profilling
from pandas_profiling import ProfileReport

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%d-%m-%Y%H_%M_%S")

# =============================================================================
# Extraindo dados de Treinamento
# =============================================================================
ord_enc = OrdinalEncoder()

df = pd.read_csv('conjunto_de_treinamento.csv')

df = df.drop([  'Id',
                'tipo',
                'bairro',
                'tipo_vendedor',
                'area_extra',
                'diferenciais',
                'estacionamento',
                'churrasqueira',
                'piscina',
                'playground',
                'quadra',
                's_festas',
                's_jogos',
                's_ginastica',
                'sauna',
                'vista_mar'
               ],axis=1)


# =============================================================================
# df["bairro"] = ord_enc.fit_transform(df[["bairro"]])
# =============================================================================

# =============================================================================
# Extraindo dados do kaggle
# =============================================================================

df_teste_kaggle = pd.read_csv('conjunto_de_teste.csv')

df_teste_kaggle = df_teste_kaggle.drop([
                'tipo',
                'bairro',
                'tipo_vendedor',
                'area_extra',
                'diferenciais',
                'estacionamento',
                'churrasqueira',
                'piscina',
                'playground',
                'quadra',
                's_festas',
                's_jogos',
                's_ginastica',
                'sauna',
                'vista_mar'
               ],axis=1)

# =============================================================================
# df_teste_kaggle["bairro"] = ord_enc.fit_transform(df_teste_kaggle[["bairro"]])
# =============================================================================

#------------------------------------------------------------------------------
# Transferir valores dos atributos e rótulos para arrays X e Y
#------------------------------------------------------------------------------

caracteristicas_imovel = df.iloc[:,:-1].to_numpy()
preco = df.iloc[:,-1].to_numpy()

caracteristicas_imovel_kaggle = df_teste_kaggle.iloc[:,1:].to_numpy()


#------------------------------------------------------------------------------
# Particionar X e Y em conjunto de treinamento e conjunto de teste
#------------------------------------------------------------------------------

x_treino, x_teste, y_treino, y_teste = train_test_split(
    caracteristicas_imovel,
    preco,
    test_size = 0.3,
    random_state = 1
    )

#------------------------------------------------------------------------------
# Ajustar a escala dos atributos
#------------------------------------------------------------------------------

std = StandardScaler()

std_caracteristicas_imovel = std.fit(x_treino)

# Normalizando os valores das caracteristicas
caracteristicas_imovel = std_caracteristicas_imovel.transform(x_treino)
caracteristicas_imovel_teste = std_caracteristicas_imovel.transform(x_teste)
caracteristicas_imovel_kaggle  = std_caracteristicas_imovel.transform(caracteristicas_imovel_kaggle)


# # Normalizando os precos de treino e teste

std_preco = std.fit(y_treino.reshape(-1, 1))

preco = std_preco.transform(y_treino.reshape(-1, 1))
preco_teste = std_preco.transform(y_teste.reshape(-1, 1))


# =============================================================================
# Removendo Outliers
# =============================================================================

standard_deviation = np.std(preco)
distance_from_mean = abs(preco - np.mean(preco))
max_deviations = 2
not_outlier = distance_from_mean < max_deviations * standard_deviation
no_outliers = preco[not_outlier]

index_no_outlier = np.where(not_outlier == False)[0]


sns.boxplot(no_outliers)


# Remover as linhas dos indices de outliers

print( 'caracteristicas com outliers: ',len(caracteristicas_imovel), ' precos com outliers:' ,len(preco))

print("Precos outliers normalizados")
print(preco[index_no_outlier])
print("Precos outliers reais")
print(std_preco.inverse_transform(preco[index_no_outlier]))

caracteristicas_imovel = np.delete(caracteristicas_imovel, index_no_outlier, axis=0)

preco = preco[not_outlier]

print( 'caracteristicas sem outliers: ',len(caracteristicas_imovel), ' precos sem outliers:' ,len(preco))


# instanciar e ajsutar um objeto PolynomialFeatures
k = 1
pf = PolynomialFeatures(degree=k)
pf = pf.fit(caracteristicas_imovel)



# transforma a matriz de caracteristicas incluindo atributos polinomiais
caracteristicas_imovel_poly = pf.transform(caracteristicas_imovel)
caracteristicas_imovel_teste_poly = pf.transform(caracteristicas_imovel_teste)
caracteristicas_imovel_kaggle_poly = pf.transform(caracteristicas_imovel_kaggle)

na = caracteristicas_imovel_poly.shape[1]

# =============================================================================
# Aplicando LinearRegression
# =============================================================================


# transforma a matriz de caracteristicas incluindo atributos polinomiais
caracteristicas_imovel_poly = pf.transform(caracteristicas_imovel)
caracteristicas_imovel_teste_poly = pf.transform(caracteristicas_imovel_teste)
caracteristicas_imovel_kaggle_poly = pf.transform(caracteristicas_imovel_kaggle)

regressor_linear = LinearRegression()

regressor_linear = regressor_linear.fit(caracteristicas_imovel_poly,preco)

linear_preco_resposta = regressor_linear.predict(caracteristicas_imovel_poly)
linear_preco_resposta_teste  = regressor_linear.predict(caracteristicas_imovel_teste_poly)
linear_preco_resposta_kaggle = regressor_linear.predict(caracteristicas_imovel_kaggle_poly)

rmspe = np.sqrt(
            np.mean(
                np.square(((preco - linear_preco_resposta) / preco)),
                axis=0
            )
        )

mse_in  = mean_squared_error(preco,linear_preco_resposta)
rmse_in = math.sqrt(mse_in)
r2_in   = r2_score(preco,linear_preco_resposta)

mse_out  = mean_squared_error(preco_teste,linear_preco_resposta_teste)
rmse_out = math.sqrt(mse_out)
r2_out   = r2_score(preco_teste,linear_preco_resposta_teste)

print(f'--------------------------------LINEAR-{k}------------------------------------')
print(' NA        RMSE_IN       R^2 IN       RMSE_OUT       R^2 OUT    RMSPE-saga')
print('%4d  %12.4f  %12.4f  %12.4f  %12.4f %12.4f' % ( na , rmse_in , r2_in, rmse_out,  r2_out, rmspe))


preco_linear = std_preco.inverse_transform(linear_preco_resposta_kaggle).reshape(1, -1)

dict_resultado = {'Id': df_teste_kaggle['Id'], 'preco': preco_linear.tolist()[0]}
df_resultado = pd.DataFrame(data=dict_resultado)
df_resultado.to_csv (f'regressao_linear_preco_grau{k} - {current_time}.csv', index = False, header=True)

# =============================================================================
# Aplicando Ridge
# =============================================================================

regressor_ridge = Ridge(alpha=125000)

regressor_ridge = regressor_ridge.fit(caracteristicas_imovel_poly,preco)

ridge_preco_resposta = regressor_ridge.predict(caracteristicas_imovel_poly)
ridge_preco_resposta_teste  = regressor_ridge.predict(caracteristicas_imovel_teste_poly)
ridge_preco_resposta_kaggle = regressor_ridge.predict(caracteristicas_imovel_kaggle_poly)


rmspe = np.sqrt(
                np.mean(
                    np.square(((preco - ridge_preco_resposta) / preco)),
                    axis=0
                )
            )

mse_in  = mean_squared_error(preco,ridge_preco_resposta)
rmse_in = math.sqrt(mse_in)
r2_in   = r2_score(preco,ridge_preco_resposta)

mse_out  = mean_squared_error(preco_teste,ridge_preco_resposta_teste)
rmse_out = math.sqrt(mse_out)
r2_out   = r2_score(preco_teste,ridge_preco_resposta_teste)



print(f'--------------------------------RIDGE-{k}------------------------------------')
print(' NA        RMSE_IN       R^2 IN       RMSE_OUT       R^2 OUT    RMSPE-saga')
print('%4d  %12.4f  %12.4f  %12.4f  %12.4f %12.4f' % ( na , rmse_in , r2_in, rmse_out,  r2_out, rmspe))


preco_ridge = std_preco.inverse_transform(ridge_preco_resposta_kaggle).reshape(1, -1)

dict_resultado = {'Id': df_teste_kaggle['Id'], 'preco': preco_ridge.tolist()[0]}
df_resultado = pd.DataFrame(data=dict_resultado)
df_resultado.to_csv (f'regressao_ridge_preco_grau{k} - {current_time}.csv', index = False, header=True)


# =============================================================================
# Aplicando Ridge
# =============================================================================

regressor_lasso = Lasso(alpha=125000,
                        max_iter=10000
                       )


regressor_lasso = regressor_lasso.fit(caracteristicas_imovel_poly,preco)

lasso_preco_resposta = regressor_lasso.predict(caracteristicas_imovel_poly)
lasso_preco_resposta_teste  = regressor_lasso.predict(caracteristicas_imovel_teste_poly)
lasso_preco_resposta_kaggle = regressor_lasso.predict(caracteristicas_imovel_kaggle_poly)

rmspe = np.sqrt(
                np.mean(
                    np.square(((preco - lasso_preco_resposta) / preco)),
                    axis=0
                )
            )

mse_in  = mean_squared_error(preco,lasso_preco_resposta)
rmse_in = math.sqrt(mse_in)
r2_in   = r2_score(preco,lasso_preco_resposta)

mse_out  = mean_squared_error(preco_teste,lasso_preco_resposta_teste)
rmse_out = math.sqrt(mse_out)
r2_out   = r2_score(preco_teste,lasso_preco_resposta_teste)


print(f'--------------------------------LASSO-{k}------------------------------------')
print(' NA        RMSE_IN       R^2 IN       RMSE_OUT       R^2 OUT    RMSPE')
print('%4d  %12.4f  %12.4f  %12.4f  %12.4f %12.4f' % ( na , rmse_in , r2_in, rmse_out,  r2_out, rmspe))


preco_lasso = std_preco.inverse_transform(lasso_preco_resposta_kaggle).reshape(1, -1)

dict_resultado = {'Id': df_teste_kaggle['Id'], 'preco': preco_lasso.tolist()[0]}
df_resultado = pd.DataFrame(data=dict_resultado)
df_resultado.to_csv (f'regressao_lasso_preco_grau{k} - {current_time}.csv', index = False, header=True)


# =============================================================================
# Aplicando ElasticNet
# =============================================================================


ENreg = ElasticNet(alpha=1, l1_ratio=0.5, normalize=False)


ENreg = ENreg.fit(caracteristicas_imovel_poly,preco)

ENreg_preco_resposta = ENreg.predict(caracteristicas_imovel_poly)
ENreg_preco_resposta_teste  = ENreg.predict(caracteristicas_imovel_teste_poly)
ENreg_preco_resposta_kaggle = ENreg.predict(caracteristicas_imovel_kaggle_poly)

rmspe = np.sqrt(
                np.mean(
                    np.square(((preco - ENreg_preco_resposta) / preco)),
                    axis=0
                )
            )

mse_in  = mean_squared_error(preco,ENreg_preco_resposta)
rmse_in = math.sqrt(mse_in)
r2_in   = r2_score(preco,ENreg_preco_resposta)

mse_out  = mean_squared_error(preco_teste,ENreg_preco_resposta_teste)
rmse_out = math.sqrt(mse_out)
r2_out   = r2_score(preco_teste,ENreg_preco_resposta_teste)

print(f'----------------------------ElasticNet-{k}-----------------------------------')
print(' NA        RMSE_IN       R^2 IN       RMSE_OUT       R^2 OUT    RMSPE')
print('%4d  %12.4f  %12.4f  %12.4f  %12.4f %12.4f' % ( na , rmse_in , r2_in, rmse_out,  r2_out, rmspe))


preco_ENreg = std_preco.inverse_transform(ENreg_preco_resposta_kaggle).reshape(1, -1)

dict_resultado = {'Id': df_teste_kaggle['Id'], 'preco': preco_ENreg.tolist()[0]}
df_resultado = pd.DataFrame(data=dict_resultado)
df_resultado.to_csv (f'regressao_ElasticNet_preco_grau{k} - {current_time}.csv', index = False, header=True)



# =============================================================================
# Aplicando SGDRegressor
# =============================================================================

SGDreg = SGDRegressor(max_iter=100, tol=1e-3)


SGDreg = SGDreg.fit(caracteristicas_imovel_poly,preco)

SGDreg_preco_resposta = SGDreg.predict(caracteristicas_imovel_poly)
SGDreg_preco_resposta_teste  = SGDreg.predict(caracteristicas_imovel_teste_poly)
SGDreg_preco_resposta_kaggle = SGDreg.predict(caracteristicas_imovel_kaggle_poly)

rmspe = np.sqrt(
                np.mean(
                    np.square(((preco - SGDreg_preco_resposta) / preco)),
                    axis=0
                )
            )

mse_in  = mean_squared_error(preco,SGDreg_preco_resposta)
rmse_in = math.sqrt(mse_in)
r2_in   = r2_score(preco,SGDreg_preco_resposta)

mse_out  = mean_squared_error(preco_teste,SGDreg_preco_resposta_teste)
rmse_out = math.sqrt(mse_out)
r2_out   = r2_score(preco_teste,SGDreg_preco_resposta_teste)


print(f'----------------------------SGD-{k}-----------------------------------')
print(' NA        RMSE_IN       R^2 IN       RMSE_OUT       R^2 OUT    RMSPE')
print('%4d  %12.4f  %12.4f  %12.4f  %12.4f %12.4f' % ( na , rmse_in , r2_in, rmse_out,  r2_out, rmspe))


preco_SGDreg = std_preco.inverse_transform(SGDreg_preco_resposta_kaggle).reshape(1, -1)

dict_resultado = {'Id': df_teste_kaggle['Id'], 'preco': preco_SGDreg.tolist()[0]}
df_resultado = pd.DataFrame(data=dict_resultado)
df_resultado.to_csv (f'regressao_SGD_preco_grau{k} - {current_time}.csv', index = False, header=True)
