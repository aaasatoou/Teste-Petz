# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Introdução
# MAGIC 
# MAGIC O teste consiste nas seguintes tarefas.
# MAGIC 
# MAGIC - Projete a demanda para cada produto nas lojas para os próximos 30, 60 e 90 dias.
# MAGIC - Trabalhe com R ou Python na ferramenta Databricks;
# MAGIC - O código deverá ser versionado no Github ou Bitbucket.
# MAGIC 
# MAGIC Devido a limitações de tempo, será priorizado o desenvolvimento de um modelo funcional. Caso o modelo de predição seja desenvolvido com certa folga, poderá vir a ser atualizado.
# MAGIC 
# MAGIC O teste será dividido em: Analise Exploratória, Tratamento dos Dados (caso necessário), Desenvolvimento de um modelo baseline (considerando o problema proposto), Teste do Modelo e Predições.

# COMMAND ----------

# Bibliotecas utilizadas

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from prophet import Prophet
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error

# COMMAND ----------

# MAGIC %md
# MAGIC #EDA e Tratamento

# COMMAND ----------

df_vendas = pd.read_csv('/dbfs/FileStore/Vendas.csv', sep=';')
df_vendas.head()

# COMMAND ----------

df_canais = pd.read_csv('/dbfs/FileStore/Canais.csv', sep=';')
df_canais.head()

# COMMAND ----------

df_produtos = pd.read_csv('/dbfs/FileStore/Produtos.csv', sep=';')
df_produtos.head()

# COMMAND ----------

df_unidades = pd.read_csv('/dbfs/FileStore/Unidades_Negócios.csv', sep=';', encoding='latin-1')
df_unidades.head()

# COMMAND ----------

df_lojas = pd.read_csv('/dbfs/FileStore/Lojas.csv', sep=';', encoding='latin-1')
df_lojas.head()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Pelos dados apresentados nas tabelas, podemos ter algumas conclusões. Inicialmente, temos apenas uma tabela fato (Vendas.csv), todas as outras tratam de tabelas dimensão, formando um star schema. Além disso, o problema proposto gira em torno da predição da demanda de **cada** produto **nas lojas**. A partir do enunciado do problema, podemos concluir que:    
# MAGIC   
# MAGIC - Lidaremos **apenas** com produtos, desconsiderando os serviços.
# MAGIC - Realizaremos a predição de **cada um** dos produtos.
# MAGIC - A questão das lojas é um pouco ambígua. A predição deve ser feita para todos os produtos nas lojas, essa informação refere-se ao canal da venda e todos os produtos de e-commerce devem ser descosiderados? Porém, as vendas de e-commerce também partem de alguma loja e então, como prosseguir? Também há a interpretação de que a predição deve ser segmentada por produtos **e** lojas, possívelmente quintuplicando o número de modelos. Será considerado que a predição da demanda deve ser feita para produtos vendidos em lojas, ou seja, desconsiderando as vendas feitas por e-commerce.

# COMMAND ----------

canais_list = df_canais[df_canais['canal'] == 'Loja']['cod_canal'].to_numpy().tolist()
unidades_list = df_unidades[df_unidades['unidade_negocio'] == 'Produtos']['id_unidade_negocio'].to_numpy().tolist()

# COMMAND ----------

df_vendas = df_vendas[df_vendas['id_canal'].isin(canais_list) & df_vendas['id_unidade_negocio'].isin(unidades_list)].copy()

# COMMAND ----------

print(f'Total de Produtos:{len(df_produtos["produto"].unique())}')
print(f'Total de Produtos no dataset de vendas: {len(df_vendas["id_produto"].unique())}')

# COMMAND ----------

# MAGIC %md
# MAGIC Será necessária a predição de 13124 produtos dos 15260 produtos totais. Além disso, a principal caracteristica da tabela fato (conjunto de observações feitas sequencialmente ao longo do tempo) nos indica que trata-se de um problema de série temporal. A abordagem mais simples para este tipo de problema, é o desenvolvimento de um modelo de predição para cada um dos produtos.

# COMMAND ----------

df_vendas.info()

# COMMAND ----------

df_vendas.isnull().values.any()

# COMMAND ----------

# MAGIC %md
# MAGIC Não há valores nulos no dataset. Porém, isso ainda não significa que não há valores incoerentes. O tipo de todas as colunas estão ajustados como 'object', é necessário ajustar o tipo das colunas numéricas e de data.
# MAGIC 
# MAGIC Em especial, as colunas númericas necessitam de certa deliberação. Um problema comum é a aplicação da virgula ou ponto para os separadores de decimal ou de milhar, na situação apresentada a virgula parece ser aplicada como seperador de **decimal**. Porém, criamos números decimais onde espera-se números inteiros (quantidade de vendas, por exemplo). Este tipo de problema seria facilmente resolvido **caso tivessemos acesso ao dicionário de dados** e pudessemos analisar a dimensão dessas unidades. Por exemplo, caso este número representasse a venda de 10 unidades, o número 0.6 representaria 6 unidades, o número 1.8 representaria 18 unidades. A mesma lógica pode ser aplicada para os valores de venda, imposto e custo, caso as unidades estejam em milhar, por exemplo, os números passam a fazer mais sentido. 
# MAGIC 
# MAGIC Sendo assim, até este ponto considera-se que estes dados estão **correto**, uma vez que o dicinário não foi disponibilizado e não há a capacidade de verificar sua coêrencia.

# COMMAND ----------

# Ajuste nos tipos das variáveis

df_vendas['id_data'] = pd.to_datetime(df_vendas['id_data'], format = '%Y-%m-%d')
df_vendas.sort_values(by='id_data', inplace = True) 
df_vendas['qtde_venda'] = df_vendas['qtde_venda'].apply(lambda x: x.replace(',', '.')).astype('float64')
df_vendas['valor_venda'] = df_vendas['valor_venda'].apply(lambda x: x.replace(',', '.')).astype('float64')
df_vendas['valor_imposto'] = df_vendas['valor_imposto'].apply(lambda x: x.replace(',', '.')).astype('float64')
df_vendas['valor_custo'] = df_vendas['valor_custo'].apply(lambda x: x.replace(',', '.')).astype('float64')

# COMMAND ----------

df_vendas[['qtde_venda', 'valor_venda', 'valor_imposto', 'valor_custo']].describe()

# COMMAND ----------

df_vendas[df_vendas['qtde_venda'] < 0]

# COMMAND ----------

# MAGIC %md
# MAGIC Neste ponto há outro problema, o dataset de vendas mostra quantidades negativas. Devido a falta de um dicinário de dados, torna-se dificil julgar qual a natureza destes números, portanto essas 21718 linhas serão consideradas corrompidas e serão excluídas do dataset.

# COMMAND ----------

df_vendas.drop(df_vendas[df_vendas['qtde_venda'] < 0].index, inplace = True)

# COMMAND ----------

df_vendas.shape

# COMMAND ----------

# MAGIC %md
# MAGIC Ainda possuímos 3,8 milhões de linhas, é correto afirmar que os modelos não serão severamente afetados.

# COMMAND ----------

# Série de vendas consolidada
plt.figure(figsize = (25,3))

sns.lineplot(data=df_vendas[['id_data','qtde_venda', 'valor_venda', 'valor_imposto']].groupby('id_data').sum(), x='id_data', y='qtde_venda')
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

# Série para um produto qualquer
# indice 77 escolhido aleatóriamente
plt.figure(figsize = (25,3))

sns.lineplot(data=df_vendas[df_vendas['id_produto'] == df_produtos['produto'].iloc[77]][['id_data','qtde_venda', 'valor_venda', 'valor_imposto']].groupby('id_data').sum(), x='id_data', y='qtde_venda')
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Do gráfico acima notamos outro problema. Apesar de existir mais de 4 milhões de linha no dataset, estas linhas não são uniformemente distribuídas entre todos os produtos. Esse comportamento torna-se um problema por que precisamos do máximo de dados possíveis para todos os produtos para que predições satisfatórias sejam feitas.

# COMMAND ----------

print((df_vendas['id_produto'].value_counts() > 50).sum())

# COMMAND ----------

check_dias = df_vendas[['id_produto', 'id_data']].groupby('id_produto').nunique().sort_values('id_data', ascending = False)
check_dias

# COMMAND ----------

# MAGIC %md
# MAGIC Dos 13107 ids de produto, apenas 5899 aparecem mais de 50 vezes no dataset. Analisando a quantidade de dias distintos para cada ID de produto, encontramos diversos produtos com uma grande escassez de registros, comportamento que inviabiliza o desenvolvimento de um bom modelo de forecast.
# MAGIC 
# MAGIC Para definir os produtos que serão previstos, precisamos definir qual a quantidade ideal de dados para uma boa predição. Numa série temporal, espera-se dados suficientes para que sejam identificados a tendência e a sazonalidade da série em questão. Como não temos informações detalhadas sobre os produtos, serão considerados apenas os produtos que tenham pelo menos 1 ano de observações pois trata-se de um volume grande o suficiente para a detecção da tendência e, caso os produtos possuam sazonalidade (em função da estação do ano, por exemplo), o período também abrange este tipo de informação.
# MAGIC 
# MAGIC **Todos os outros produtos serão desconsiderados pois não possuem dados suficientes para uma boa predição.**

# COMMAND ----------

check_dias[check_dias['id_data'] >= 365]

# COMMAND ----------

# MAGIC %md
# MAGIC As predições e análises a seguir serão feitas para os 1369 produtos restantes.

# COMMAND ----------

lista_ids_drop = check_dias[~(check_dias['id_data'] >= 365)].index
df_vendas.drop(df_vendas[df_vendas['id_produto'].isin(lista_ids_drop)].index, inplace = True)

# COMMAND ----------

# Verificando a série do produto com mais registros.
df_vendas.groupby(['id_produto','id_data']).sum().loc['FT9?!NYW(D/S.`/<DM+V:#', 'qtde_venda'].plot(figsize = (20,3))

# COMMAND ----------

# Verificando a série do produto com menos registros.
df_vendas.groupby(['id_produto','id_data']).sum().loc['M8[-HC+7TN<);QP6]_6U*)', 'qtde_venda'].plot(figsize = (20,3))

# COMMAND ----------

print(df_vendas['id_data'].min())
print(df_vendas['id_data'].max())

# COMMAND ----------

# MAGIC %md
# MAGIC Apesar dos períodos serem semelhantes, uma das séries possui menos registros que a outra. Existem duas abordagens possíveis, a primeira envolve imputar estes dados baseados em algum critério, para series temporais existem algumas técnicas como LOCF (Last Observation Carried Forward), NOCB (Next Observation Carried Backward), média móvel e etc, porém essas técnicas não são uteis para imputar grandes períodos sem dados (períodos de 3 meses como no gráfico visto acima) e por isso serão descartadas.
# MAGIC 
# MAGIC A outra estratégia é utilizar um modelo que consegue lidar bem com dados faltantes, que parece ser mais adequada a essa aplicação.

# COMMAND ----------

# MAGIC %md
# MAGIC # Desenvolvimento do modelo

# COMMAND ----------

# MAGIC %md
# MAGIC ##Baseline

# COMMAND ----------

lista_produtos = df_vendas['id_produto'].unique()
df_timeseries = df_vendas.groupby(['id_produto','id_data']).sum().copy()
df_timeseries

# COMMAND ----------

decom_conf = seasonal_decompose(df_timeseries.loc[('A!^P6LFR+1:<"FF\9:G,X0',),:]['qtde_venda'], period = 7)

fig, (ax1,ax2,ax3, ax4) = plt.subplots(4,1, figsize=(15,8))
decom_conf.observed.plot(ax=ax1,title='Série Original')
decom_conf.trend.plot(ax=ax2,title='Tendência')
decom_conf.seasonal.plot(ax=ax3,title='Sazonalidade')
decom_conf.resid.plot(ax=ax4,title='Ruido')
plt.tight_layout()

# COMMAND ----------

decom_conf = seasonal_decompose(df_vendas.groupby('id_data').sum()['qtde_venda'], period = 7)

fig, (ax1,ax2,ax3, ax4) = plt.subplots(4,1, figsize=(15,8))
decom_conf.observed.plot(ax=ax1,title='Série Original')
decom_conf.trend.plot(ax=ax2,title='Tendência')
decom_conf.seasonal.plot(ax=ax3,title='Sazonalidade')
decom_conf.resid.plot(ax=ax4,title='Ruido')
plt.tight_layout()

# COMMAND ----------

from statsmodels.tsa.stattools import adfuller
 
result = adfuller(df_vendas.groupby('id_data').sum()['qtde_venda'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
 
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# COMMAND ----------

# MAGIC %md
# MAGIC Apesar dos gráficos indicarem certa sazonalidade na série consolidada dos produtos, vemos no teste de Dickey-Fuller que a série pode sim ser considerada estacionária. Nesta situação, podemos utilizar o modelo prophet, desenvolvido pela Meta. O prophet funciona bem com dados estacionários e que mostram uma certa aleatoriedade (lembrando sazonalidade).

# COMMAND ----------

prophet_basic = Prophet(yearly_seasonality = 20)
prophet_basic.add_seasonality(name='weekly', period=7, fourier_order=5, prior_scale=0.5)

prophet_basic.fit(df_vendas.groupby('id_data').sum()['qtde_venda'].resample('D').last().reset_index().rename(columns = {'id_data': 'ds', 'qtde_venda': 'y'}))

future = prophet_basic.make_future_dataframe(periods=30)
forecast = prophet_basic.predict(future)

# COMMAND ----------

pd.concat([df_vendas.groupby('id_data').sum()['qtde_venda'].resample('D').last().reset_index().rename(columns = {'id_data': 'ds', 'qtde_venda': 'y'}).set_index('ds'), forecast[['ds', 'yhat']].rename(columns = {'yhat':'y'}).set_index('ds')]).plot(figsize = (30,5))

# COMMAND ----------

# MAGIC %md
# MAGIC ##Modelo Final
# MAGIC Seguindo o baseline proposto para a série consolidada de dados, o modelo prophet será treinado e aplicada para cada um dos produtos.

# COMMAND ----------

# Os 30 ultimos dias de dados serão utilizados como teste
# Normamente utilizariamos TimeSeriesSplit para utilizar um cross validator, porém, são muitos produtos e a inconsistencia dos dados pode ser problematica
df_teste = pd.DataFrame()
predicoes = pd.DataFrame()
for i in lista_produtos:
    timeseries = df_timeseries.loc[(i,),:]['qtde_venda'].resample('D').last().reset_index().rename(columns = {'id_data':'ds', 'qtde_venda':'y'})

    teste = timeseries[-30:].copy()
    treino = timeseries[0:-30].copy()

    transformer = RobustScaler() # Escolha do RobustScaler parte da ideia que são dados de venda, com vários outliers naturais pelo caminho
    transformer.fit(treino['y'].to_numpy().reshape(-1, 1))
    treino['y'] = transformer.transform(treino['y'].to_numpy().reshape(-1, 1))
    teste['y'] = transformer.transform(teste['y'].to_numpy().reshape(-1, 1))


    prophet_basic = Prophet(yearly_seasonality = 20)
    prophet_basic.add_seasonality(name='weekly', period=7, fourier_order=5, prior_scale=0.5)

    prophet_basic.fit(treino)
    
    
    for periodo in [30, 60, 90]:
        future = prophet_basic.make_future_dataframe(periods=periodo)
        forecast = prophet_basic.predict(future)
        forecast['yhat'] = transformer.inverse_transform(forecast['yhat'].to_numpy().reshape(-1, 1))
        forecast['periodo'] = periodo
        forecast['produto'] = i
        
        predicoes = pd.concat([predicoes, forecast[['ds','yhat', 'periodo', 'produto']]])
        
        
    forecast_teste = prophet_basic.predict(teste[['ds']])   
    forecast_teste['yhat'] = transformer.inverse_transform(forecast_teste['yhat'].to_numpy().reshape(-1, 1))
    forecast_teste['produto'] = i
    teste['y'] = transformer.inverse_transform(teste['y'].to_numpy().reshape(-1, 1))
    
    df_teste = pd.concat([df_teste, pd.concat([forecast_teste[['ds', 'yhat', 'produto']], teste[['y']].reset_index(drop = True)], axis = 1)])
    

# COMMAND ----------

predicoes = predicoes.set_index(['produto','periodo','ds'])
predicoes

# COMMAND ----------

df_teste = df_teste.set_index(['produto','ds'])
df_teste

# COMMAND ----------

metrics_dict = {}
for i in lista_produtos:
    y_true = df_teste.loc[(i,),:].dropna()['y']
    y_pred = df_teste.loc[(i,),:].dropna()['yhat']
    metrics_dict[f'{i}'] = mean_absolute_error(y_true, y_pred)

metrics_dict
