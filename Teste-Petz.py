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
# MAGIC Devido a limitações de tempo, será priorizado o desenvolvimento de um modelo funcional. Caso o modelo de predição seja desenvolvido com certa folga, novas versões serão desenvolvidas.
# MAGIC 
# MAGIC O teste será dividido em: Analise Exploratória, Tratamento dos Dados (caso necessário), Desenvolvimento de um modelo baseline (considerando o problema proposto), Teste do Modelo e Predições.

# COMMAND ----------

import pandas as pd

df_vendas = pd.read_csv('/dbfs/FileStore/PetzTest/Vendas.csv', sep=';')
df_vendas.head()

# COMMAND ----------

df_canais = pd.read_csv('/dbfs/FileStore/PetzTest/Canais.csv', sep=';')
df_canais.head()

# COMMAND ----------

df_produtos = pd.read_csv('/dbfs/FileStore/PetzTest/Produtos.csv', sep=';')
df_produtos.head()

# COMMAND ----------

df_unidades = pd.read_csv('/dbfs/FileStore/PetzTest/Unidades_Negócios.csv', sep=';', encoding='latin-1')
df_unidades.head()

# COMMAND ----------

df_lojas = pd.read_csv('/dbfs/FileStore/PetzTest/Lojas.csv', sep=';', encoding='latin-1')
df_lojas.head()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Pelos dados apresentados nas tabelas, podemos já ter algumas conclusões. Inicialmente, temos apenas uma tabela fato (Vendas.csv), todas as outras tratam de tabelas dimensão, formando um star schema. Além disso, o problema proposto gira em torno da predição da demanda de **cada** produto

# COMMAND ----------

len(df_produtos['produto'].unique())

# COMMAND ----------

# MAGIC %md
# MAGIC Será necessária a predição de 15260 produtos diferentes
