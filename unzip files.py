# Databricks notebook source
# MAGIC %sh unzip /dbfs/FileStore/Vendas.zip

# COMMAND ----------

# MAGIC %fs ls file:/databricks/driver/ 

# COMMAND ----------

dbutils.fs.mv("file:/databricks/driver/Vendas.csv", "dbfs:/FileStore")
