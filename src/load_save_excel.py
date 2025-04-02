# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 21:22:15 2025

@author: joaoc
"""
import pandas as pd
import glob
import os

# Diretório onde os arquivos estão salvos
diretorio = 'C:/Users/joaoc/Documents/MT/pagamento_servidores/data/raw/concat'

# Padrão dos nomes dos arquivos (ex: todos os .xlsx)
arquivos = glob.glob(os.path.join(diretorio, '*.xlsx'))

# Lista para armazenar os DataFrames individuais
dfs = []

for arquivo in arquivos:
    # Ler cada arquivo Excel
    df = pd.read_excel(arquivo)
    dfs.append(df)

# Concatenar todos os DataFrames em um único DataFrame
df_final = pd.concat(dfs, ignore_index=True)

# Salvar em um novo arquivo Excel
df_final.to_excel('C:/Users/joaoc/Documents/MT/pagamento_servidores/data/raw/servidores_mes_rubricas.xlsx', index=False)