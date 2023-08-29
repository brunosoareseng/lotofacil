import os
import pandas as pd

# Defina o diretório onde estão os arquivos .xls
diretorio = './jogos/'

# Lista para armazenar os DataFrames de cada arquivo .xls
dataframes = []

# Percorre todos os arquivos no diretório
for arquivo in os.listdir(diretorio):
    if arquivo.endswith('.xlsx'):
        caminho_arquivo = os.path.join(diretorio, arquivo)
        df = pd.read_excel(caminho_arquivo, engine='openpyxl')
        dataframes.append(df)

# Concatena todos os DataFrames em um único DataFrame
df_final = pd.concat(dataframes, ignore_index=True)

# Exibe o DataFrame final
print(df_final)
print(df_final.describe()) 
df_final.describe().to_excel('estatistica_res.xlsx')

