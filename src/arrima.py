# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 14:33:25 2025

@author: joaoc
"""
import process_servants as ps
import pandas as pd
from janitor import clean_names
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

def process_registration_data(file_path, reference_date_str, cut_date):
    """Process registration data from Excel file"""
    
    df = (
        pd.read_excel(file_path)
        .pipe(clean_names)
        .pipe(ps.convert_rendim_to_numeric)
        .pipe(ps.process_month_column)
        )    
    
    # Converter para datetime
    reference_date = pd.to_datetime(reference_date_str)
    
    # Calcular data de corte
    cutoff_date = reference_date - pd.DateOffset(months=cut_date)
    
    # Filtrar linhas
    df = df[(df['mes'] >= cutoff_date) & (df['mes'] <= reference_date)].copy()
    return df    
    
    df_rendim = df[['cpf_servidor', 'mes', 'rendim']]
    df_rendim = df_rendim.groupby(['cpf_servidor', 'mes'])['rendim'].sum().reset_index()
    df_rendim = df_rendim.pivot(index = 'cpf_servidor', columns='mes', values='rendim').reset_index()
    df_rendim = df_rendim.fillna(0)
    df_rendim = df_rendim.melt(id_vars =  'cpf_servidor', var_name='mes', value_name='rendim')
      
        
    return df_rendim

# Save Processed Data
def processed_data(brute_data_file_path, file_name, reference_date_str, cut_date):
    registration_df = process_registration_data(brute_data_file_path, reference_date_str, cut_date)
    
    processed_path  = "C:/Users/joaoc/Documents/MT/pagamento_servidores/data/processed/" + file_name + ".xlsx"
    print("Processed data shapes:")
    print(f"Registration data: {registration_df.shape}")
    
    registration_df.to_excel(processed_path, index=False)
        
    return registration_df
def analise_exploratoria(servidor):
    return

# Example usage
if __name__ == '__main__':

    # Set reference date
    brute_file_path = 'C:/Users/joaoc/Documents/MT/pagamento_servidores/data/raw/servidores_mes.xlsx'
    cut_date = 12
    reference_date_str = '2022-04-01'
    
    registration_df = processed_data(brute_file_path, "dados_12_meses_cadastro_e_std", reference_date_str ,cut_date)
    
   