import process_servants as ps
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import pandas as pd
from janitor import clean_names

def process_registration_data_last_month_rendim(file_path, reference_date_str):
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
    cutoff_date = reference_date - pd.DateOffset(months=12)
    
    # Filtrar linhas
    df = df[(df['mes'] >= cutoff_date) & (df['mes'] <= reference_date)].copy()
    df_rendim = ps.categorizar_rendimentos(df, reference_date )
    df_rendim = df_rendim.drop('mes', axis=1)
    

    # Calculate Std of Redim by CPF
    df_std = ps.calculate_std_by_cpf(df)
    # Drop para nao contabilizar nas mudancas 
    df = df.drop('rendim', axis=1)
    
    # Filter for reference month
    current_month_data = df[df['mes'] == reference_date].copy()
    current_month_data = (
        ps.track_registration_changes(current_month_data, df, reference_date)
        .pipe(ps.process_service_time)
        .merge(df_std, on='cpf_servidor', how='left')
        .merge(df_rendim, on='cpf_servidor', how='left')
        .pipe(ps.process_esc_cargo)
        .pipe(ps.process_ingresso_spub)
        .pipe(ps.process_mt_entrada)
        .pipe(ps.process_permanence_allowance)
        .pipe(ps.process_exclusion_column)
        .pipe(ps.process_reversion)
        .pipe(ps.process_aposentadoria)
        .pipe(ps.processar_nivel_funcao)
        .pipe(ps.processar_situacao_funcional)
        )

    # Drop columns column if exists
    if 'mes' in df.columns:
        current_month_data = current_month_data.drop('mes', axis=1)
    if 'mes_ing_spub' in df.columns:
        current_month_data = current_month_data.drop('mes_ing_spub', axis=1)
       

    return current_month_data

# Save Processed Data
def processed_data(brute_data_file_path, reference_date, file_name, tipo: str):
    
    ###Verificar quanto funcao process_registration_data chamar
    if( tipo == 'last_month'):
        registration_df = process_registration_data_last_month_rendim(brute_data_file_path, reference_date)
    if( tipo == 'all_months'):
        return
    
    processed_path  = "C:/Users/joaoc/Documents/MT/pagamento_servidores/data/processed/" + file_name + ".xlsx"
    print("Processed data shapes:")
    print(f"Registration data: {registration_df.shape}")
    
    registration_df.to_excel(processed_path, index=False)
    
    
    return registration_df

# Example usage
if __name__ == '__main__':

    # Set reference date
    reference_date = '2022-04-01'
    brute_file_path = 'C:/Users/joaoc/Documents/MT/pagamento_servidores/data/raw/servidores_mes.xlsx'
    
    registration_df = processed_data(brute_file_path, reference_date, "dados_12_meses_cadastro_e_std", 'last_month')
    registration_df = registration_df[registration_df["rendim"] < 4]
    
    
    registration_df_clone = registration_df
    registration_df_clone = registration_df_clone.drop('cpf_servidor', axis = 1 )
    registration_df_clone['tempo_servico'] = pd.to_numeric(registration_df_clone['tempo_servico'], errors='coerce').fillna(0.0)
    registration_df_clone['std_rendim'] = pd.to_numeric(registration_df_clone['std_rendim'], errors='coerce').fillna(0.0)
    registration_df_clone = pd.get_dummies(registration_df_clone, columns=['situacao_funcional', 'exclusao', 'ingresso', 'nivel_funcao', 'esc_cargo', 'aposentadoria'])
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(registration_df_clone[['rendim', 'std_rendim', 'tempo_servico']])
    
    
    # Treinar o modelo
    model = IsolationForest(
        n_estimators=100,  # número de árvores
        contamination=0.04,  
        random_state=42
    )

    model.fit(df_scaled)
    
    anomalias = model.predict(df_scaled)
    registration_df['Anomalia'] = anomalias
    
    fraudes_suspeitas = registration_df[registration_df['Anomalia'] == -1]
