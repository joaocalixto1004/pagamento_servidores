import pandas as pd
import numpy as np
from janitor import clean_names
from typing import Optional
from datetime import datetime
from dateutil.relativedelta import relativedelta


# =============================================
# Helper Functions 
# =============================================
def convert_month_names(month_str):
    """Convert Portuguese month abbreviations to English"""
    month_map = {
        'Fev': 'Feb',
        'Abr': 'Apr',
        'Mai': 'May',
        'Ago': 'Aug',
        'Set': 'Sep',
        'Out': 'Oct',
        'Dez': 'Dec'
    }
    
    if not isinstance(month_str, str):
        return month_str
        
    month_abbr = month_str[:3]
    if month_abbr in month_map:
        return f"{month_map[month_abbr]} {month_str[4:]}"
    return month_str

def convert_redim_values(rendim):
    try:
        # Remove pontos de milhar e substitui vírgula decimal por ponto
        if isinstance(rendim, str):
            # Remove caracteres não numéricos (exceto , e .)
            rendim_clean = rendim.replace('R$', '').replace('%', '').strip()
            rendim_clean = rendim_clean.replace('.', '').replace(',', '.')
            return float(rendim_clean)
        else:
            return float(rendim)  # Caso já seja numérico
    except (ValueError, TypeError, AttributeError):
        return pd.NA  # Retorna NaN para valores inválidos
    
# =============================================
# Core Processing Functions 
# =============================================

def check_registration_changes(df, reference_date, months_back):
    """Check for registration changes in the specified period"""
    mask = df['mes'] >= (reference_date - relativedelta(months=months_back))
    
    # Group by CPF and check if any column has more than one unique value
    changes = df[mask].drop('mes', axis=1)\
        .groupby('cpf_servidor')\
        .agg(lambda x: x.nunique() > 1)\
        .any(axis=1)

    return changes[changes].index.tolist()

def convert_rendim_to_numeric(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    
    
    """
    Converte a coluna 'rendim' para valores numéricos de forma segura.
    
    Parâmetros:
        df (pd.DataFrame): DataFrame contendo a coluna 'rendim'
        inplace (bool): Se False (padrão), retorna uma cópia do DataFrame modificado.
                        Se True, modifica o DataFrame original.
    
    Retorna:
        pd.DataFrame: DataFrame com a coluna 'rendim' convertida para float
    
    Levanta:
        ValueError: Se a coluna 'rendim' não existir no DataFrame
        TypeError: Se a conversão falhar para valores críticos
    
    Exemplo:
        >>> df = pd.DataFrame({'rendim': ['1.500,50', 'R$ 2.000', '15%']})
        >>> df = convert_rendim_to_numeric(df)
        >>> print(df.dtypes)
        rendim    float64
        dtype: object
    """
    # Validação da coluna
    if 'rendim' not in df.columns:
        raise ValueError("Coluna 'rendim' não encontrada no DataFrame")
    
    # Trabalhar em cópia ou no original
    working_df = df if inplace else df.copy()
    
    try:
        # Converter apenas se for tipo objeto
        if working_df['rendim'].dtype == 'O':
            working_df['rendim'] = (
                working_df['rendim']
                .astype(str)
                .apply(convert_redim_values)
                .pipe(pd.to_numeric, errors='coerce')  # Conversão final segura
            )
            
            # Verificar taxa de sucesso na conversão
            conversion_rate = working_df['rendim'].notna().mean()
            if conversion_rate < 0.9:
                raise TypeError(f"Taxa de conversão crítica: {conversion_rate:.1%} de valores válidos")
                
        return working_df
    
    except Exception as e:
        raise TypeError(f"Falha na conversão da coluna 'rendim': {str(e)}") from e

def calculate_std_by_cpf(df: pd.DataFrame, 
                        group_col: str = 'cpf_servidor',
                        value_col: str = 'rendim',
                        new_col_name: str = 'std_rendim') -> pd.DataFrame:
    """
    Calcula o desvio padrão de uma coluna numérica agrupado por CPF
    
    Parâmetros:
        df (pd.DataFrame): DataFrame original
        group_col (str): Coluna de agrupamento (padrão: 'cpf_servidor')
        value_col (str): Coluna numérica para cálculo (padrão: 'rendim')
        new_col_name (str): Nome da nova coluna (padrão: 'std_rendim')
    
    Retorna:
        pd.DataFrame: DataFrame com duas colunas: group_col e new_col_name
    
    Levanta:
        ValueError: Se as colunas especificadas não existirem
        TypeError: Se a coluna value_col não for numérica
    """
    
    # Validação das colunas
    required_cols = {group_col, value_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Colunas faltantes: {missing_cols}")
    
    # Verificação de tipo numérico
    if not np.issubdtype(df[value_col].dtype, np.number):
        raise TypeError(f"Coluna {value_col} deve ser numérica")
    
    # Cálculo do desvio padrão
    result_df = (
        df
        .groupby(group_col, as_index=False)
        [value_col]
        .std(ddof=1)
        .rename(columns={value_col: new_col_name})
    )
    
    return result_df


def calculate_mean_by_cpf(df: pd.DataFrame, 
                        group_col: str = 'cpf_servidor',
                        value_col: str = 'rendim',
                        new_col_name: str = 'mean_rendim') -> pd.DataFrame:
    """
    Calcula o desvio padrão de uma coluna numérica agrupado por CPF
    
    Parâmetros:
        df (pd.DataFrame): DataFrame original
        group_col (str): Coluna de agrupamento (padrão: 'cpf_servidor')
        value_col (str): Coluna numérica para cálculo (padrão: 'rendim')
        new_col_name (str): Nome da nova coluna (padrão: 'std_rendim')
    
    Retorna:
        pd.DataFrame: DataFrame com duas colunas: group_col e new_col_name
    
    Levanta:
        ValueError: Se as colunas especificadas não existirem
        TypeError: Se a coluna value_col não for numérica
    """
    
    # Validação das colunas
    required_cols = {group_col, value_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Colunas faltantes: {missing_cols}")
    
    # Verificação de tipo numérico
    if not np.issubdtype(df[value_col].dtype, np.number):
        raise TypeError(f"Coluna {value_col} deve ser numérica")
    
    # Cálculo do desvio padrão
    result_df = (
        df
        .groupby(group_col, as_index=False)
        [value_col]
        .std(ddof=1)
        .rename(columns={value_col: new_col_name})
    )
    
    return result_df

def process_month_column(df: pd.DataFrame, 
                        date_col: str = 'mes',
                        inplace: bool = False) -> pd.DataFrame:
    """
    Processa a coluna de datas convertendo meses em português para formato datetime
    
    Parâmetros:
        df (pd.DataFrame): DataFrame contendo a coluna de datas
        date_col (str): Nome da coluna de datas (padrão: 'mes')
        inplace (bool): Se False (padrão), retorna uma cópia do DataFrame modificado
                        Se True, modifica o DataFrame original
    
    Retorna:
        pd.DataFrame: DataFrame com a coluna processada
    
    Levanta:
        ValueError: Se a coluna especificada não existir no DataFrame
        TypeError: Se a conversão de data falhar para múltiplos valores
    
    Exemplo:
        >>> df = pd.DataFrame({'mes': ['Fev 2023', 'Abr 2022', 'Dez 2021']})
        >>> processed_df = process_month_column(df)
        >>> print(processed_df['mes'].dtype)
        datetime64[ns]
    """
    # Validação inicial
    if date_col not in df.columns:
        raise ValueError(f"Coluna '{date_col}' não encontrada no DataFrame")
    
    # Trabalhar em cópia ou no original
    working_df = df if inplace else df.copy()
    
    try:
        # Converter nomes de meses
        working_df[date_col] = working_df[date_col].apply(convert_month_names)
        
        # Converter para datetime com tratamento de erros
        working_df[date_col] = pd.to_datetime(
            working_df[date_col],
            format='%b %Y',
            errors='coerce'
        )
        
        # Verificar qualidade da conversão
        conversion_rate = working_df[date_col].notna().mean()
        if conversion_rate < 0.8:
            raise TypeError(f"Taxa de conversão baixa: {conversion_rate:.1%} de datas válidas")
            
        return working_df
    
    except Exception as e:
        error_msg = f"Falha no processamento da coluna {date_col}: {str(e)}"
        raise TypeError(error_msg) from e
        
def track_registration_changes(current_month_data, df, reference_date, change_periods=[1, 3, 6, 12]):
    """
    Track registration changes over specified periods and mark changes in the data.

    Parameters:
        current_month_data (pd.DataFrame): DataFrame for the current month
        df (pd.DataFrame): Full dataset
        reference_date (datetime): Reference date for change tracking
        change_periods (list): List of periods (in months) to track changes

    Returns:
        pd.DataFrame: DataFrame with change indicators for each period
    """
    for period in change_periods:
        col_name = f'mudanca_cadastral_{period}_meses' if period != 1 else 'mudanca_cadastral_mes_atual'
        current_month_data[col_name] = 0
        
        cpfs_with_changes = check_registration_changes(df, reference_date, period)
        
        current_month_data.loc[current_month_data['cpf_servidor'].isin(cpfs_with_changes), col_name] = 1
    
    return current_month_data        

def process_service_time(df: pd.DataFrame, date_col: str = 'mes_ing_spub', service_time_col: str = 'tempo_servico') -> pd.DataFrame:
    """
    Process service time data by handling missing values, converting date formats, and calculating service time.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame containing date information
        date_col (str): Name of the date column (default: 'mes_ing_spub')
        service_time_col (str): Name of the service time column to create (default: 'tempo_servico')
    
    Returns:
        pd.DataFrame: DataFrame with processed dates and calculated service times
    """
    # Handle missing values
    df[date_col] = df[date_col].replace(['s/info', 's info', 'S/INFO', 'S INFO', 'S/Info'], np.nan)
    
    # Only process non-null values
    mask = df[date_col].notna()
    if mask.any():
        # Convert valid dates
        df.loc[mask, date_col] = (
            df.loc[mask, date_col]
            .str.title()
            .str.replace('/', ' ')
            .apply(convert_month_names)
        )
        df.loc[mask, date_col] = pd.to_datetime(df.loc[mask, date_col], format='%b %Y')
        # Calculate service time
        data_atual = datetime.today()
        df.loc[mask, service_time_col] = df.loc[mask, date_col].apply(
            lambda x: (relativedelta(data_atual, x).years + (relativedelta(data_atual, x).months / 12))
        )
    
    return df

def process_esc_cargo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'esc_cargo' column by replacing 'S/cargo' with NaN values

    Parameters:
        df (pd.DataFrame): Input DataFrame containing 'esc_cargo' column

    Returns:
        pd.DataFrame: DataFrame with processed 'esc_cargo' column
    """
    df['esc_cargo'] = df['esc_cargo'].replace(['S/cargo'], np.nan)
    return df

def process_ingresso_spub(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processa a coluna 'ingresso_spub' agrupando valores em categorias específicas.
    
    Parâmetros:
    df (pd.DataFrame): DataFrame contendo a coluna 'ingresso_spub'
    
    Retorna:
    pd.DataFrame: DataFrame com a coluna processada
    
    Categorias:
    - 'nomeacao para cargo em comissao'
    - 'decisao judicial'
    - 'carater efetivo'
    - 'sem concurso'
    - 'outros'
    """
    # Dicionário de mapeamento para categorização
    CATEGORIAS = {
        'nomeacao para cargo em comissao': [
            'Nomea.Comis.Carg.A9,II,8112/90',
            'Nomeacao para Cargo',
            'Nomeacao Cargo Nat.Esp.8112/90',
            'Nomeacao P/CargoemComissao'
        ],
        'decisao judicial': [
            'RETORNO - EMPREGADO ANS - DEC',
            'SUCESSAO TRABALHISTA',
            'Anistiado Lei 8878/94',
            'Decisao Judicial'
        ],
        'carater efetivo': [
            'Nomea.Carat.Efet. A9,I,8112/90',
            'Admissao por Concurso Publico',
            'Admissao por Concurso/Empresa',
            'ContratodeTrabalho'
        ],
        'sem concurso': [
            'Admissao sem Concurso/Empresa',
            'Admissao sem Concurso Publico'
        ]
    }

    # Cópia para evitar modificação do dataframe original
    df = df.copy()
    
    # Etapa 1: Substituir valores específicos por NaN
    df['ingresso_spub'] = df['ingresso_spub'].replace(['S/Info'], np.nan)

    # Etapa 2: Substituições baseadas nas categorias
    for categoria, valores in CATEGORIAS.items():
        df['ingresso_spub'] = df['ingresso_spub'].replace(valores, categoria)

    # Etapa 3: Tratar valores remanescentes
    mask = ~df['ingresso_spub'].str.strip().str.lower().isin(
        ['nomeacao para cargo em comissao', 'decisao judicial', 'carater efetivo']
    )
    
    df['ingresso_spub'] = np.where(
        mask,
        'outros',
        df['ingresso_spub']
    )

    return df

def process_mt_entrada(df: pd.DataFrame, 
                      coluna: str = 'ingresso',
                      inplace: bool = False) -> pd.DataFrame:
    """
    Processa a coluna de ingresso categorizando valores específicos
    
    Parâmetros:
        df (pd.DataFrame): DataFrame contendo a coluna de ingresso
        coluna (str): Nome da coluna a ser processada (padrão: 'ingresso')
        inplace (bool): Se False (padrão), retorna uma cópia do DataFrame modificado
                        Se True, modifica o DataFrame original
    
    Retorna:
        pd.DataFrame: DataFrame com a coluna processada
    
    Categorizações:
        - 'nomeado cargo comissao'
        - 'anistiado/dec.judicial'
        - 'cedido'
        - 'outros' (categoria padrão para valores não mapeados)
    
    Levanta:
        ValueError: Se a coluna especificada não existir
    """
    # Mapeamento de categorias
    CATEGORIAS = {
        'nomeado cargo comissao': [
            'Nom.Comis.Carg. 8112/90,A.9,II',
            'Nom.Cargo Natureza Esp.8112/90'
        ],
        'anistiado/dec.judicial': [
            'Retorno - Empregado Ans - Dec',
            'Decisao Judicial'
        ],
        'cedido': [
            'Req. sem Onus 8112/90',
            'Ex. Descentralizado, Carreiras',
            'Requisitado Com Onus/Empresa',
            'Requisicao com Onus 8112/90'
        ]
    }

    # Validação inicial
    if coluna not in df.columns:
        raise ValueError(f"Coluna '{coluna}' não encontrada no DataFrame")
    
    # Trabalhar em cópia ou no original
    working_df = df if inplace else df.copy()

    try:
        # Pré-processamento: normalização
        working_df[coluna] = working_df[coluna].str.strip().str.lower()
        
        # Aplicar categorizações
        conditions = []
        choices = []
        
        for categoria, valores in CATEGORIAS.items():
            normalized_values = [v.strip().lower() for v in valores]
            conditions.append(working_df[coluna].isin(normalized_values))
            choices.append(categoria)
        
        # Categoria padrão para valores não mapeados
        conditions.append(True)
        choices.append('outros')
        
        working_df[coluna] = np.select(conditions, choices, default='outros')
        
        return working_df
        
    except Exception as e:
        error_msg = f"Falha no processamento da coluna {coluna}: {str(e)}"
        raise RuntimeError(error_msg) from e
        
def process_permanence_allowance(
    df: pd.DataFrame,
    column: str = 'abono_permanencia',
    missing_indicator: str = 'NÃO INFORMADO',
    inplace: bool = False
) -> pd.DataFrame:
    """
    Processa a coluna de abono de permanência convertendo para binário
    
    Parâmetros:
        df (pd.DataFrame): DataFrame contendo a coluna
        column (str): Nome da coluna a ser processada (padrão: 'abono_permanencia')
        missing_indicator (str): Valor que indica informação faltante (padrão: 'NÃO INFORMADO')
        inplace (bool): Se False (padrão), retorna uma cópia do DataFrame modificado
                        Se True, modifica o DataFrame original
    
    Retorna:
        pd.DataFrame: DataFrame com a coluna convertida para binário (0/1)
    
    Levanta:
        ValueError: Se a coluna especificada não existir
    
    Exemplo:
        >>> df = pd.DataFrame({'abono_permanencia': ['NÃO INFORMADO', 'ATIVO', None]})
        >>> process_permanence_allowance(df)
           abono_permanencia
        0                  0
        1                  1
        2                  1
    """
    # Validação inicial
    if column not in df.columns:
        raise ValueError(f"Coluna '{column}' não encontrada no DataFrame")
    
    # Trabalhar em cópia ou no original
    working_df = df if inplace else df.copy()

    try:
        # Converter para binário
        working_df[column] = np.where(
            working_df[column].isin([missing_indicator]),
            False,
            True
        )
        
        return working_df
        
    except Exception as e:
        error_msg = f"Falha no processamento da coluna {column}: {str(e)}"
        raise RuntimeError(error_msg) from e
  
def process_exclusion_column(
    df: pd.DataFrame,
    column: str = 'exclusao',
    exclusion_indicator: str = 'S/Exclusao',
    inplace: bool = False
) -> pd.DataFrame:
    """
    Processa a coluna de exclusão substituindo valores específicos por NaN
    
    Parâmetros:
        df (pd.DataFrame): DataFrame contendo a coluna
        column (str): Nome da coluna a ser processada (padrão: 'exclusao')
        exclusion_indicator (str): Valor que indica ausência de informação (padrão: 'S/Exclusao')
        inplace (bool): Se False (padrão), retorna uma cópia do DataFrame modificado
                        Se True, modifica o DataFrame original
    
    Retorna:
        pd.DataFrame: DataFrame com a coluna processada
    
    Levanta:
        ValueError: Se a coluna especificada não existir
        TypeError: Se ocorrer erro na conversão dos valores
    
    Exemplo:
        >>> df = pd.DataFrame({'exclusao': ['S/Exclusao', 'Aposentadoria', 'Demissão']})
        >>> process_exclusion_column(df)
           exclusao
        0       NaN
        1  Aposentadoria
        2     Demissão
    """
    # Validação inicial
    if column not in df.columns:
        raise ValueError(f"Coluna '{column}' não encontrada no DataFrame")
    
    # Trabalhar em cópia ou no original
    working_df = df if inplace else df.copy()

    try:
        # Normalização para tratamento case-insensitive e espaços
        working_df[column] = (
            working_df[column]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace(exclusion_indicator.strip().lower(), pd.NA)
        )
        
        # Restaura valores originais exceto pelo indicador
        mask = working_df[column].isna()
        working_df.loc[~mask, column] = df.loc[~mask, column]
        
        return working_df
        
    except Exception as e:
        error_msg = f"Falha no processamento da coluna {column}: {str(e)}"
        raise TypeError(error_msg) from e

def process_reversion(
    df: pd.DataFrame,
    column: str = 'reversao',
    missing_indicators: list = ['S/ reversao', 's/reversao'],
    inplace: bool = False,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Processa a coluna de reversão convertendo para booleano, tratando múltiplos indicadores de missing
    
    Parâmetros:
        df (pd.DataFrame): DataFrame contendo a coluna
        column (str): Nome da coluna a ser processada (padrão: 'reversao')
        missing_indicators (list): Lista de valores que indicam ausência de informação
        inplace (bool): Se False (padrão), retorna uma cópia do DataFrame
        verbose (bool): Se True, exibe informações de debug
    
    Retorna:
        pd.DataFrame: DataFrame com a coluna convertida para booleano (True/False)
    
    Levanta:
        ValueError: Se a coluna não existir ou estiver vazia
        TypeError: Se a conversão falhar
    
    Exemplo:
        >>> df = pd.DataFrame({'reversao': ['S/ reversao', 'Ativa', 's/reversao', 'Inativa']})
        >>> process_reversion(df)
           reversao
        0      False
        1       True
        2      False
        3       True
    """
    # Validação inicial
    if column not in df.columns:
        raise ValueError(f"Coluna '{column}' não encontrada no DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame está vazio")
    
    # Trabalhar em cópia ou no original
    working_df = df if inplace else df.copy()
    
    try:
        # Pré-processamento: normalização
        working_df[column] = (
            working_df[column]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        
        # Criar lista de valores missing normalizados
        missing_list = [str(x).strip().lower() for x in missing_indicators]
        
        # Criar máscara booleana (True = valor válido, False = missing)
        mask = ~working_df[column].isin(missing_list)
        
        # Aplicar conversão diretamente para booleano
        working_df[column] = mask
        
        # Validação pós-processamento
        if verbose:
            print(f"Valores únicos após conversão: {working_df[column].unique()}")
            print(f"Distribuição:\n{working_df[column].value_counts(dropna=False)}")
        
        return working_df
        
    except Exception as e:
        error_msg = (f"Falha no processamento da coluna {column}. "
                     f"Tipo original: {df[column].dtype}. Erro: {str(e)}")
        raise TypeError(error_msg) from e

def process_aposentadoria(df: pd.DataFrame, col: str = 'aposentadoria') -> pd.DataFrame:
    """
    Processa a coluna de aposentadoria com:
    1. Substituição de valores missing
    2. Unificação de categorias jurídicas
    3. Categorização de valores residuais
    
    Parâmetros:
        df (pd.DataFrame): DataFrame contendo a coluna
        col (str): Nome da coluna a ser processada (padrão: 'aposentadoria')
    
    Retorna:
        pd.DataFrame: DataFrame com a coluna processada
    
    Lógica:
        - 'S/Aposentadoria' → NaN
        - Termos jurídicos → 'DECISAO JUDICIAL'
        - Demais valores → 'outros'
    """
    # Validação
    if col not in df.columns:
        raise ValueError(f"Coluna '{col}' não encontrada")
    
    df = df.copy()
    
    # Lista de padrões jurídicos
    padroes_juridicos = {
        'DECISAO JUDICIAL': [
            'DECISAO JUDICIAL',
            'JUDICIAL PROVENTOS INTEGRAIS',
            'JUDICIAL PROVENTO INTEGRAL'
        ]
    }
    
    try:
        # Passo 1: Substituir valores missing
        df[col] = df[col].replace('S/Aposentadoria', np.nan)
        
        # Passo 2: Unificar termos jurídicos
        df[col] = np.where(
            df[col].isin(padroes_juridicos['DECISAO JUDICIAL']),
            'DECISAO JUDICIAL',
            df[col]
        )
        
        # Passo 3: Categorizar valores restantes
        df[col] = np.where(
            df[col].isin(['DECISAO JUDICIAL'])  | df[col].isna(),
            df[col],
            'APOSENTADO'
        )
        
        return df
        
    except Exception as e:
        raise TypeError(f"Falha no processamento: {str(e)}") from e

def processar_nivel_funcao(df: pd.DataFrame, 
                          coluna: str = 'nivel_funcao',
                          inplace: bool = False) -> pd.DataFrame:
    """
    Processa a coluna de nível funcional com:
    1. Limpeza de espaços
    2. Substituição de valores específicos
    3. Categorização em faixas
    
    Parâmetros:
        df (pd.DataFrame): DataFrame original
        coluna (str): Nome da coluna a ser processada
        inplace (bool): Se False retorna novo DataFrame, se True modifica o original
        
    Retorna:
        pd.DataFrame: DataFrame com a coluna processada
    """
    
    # Dicionário de mapeamento para categorias
    MAPEAMENTO = {
        'FAIXA 1': [
            'ETG-0002', 'CTE-0027(EPL', 'ETG-0009', 'ETG-0011', 'ETG-0001',
            'GF-0004 (GEIPOT)', 'GF-0009 (GEIPOT', 'GF-0014 (GEIPOT)',
            'GF-0005 (FRANAVE)', 'GF-0002 (GEIPOT)', 'FPE-1025', 'FPE-1015',
            'GF-0003 (GEIPOT)', 'GF-0002 (ENASA)', 'DAI-1111',
            'PRESIDENTE (FRANAVE)', 'GF-0001 (ENASA)', 'GF-0001 (GEIPOT)',
            'GF-0006 (FRANAVE)', 'DAI-1112', 'GF-0037(FRAN',
            'GF-0009 (FRANAVE)', 'GF-0007 (FRANAVE)', 'GF-0008 (FRANAVE)',
            'GF-0004 (FRANAVE)', 'CCX-0105', 'GF-0007 (GEIPOT)',
            'GF-0003 (FRANAVE)', 'ASSISTENTE JURIDICO (GEIPOT)', 'FPE-1012',
            'FPE-1042', 'FPE-1022', 'DAS-2', 'DIRETOR (FRANAVE)',
            'GF-0036(FRAN', 'GF-0006 (ENASA)', 'DAS-6', 'FEX-0407',
            'GF-0035(FRAN', 'GF-0033(FRAN', 'GF-0034(FRAN', 'CTE-0025(EPL',
            'CCX-0109', 'GF-0001 (FRANAVE)', 'GF-0002 (FRANAVE)',
            'GF-0031(FRAN', 'GF-0032(FRAN'
        ],
        'FAIXA 2': [
            'GF-0029(FRAN', 'GF-0012 (GEIPOT)', 'DAS-3', 'FPE-1013',
            'GF-0015 (GEIPOT)', 'CCX-0206', 'CCX-0306', 'GF-0010 (GEIPOT)',
            'GF-0013 (GEIPOT)', 'GF-0011 (GEIPOT)', 'DAS-1', 'FEX-0210',
            'FEX-0310', 'CCX-0107', 'CTE-0018(EPL', 'GF-0030(FRAN',
            'VICE-DIRETOR (FRANAVE)', 'GF-0016 (GEIPOT)', 'CTE-0024(EPL',
            'CCX-0207', 'FGR-0003', 'GF-0006 (GEIPOT)', 'FGR-0001',
            'CCX-0308', 'FCT-0001', 'DAI-1122', 'GF-0008 (GEIPOT)',
            'FPE-1021', 'FPE-1033', 'CTE-0023(EPL', 'FCT-0004', 'CCX-0210',
            'FGD-0026(EPL'
        ],
        'FAIXA 3': [
            'GF-0005 (ENASA)', 'CCX-0110', 'FPE-1024', 'FPE-1034',
            'CCX-0310', 'CTE-0010(EPL', 'FCT-0009', 'FGR-0002', 'FCT-0015',
            'FCT-0010', 'FEX-0213', 'FPE-1014', 'Subchefe Executivo Casa Civil',
            'FPE-1041', 'FEX-0110', 'GF-0005 (GEIPOT)', 'FEX-0207',
            'FEX-0410', 'FEX-0214', 'FEX-0114', 'FEX-0313', 'FCT-0011',
            'FPE-1011', 'FEX-0405', 'CTE-0020(EPL', 'CTE-0022(EPL',
            'Procurador Geral da Uniao', 'FPE-1035', 'FEX-0113', 'FEX-0105',
            'GF-0028(FRAN', 'FPE-1043', 'CTE-0021(EPL', 'CTE-0001(EPL',
            'FPE-1023', 'FEX-0115', 'FEX-0315', 'CCX-0215', 'CCX-0113',
            'FEX-0107', 'CCX-0313', 'FCT-0007'
        ],
        'FAIXA 4': [
            'CCX-0114', 'DAS-4', 'Secretario Executivo', 'GF-0017 (GEIPOT)',
            'CCX-0106', 'CCX-0117', 'CCX-0115', 'CTE-0019(EPL', 'CCX-0118',
            'CTE-0016(EPL', 'CTE-0017(EPL', 'CCX-0213', 'CCX-0214',
            'FEX-0307', 'DAS-5', 'FGD-0027(EPL', 'CTE-0015(EPL',
            'CCX-0315', 'CTE-0014(EPL', 'CTE-0012(EPL'
        ],
        'FAIXA 5': [
            'CTE-0011(EPL', 'CCX-0316', 'CTE-0013(EPL', 'CTE-0008(EPL',
            'CTE-0005(EPL', 'CTE-0004(EPL', 'CTE-0003(EPL', 'CTE-0002(EPL',
            'CTE-0026(EPL'
        ]
    }

    working_df = df if inplace else df.copy()

    try:
        # Pré-processamento
        working_df[coluna] = (
            working_df[coluna]
            .astype(str)
            .str.strip()
            .replace('S/nivel funcao', np.nan)
        )

        # Aplicar mapeamento
        for faixa, valores in MAPEAMENTO.items():
            working_df[coluna] = working_df[coluna].replace(valores, faixa)

        return working_df if not inplace else None

    except KeyError as e:
        raise ValueError(f"Coluna {coluna} não encontrada no DataFrame") from e
    except Exception as e:
        raise RuntimeError(f"Erro no processamento: {str(e)}") from e     

def processar_situacao_funcional(
    df: pd.DataFrame,
    coluna: str = 'situacao_funcional',
    inplace: bool = False,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Processa a coluna de situação funcional removendo informações após o hífen
    
    Parâmetros:
        df (pd.DataFrame): DataFrame contendo a coluna
        coluna (str): Nome da coluna a ser processada (padrão: 'situacao_funcional')
        inplace (bool): Se False (padrão), retorna uma cópia modificada
        verbose (bool): Se True, exibe informações do processamento
    
    Retorna:
        pd.DataFrame: DataFrame com a coluna processada
    
    Exemplo:
        >>> df = pd.DataFrame({'situacao_funcional': ['ATIVO-123', 'INATIVO-456', 'AFASTADO-789']})
        >>> processar_situacao_funcional(df)
          situacao_funcional
        0              ATIVO
        1            INATIVO
        2           AFASTADO
    """
    # Validação inicial
    if coluna not in df.columns:
        raise ValueError(f"Coluna '{coluna}' não encontrada no DataFrame")
    
    # Criar cópia se necessário
    working_df = df if inplace else df.copy()

    try:
        # Registrar estatísticas antes (se verbose)
        if verbose:
            original_values = working_df[coluna].value_counts(dropna=False)
            print(f"\nValores originais ({coluna}):")
            print(original_values)

        # Converter para string e processar
        working_df[coluna] = (
            working_df[coluna]
            .astype(str)
            .str.replace(r"-.*", "", regex=True)  # Remove tudo após o primeiro hífen
            .replace('nan', np.nan)  # Reconverte strings 'nan' para valores nulos
        )

        return working_df

    except Exception as e:
        error_msg = f"Falha ao processar {coluna}: {str(e)}"
        raise RuntimeError(error_msg) from e
        
def categorizar_rendimentos(
    df: pd.DataFrame,
    reference_date: str,
    col_cpf: str = 'cpf_servidor',
    col_rendim: str = 'rendim',
    col_mes: str = 'mes',
    inplace: bool = False
) -> Optional[pd.DataFrame]:
    """
    Categoriza valores de rendimento em faixas específicas para um mês de referência.
    
    Parâmetros:
        df (pd.DataFrame): DataFrame original com os dados
        reference_date (str): Data de referência no formato 'YYYY-MM'
        col_cpf (str): Nome da coluna do CPF (padrão: 'cpf_servidor')
        col_rendim (str): Nome da coluna de rendimento (padrão: 'rendim')
        col_mes (str): Nome da coluna de mês (padrão: 'mes')
        inplace (bool): Se False retorna novo DataFrame (padrão), se True modifica o original
    
    Retorna:
        pd.DataFrame: DataFrame com as categorias de rendimento
        
    Exemplo:
        >>> df_categorizado = categorizar_rendimentos(df, '2023-01')
    """
    
    # Validação inicial
    required_cols = {col_cpf, col_rendim, col_mes}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Colunas faltantes: {missing}")
    
    # Trabalhar em cópia se necessário
    working_df = df if inplace else df.copy()
    
    try:
        # 1. Filtrar e selecionar colunas
        df_filtrado = working_df[[col_cpf, col_rendim, col_mes]].copy()
        
        # 2. Converter datas e filtrar mês de referência
        df_filtrado[col_mes] = pd.to_datetime(df_filtrado[col_mes])
        ref_date = pd.to_datetime(reference_date)
        df_filtrado = df_filtrado[df_filtrado[col_mes] == ref_date]
        
        # 3. Converter rendimentos para numérico
        df_filtrado[col_rendim] = pd.to_numeric(df_filtrado[col_rendim], errors='coerce')
        
        # 4. Aplicar categorização
        conditions = [
            (df_filtrado[col_rendim] < 5000),
            (df_filtrado[col_rendim] >= 5000) & (df_filtrado[col_rendim] < 10000),
            (df_filtrado[col_rendim] >= 10000) & (df_filtrado[col_rendim] < 15000),
            (df_filtrado[col_rendim] >= 15000) & (df_filtrado[col_rendim] < 20000),
            (df_filtrado[col_rendim] >= 20000) & (df_filtrado[col_rendim] < 25000),
            (df_filtrado[col_rendim] >= 25000) & (df_filtrado[col_rendim] < 30000),
            (df_filtrado[col_rendim] >= 30000)
        ]
        
        choices = [1, 2, 3, 4, 5, 6, 7]
        
        df_filtrado[col_rendim] = np.select(conditions, choices, default=np.nan)
        
        return df_filtrado if not inplace else None
        
    except Exception as e:
        raise ValueError(f"Erro no processamento: {str(e)}") from e
        
def combinar_niveis_escolaridade(
    df: pd.DataFrame,
    col_nivel: str = 'nivel_funcao',
    col_esc_cargo: str = 'esc_cargo',
    nova_coluna: str = 'nivel_escolaridade',
    remover_originais: bool = True,
    inplace: bool = False
) -> Optional[pd.DataFrame]:
    """
    Combina duas colunas de informação educacional em uma nova coluna e remove as originais.

    Parâmetros:
        df (pd.DataFrame): DataFrame de entrada
        col_nivel (str): Nome da coluna com nível funcional (padrão: 'nivel_funcao')
        col_esc_cargo (str): Nome da coluna com escolaridade do cargo (padrão: 'esc_cargo')
        nova_coluna (str): Nome da nova coluna combinada (padrão: 'nivel_escolaridade')
        remover_originais (bool): Se True remove as colunas originais (padrão: True)
        inplace (bool): Se False retorna novo DataFrame (padrão), se True modifica o original

    Retorna:
        pd.DataFrame: DataFrame com a nova coluna combinada

    Exemplo:
        >>> df = pd.DataFrame({
        ...     'nivel_funcao': ['A', None, 'B'],
        ...     'esc_cargo': [None, 'X', 'Y']
        ... })
        >>> combinar_niveis_escolaridade(df)
          nivel_escolaridade
        0                  A
        1                  X
        2                 BY
    """
    
    # Validação de colunas
    required_cols = {col_nivel, col_esc_cargo}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise KeyError(f"Colunas faltantes: {missing_cols}")

    working_df = df if inplace else df.copy()

    try:
        # Converter para string e substituir NaN
        working_df[nova_coluna] = (
            working_df[col_nivel].fillna('').astype(str) + 
            working_df[col_esc_cargo].fillna('').astype(str)
        )

        # Remover colunas originais se necessário
        if remover_originais:
            working_df.drop([col_nivel, col_esc_cargo], axis=1, inplace=True)

        return working_df if not inplace else None

    except Exception as e:
            raise RuntimeError(f"Erro na combinação de colunas: {str(e)}") from e