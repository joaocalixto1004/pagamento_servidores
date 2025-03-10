import pandas as pd


def ler_planilha_servidores(caminho_arquivo, planilha="Servidores Mes"):
    """

    Lê a planilha de servidores do arquivo Excel fornecido.
    

    Parâmetros:

    caminho_arquivo (str): Caminho do arquivo Excel

    planilha (str): Nome da planilha a ser lida (padrão: "Servidores Mes")
    

    Retorna:

    DataFrame: Dados dos servidores formatados
    """
    

    try:

        # Configurações de leitura

        dtype = {

            'CPF SERVIDOR': str,

            'ESC CARGO': 'category'

        }
        

        # Ler o arquivo

        df = pd.read_excel(

            caminho_arquivo,

            sheet_name=planilha,

            dtype=dtype,

            parse_dates=['MÊS', 'MÊS ING SPUB'],

            na_values=['NÃO INFORMADO', 'S/Exclusao', 'S/ reversao', 'S/nivel funcao']

        )
        

        # Converter números com vírgula decimal

        df['RENDIM'] = df['RENDIM'].str.replace('.', '').str.replace(',', '.').astype(float)
        

        # Ordenar por CPF e Mês

        df = df.sort_values(['CPF SERVIDOR', 'MÊS'])
        
        return df
    

    except FileNotFoundError:

        raise ValueError("Arquivo não encontrado. Verifique o caminho fornecido.")

    except Exception as e:

        raise RuntimeError(f"Erro ao ler o arquivo: {str(e)}")


# Exemplo de uso:

# df = ler_planilha_servidores("c:\\Users\\joaoc\\Documents\\MT\\pagamento_servidores\\data\\raw\\servidores_mes.xlsx")
# jls_extract_var = print(df.head())
# jls_extract_var