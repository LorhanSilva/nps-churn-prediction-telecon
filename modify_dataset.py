import pandas as pd
from pathlib import Path
# Read the CSV file

def trasnforma_columns_str_to_int(root: str|Path) -> None:
    """ #### Transforma colunas categóricas de string para int.
    Args:
        root (str): Local com arquivos csv
    """
    for file in Path(root).glob('recl_tim_*.csv'):
        df = pd.read_csv(file)
        # Definir mapping para as colunas categóricas.

        keys = ["tipo_assinante", "tipo_terminal", "dsc_origem", "tripleta"]

        for key in keys:
            unique_values = df[key].unique()
            mapping = {value: idx for idx, value in enumerate(unique_values)}
            df[key] = df[key].map(mapping)
        

        df.to_csv(f'logs_mdf\\modify_{file.name}', index=False)
        break


if __name__ == "__main__":
    trasnforma_columns_str_to_int('logs')