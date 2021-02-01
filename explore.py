import pandas as pd


class ConvertTypes:
    """
    Class pour convertir les types, elle a une methode fit et une methode transforme
    """

    def __init__(self, dtypes=None):
        self.dtypes = dtypes

    def transform(self, X, y=None):
        return self.convert_types(X, self.dtypes)

    def fit(self, X, y=None):
        pass

    def convert_types(df: pd.DataFrame, dtypes: dict) -> pd.DataFrame:
        """
        effectue une conversion des colonnes

        - df : pd.DataFrame
        - dtypes: dict
        exemple :

          dtypes = {
            "int32" : ["colName1", "colName3"],
            "category" : ['colName2'],
            "float" : ['colName4'],
            "str" : ['colName5'],
            "datetime64[ns]" : [colName6]
          }
          convert_type(df, dtypes)
        """
        result_df = df.copy()
        for dtype in dtypes:
            cols = dtypes[dtype]
            result_df[cols] = df[cols].astype(dtype, erros="ignore")

        return result_df


def missings_values_infos(df: pd.DataFrame):
    """"""
    data = df.copy()
    left_size_formating = 20
    right_size_formating = 20
    line_formating = "\n" + "-" * (left_size_formating + right_size_formating) + "\n"

    info = ""

    n_obs, n_cols = data.shape

    info += "Nb variables".ljust(left_size_formating) + f"{n_cols}".rjust(
        right_size_formating
    )
    info += line_formating
    info += "Nb observations".ljust(left_size_formating) + f"{n_obs}".rjust(
        right_size_formating
    )
    info += line_formating

    nb_ligne_nan = data.isna().any(axis=1).sum()
    nb_col_nan = data.isna().any(axis=0).sum()
    nb_nan_per_col = data.isna().sum()

    # nb_nan_val = {col: nb_nan_per_col[col] for col in nb_nan_per_col.index if nb_nan_per_col[col]!=0}

    info += f"VALEURS MANQUANTES".center(left_size_formating + right_size_formating)
    info += line_formating
    info += f"Nb colonne avec au moin une valeur manquante".ljust(
        left_size_formating
    ) + f"{ nb_col_nan }".rjust(right_size_formating)
    info += line_formating
    info += f"Nb ligne avec au moin une valeur manquante".ljust(
        left_size_formating
    ) + f"{ nb_ligne_nan }".rjust(right_size_formating)
    info += line_formating
    info += "Détails"
    info += line_formating
    for col in nb_nan_per_col.index:
        nb_null = nb_nan_per_col[col]
        col_type = data[col].dtype
        info += f"{col}({col_type})".ljust(
            left_size_formating
        ) + f"{nb_null}/{n_obs} = {round(100*nb_null/n_obs, 4)}%".rjust(
            right_size_formating
        )
        info += line_formating

    return print(info)
