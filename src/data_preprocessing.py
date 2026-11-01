from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def get_preprocessor():
    """
    Crée le ColumnTransformer pour le dataset Credit Card Fraud.
    - Normalise 'Time' et 'Amount'.
    - Laisse passer 'V1' à 'V28' (déjà normalisées par l'ACP).
    """
    # Les 2 seules colonnes brutes à normaliser
    cols_to_scale = ['Time', 'Amount']
    
    # On liste explicitement les colonnes V1 à V28 pour le passthrough
    v_cols = [f'V{i}' for i in range(1, 29)]

    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), cols_to_scale),
            ('passthrough', 'passthrough', v_cols)
        ]
    )
    
    return preprocessor