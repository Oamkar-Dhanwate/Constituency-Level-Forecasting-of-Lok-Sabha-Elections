import pandas as pd

def create_alliance_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Adds an 'Alliance' column based on party affiliation."""
    print("Creating Alliance feature.")
    
    alliance_mapping = {
        # NDA Parties
        'bjp': 'NDA',
        'shiv sena': 'NDA',
        'jd(u)': 'NDA',
        'aiadmk': 'NDA',
        # UPA Parties
        'inc': 'UPA',
        'ncp': 'UPA',
        'rjd': 'UPA',
        'dmk': 'UPA',
        'jmm': 'UPA',
        # Third Front / Others
        'sp': 'Third Front',
        'bsp': 'Third Front',
        'aitc': 'Third Front',
        'cpi(m)': 'Third Front',
        'cpi': 'Third Front',
        'ind': 'Independent'
    }
    
    # Map the party to its alliance, fill others as 'Other'
    df['Alliance'] = df['Party'].map(alliance_mapping).fillna('Other')
    return df