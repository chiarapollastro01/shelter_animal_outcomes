"""
Unit tests for the preprocessing module.

"""

import pandas as pd
import numpy as np
from src.preprocessing import extract_age_in_days, DataCleaner

def test_extract_age_in_days():
    """
    Test the vectorized extraction and conversion of age strings into numeric days.
    """
    # 1. To test the function, we create a mock Series of age strings that includes various formats and edge cases.
    age_series = pd.Series([
        "1 year", 
        "2 months", 
        "3 weeks", 
        "5 days", 
        np.nan, 
        "Unknown"
    ])
    
    # 2. Call the function to convert the age strings into numeric days.
    result = extract_age_in_days(age_series)
    
    # 3. Based on the input, we expect the following conversions:
    assert result.iloc[0] == 365.0   # 1 * 365
    assert result.iloc[1] == 60.0    # 2 * 30
    assert result.iloc[2] == 21.0    # 3 * 7
    assert result.iloc[3] == 5.0     # 5 * 1
    assert np.isnan(result.iloc[4])  # Missing remains missing
    assert np.isnan(result.iloc[5])  # Unparseable text becomes NaN

def test_extract_age_edge_cases():
    """
    Test that the function can handle edge cases, such as empty strings, 
    non-numeric values, and mixed formats.

    """
    age_series = pd.Series([
        "1 YEAR",       # Uppercase
        "  2 months  ", # Spaces around the text
        "1 day",        # Singular
        "10 DAYS",      # Plural uppercase
    ])


def test_datacleaner_pipeline():
    """
    Test the entire DataCleaner pipeline: dropping columns, 
    mode imputation for categorical data, and median imputation for numeric data.
    """
    # 1. Create a mock DataFrame with various columns, some of which will be dropped, and some with missing values.
    df_mock = pd.DataFrame({
        "AnimalID": ["A1", "A2", "A3", "A4", "A5"],
        "OutcomeSubtype": ["Partner", "Foster", np.nan, "Partner", "Foster"],
        "Name": ["Bella", "Max", np.nan, "Luna", "Charlie"],
        # For SexuponOutcome, "Neutered Male" appears 3 times, making it the mode.
        "SexuponOutcome": ["Neutered Male", "Neutered Male", "Intact Female", np.nan, "Neutered Male"],
        # For AgeuponOutcome, values are 365, 730, 1095, NaN, 730. Median of valid is 730.0.
        "AgeuponOutcome": ["1 year", "2 years", "3 years", np.nan, "2 years"]
    })

    cleaner = DataCleaner()

    # 2. Run the data through the cleaning pipeline
    df_clean = cleaner.clean_data(df_mock)

    # 3. Verify the results
    
    # A) Check column removals
    assert "AnimalID" not in df_clean.columns, "AnimalID should be dropped."
    assert "OutcomeSubtype" not in df_clean.columns, "OutcomeSubtype should be dropped."
    assert "AgeuponOutcome" not in df_clean.columns, "AgeuponOutcome should be dropped after conversion."
    assert "Name" in df_clean.columns, "Name should NOT be dropped."

    # B) Check SexuponOutcome imputation (Mode)
    assert df_clean["SexuponOutcome"].isnull().sum() == 0, "There should be no missing values in SexuponOutcome."
    # Index 3 was NaN, it should now be the mode ("Neutered Male")
    assert df_clean.loc[3, "SexuponOutcome"] == "Neutered Male", "NaN in SexuponOutcome was not filled with the mode."

    # C) Check Age conversion and imputation (Median)
    assert "age_in_days" in df_clean.columns, "age_in_days column is missing."
    assert df_clean["age_in_days"].isnull().sum() == 0, "There should be no missing values in age_in_days."
    # Index 0 should be perfectly parsed to 365.0
    assert df_clean.loc[0, "age_in_days"] == 365.0, "Age parsing failed for '1 year'."
    # Index 3 was NaN, it should now be the median (730.0)
    assert df_clean.loc[3, "age_in_days"] == 730.0, "NaN in age_in_days was not filled with the median."

def test_datacleaner_extreme_cases():
    """
    Testa il comportamento del DataCleaner con estremi:
    un dataset senza alcun valore mancante e un dataset con sole colonne vuote.
    """
    cleaner = DataCleaner()
    
    # CASO A: Dati perfetti (0 NaN)
    df_perfect = pd.DataFrame({
        "SexuponOutcome": ["Neutered Male", "Intact Female"],
        "AgeuponOutcome": ["1 year", "2 years"]
    })
    
    df_clean_perfect = cleaner.clean_data(df_perfect)
    assert df_clean_perfect["SexuponOutcome"].isnull().sum() == 0
    assert df_clean_perfect["age_in_days"].isnull().sum() == 0
    
    # CASO B: Il disastro totale (100% NaN)
    df_disaster = pd.DataFrame({
        "SexuponOutcome": [np.nan, np.nan],
        "AgeuponOutcome": [np.nan, np.nan]
    })
    
    df_clean_disaster = cleaner.clean_data(df_disaster)
    
    # Il sesso dovrebbe usare il fallback "Unknown" perché non c'è una moda
    assert df_clean_disaster.loc[0, "SexuponOutcome"] == "Unknown"
    
    # L'età dovrebbe rimanere NaN perché la mediana di soli NaN è NaN
    assert np.isnan(df_clean_disaster.loc[0, "age_in_days"])