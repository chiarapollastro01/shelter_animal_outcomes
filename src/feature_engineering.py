import numpy as np
import pandas as pd
from src.preprocessing import TemporalFeaturesExtractor


class AdvancedTemporalFeaturesExtractor(TemporalFeaturesExtractor):
    """Advanced feature extractor for high-level operational features."""

    def __init__(self, datetime_col: str = "DateTime", add_kitten_season: bool = False):
        """Initialize with option to conditionally add the Kitten Season flag."""
        super().__init__(datetime_col)
        self.add_kitten_season = add_kitten_season

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Obtain Hour and Weekday from TemporalFeaturesExtractor
        """
        df_out = super().transform(df)

        if "Weekday" not in df_out.columns or "Hour" not in df_out.columns:
            return df_out

        df_out["IsWeekend"] = (df_out["Weekday"] >= 5).astype(int)

        bins = [-1, 5, 11, 16, 23] 
        labels = ["Night", "Morning", "Afternoon", "Evening"]
        df_out["TimeOfDay"] = pd.cut(
            df_out["Hour"], bins=bins, labels=labels, right=True
        ).astype(str)

        
        if self.add_kitten_season:
            doy = df_out[self.datetime_col].dt.dayofyear
            df_out["IsKittenSeason"] = ((doy >= 91) & (doy <= 304)).astype(int)

        return df_out