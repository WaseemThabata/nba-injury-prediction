"""
Data loading and merging for NBA injury prediction
Converts R data loading logic to Python
"""

import pandas as pd
from pathlib import Path
from typing import Tuple
import warnings

warnings.filterwarnings('ignore')


class NBADataLoader:
    """Load and merge NBA datasets (injuries, player stats, salaries)"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        
    def load_injuries(self) -> pd.DataFrame:
        """Load NBA injury data 2010-2020"""
        file_path = self.data_dir / "injuries_2010_2020.xlsx"
        if not file_path.exists():
            print(f"Warning: {file_path} not found. Using empty DataFrame.")
            return pd.DataFrame()
        return pd.read_excel(file_path)
    
    def load_player_stats(self) -> pd.DataFrame:
        """Load all seasons player statistics"""
        file_path = self.data_dir / "all_seasons.xlsx"
        if not file_path.exists():
            print(f"Warning: {file_path} not found. Using empty DataFrame.")
            return pd.DataFrame()
        
        df = pd.read_excel(file_path)
        # Parse season column (e.g., "2019-20" -> 2019)
        if 'season' in df.columns:
            df['season_year'] = df['season'].str[:4].astype(int)
        return df
    
    def load_salaries(self) -> pd.DataFrame:
        """Load NBA salary data"""
        file_path = self.data_dir / "NBA_Salaries.xlsx"
        if not file_path.exists():
            print(f"Warning: {file_path} not found. Using empty DataFrame.")
            return pd.DataFrame()
        return pd.read_excel(file_path)
    
    def merge_datasets(self) -> pd.DataFrame:
        """
        Merge injuries, stats, and salaries into unified dataset
        Replicates R merge logic: merge(nba_injury_data, all_seasons, by.x = c("name", "season"), ...)
        """
        injuries = self.load_injuries()
        stats = self.load_player_stats()
        salaries = self.load_salaries()
        
        if injuries.empty or stats.empty:
            print("Warning: Cannot merge datasets - missing required data files")
            return pd.DataFrame()
        
        # Format dates and extract season
        if 'Date' in injuries.columns:
            injuries['date'] = pd.to_datetime(injuries['Date'])
            injuries['season'] = injuries['date'].apply(self._extract_season)
        
        # Merge datasets
        merged = injuries.merge(
            stats, 
            left_on=['name', 'season'], 
            right_on=['name', 'season_year'],
            how='left'
        )
        
        if not salaries.empty:
            merged = merged.merge(
                salaries,
                on=['name', 'season'],
                how='left'
            )
        
        print(f"Merged dataset: {len(merged)} records")
        return merged
    
    @staticmethod
    def _extract_season(date) -> int:
        """
        Convert date to NBA season year (e.g., Jan 2020 -> 2019 season)
        NBA season spans Oct-Jun, so months < 7 belong to previous year's season
        """
        year = date.year
        return year - 1 if date.month < 7 else year
