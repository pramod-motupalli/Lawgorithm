import pandas as pd
from collections import Counter
from typing import List, Dict, Any
from . import config
from . import utils

class AnalyticsEngine:
    def __init__(self):
        self.data_frames = {}
        self._load_data()

    def _load_data(self):
        """
        Loads data into Pandas DataFrames for analysis.
        """
        datasets = {
            "Civil": config.CIVIL_DATA_PATH,
            "Criminal": config.CRIMINAL_DATA_PATH,
            "Traffic": config.TRAFFIC_DATA_PATH
        }
        
        for category, path in datasets.items():
            data = utils.load_json_data(path)
            if data:
                self.data_frames[category] = pd.DataFrame(data)
            else:
                self.data_frames[category] = pd.DataFrame()

    def get_win_rate(self, category: str, crime_type: str = None) -> Dict[str, float]:
        """
        Calculates the percentage of Convictions vs Acquittals (or similar outcomes).
        """
        if category not in self.data_frames or self.data_frames[category].empty:
            return {"error": "No data"}

        df = self.data_frames[category]
        
        # Filter by crime type if provided (simple string match)
        if crime_type:
            df = df[df['crime_committed'].str.contains(crime_type, case=False, na=False) | 
                    df['judgment_summary'].str.contains(crime_type, case=False, na=False)]

        if df.empty:
            return {"message": "No matching cases found for analysis"}

        # Analyze 'verdict' column
        # We need to normalize verdict strings
        verdicts = df['verdict'].dropna().apply(lambda x: x.split(', '))
        all_outcomes = [item for sublist in verdicts for item in sublist]
        
        total = len(all_outcomes)
        if total == 0:
            return {"message": "No verdict data available"}

        counts = Counter(all_outcomes)
        stats = {k: round((v / total) * 100, 1) for k, v in counts.items()}
        
        return stats

    def get_top_statutes(self, category: str, limit: int = 5) -> Dict[str, int]:
        """
        Returns the most frequently cited IPC sections or Acts.
        """
        if category not in self.data_frames or self.data_frames[category].empty:
            return {}

        df = self.data_frames[category]
        
        # 'ipc_sections' is a list of strings
        all_sections = []
        for sections in df['ipc_sections']:
            if isinstance(sections, list):
                all_sections.extend(sections)
        
        return dict(Counter(all_sections).most_common(limit))

    def get_year_trend(self, category: str) -> Dict[int, int]:
        """
        Returns the number of cases per year.
        """
        if category not in self.data_frames or self.data_frames[category].empty:
            return {}

        df = self.data_frames[category]
        if 'year' not in df.columns:
            return {}
            
        return df['year'].value_counts().sort_index().to_dict()
