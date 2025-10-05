import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Optional, Dict, List


class NewaveLoader:
    """
    A class to load and process Newave cmarg*.out files.
    
    The Newave files contain marginal costs (CMO - Custo Marginal de Operação)
    for different months, scenarios, and load patterns.
    
    File structure:
    - Header with metadata (PLD date, version, submercado)
    - Data organized by year and month
    - 3 load patterns (PAT 1, 2, 3) per month representing light, medium, heavy loads
    - Columns 1-12 representing different scenarios/simulations
    - Last column (MEDIA) representing the average across scenarios
    """
    
    def __init__(self, newave_path: Optional[str] = None):
        """
        Initialize the NewaveLoader.
        
        Args:
            newave_path (str, optional): Path to the cmarg file. If None, uses default path.
        """
        if newave_path is None:
            self.newave_path = "data/newave/CCEE_Resultados_agosto2025/saidas_log/cmarg001.out"
        else:
            self.newave_path = newave_path
    
    def _extract_metadata(self, lines: List[str]) -> Dict:
        """
        Extract metadata from file header.
        
        Args:
            lines: List of file lines
            
        Returns:
            Dictionary with metadata
        """
        metadata = {
            'title': '',
            'submercado': '',
            'month': '',
            'year': '',
            'version': ''
        }
        
        if len(lines) > 0:
            # First line usually contains title with month/year
            header = lines[0].strip()
            metadata['title'] = header
            
            # Extract month and year from header
            month_year_match = re.search(r'(\w+)\s*-\s*(\d{4})', header)
            if month_year_match:
                metadata['month'] = month_year_match.group(1)
                metadata['year'] = month_year_match.group(2)
            
            # Extract version
            version_match = re.search(r'NW.*Versao\s+([\d.]+)', header)
            if version_match:
                metadata['version'] = version_match.group(1)
        
        if len(lines) > 1:
            # Second line contains submercado
            submercado_line = lines[1].strip()
            submercado_match = re.search(r'SUBMERCADO:\s*(\w+)', submercado_line)
            if submercado_match:
                metadata['submercado'] = submercado_match.group(1)
        
        return metadata
    
    def _parse_data_block(self, lines: List[str], start_year: int = 2025) -> pd.DataFrame:
        """
        Parse the data block from the file.
        
        Args:
            lines: List of file lines
            start_year: Starting year for the data
            
        Returns:
            DataFrame with parsed data
        """
        data_rows = []
        current_year = start_year
        current_month = None
        
        for line in lines[3:]:  # Skip first 3 header lines
            line = line.strip()
            if not line or 'ANO:' in line or 'PAT' in line:
                continue
            
            # Try to parse data line
            parts = line.split()
            
            if len(parts) < 2:
                continue
            
            try:
                # Check if first element is a month number
                if parts[0].isdigit() and int(parts[0]) <= 15:
                    current_month = int(parts[0])
                    pat = int(parts[1])
                    
                    # Extract values (skip month and pat columns)
                    values = [float(v) if v != '0.00' else np.nan for v in parts[2:]]
                    
                    # Create date (approximate - using mid-month)
                    date = pd.Timestamp(year=current_year, month=current_month, day=15)
                    
                    data_rows.append({
                        'date': date,
                        'year': current_year,
                        'month': current_month,
                        'pat': pat,
                        'values': values
                    })
                    
            except (ValueError, IndexError):
                continue
        
        if not data_rows:
            return pd.DataFrame()
        
        # Process data rows into DataFrame
        processed_data = []
        
        for row in data_rows:
            date = row['date']
            values = row['values']
            pat = row['pat']
            
            # Create row with scenario values
            row_dict = {'date': date, 'pat': pat}
            
            # Add scenario columns (1-12) and MEDIA
            for i, val in enumerate(values):
                if i < 12:
                    row_dict[f'scenario_{i+1}'] = val
                elif i == 12:  # MEDIA column
                    row_dict['mean'] = val
            
            processed_data.append(row_dict)
        
        df = pd.DataFrame(processed_data)
        
        return df
    
    def load_newave(self) -> Optional[pd.DataFrame]:
        """
        Load the cmarg*.out file and return a pandas DataFrame.
        
        The DataFrame has:
        - Index: dates (monthly, mid-month timestamps)
        - Columns: 
            - 'pat': load pattern (1, 2, or 3)
            - 'scenario_1' to 'scenario_12': individual scenario results
            - 'mean': average across scenarios (MEDIA column)
        
        Returns:
            pandas.DataFrame: DataFrame with marginal costs or None if file not found
        """
        try:
            # Read file
            with open(self.newave_path, 'r', encoding='latin-1') as f:
                lines = f.readlines()
            
            # Extract metadata
            self.metadata = self._extract_metadata(lines)
            
            # Parse data
            df = self._parse_data_block(lines)
            
            if df.empty:
                print(f"Warning: No data parsed from {self.newave_path}")
                return None
            
            # Average across load patterns (PAT) to get single value per month
            df_agg = df.groupby('date').agg({
                **{f'scenario_{i}': 'mean' for i in range(1, 13)},
                'mean': 'mean'
            }).reset_index()
            
            # Set date as index
            df_agg.set_index('date', inplace=True)
            
            # Sort by date
            df_agg.sort_index(inplace=True)
            
            return df_agg
            
        except FileNotFoundError:
            print(f"Error: File not found at {self.newave_path}")
            return None
        except Exception as e:
            print(f"Error loading Newave data: {str(e)}")
            return None
    
    def get_mean_marginal_cost(self) -> Optional[pd.Series]:
        """
        Get the mean marginal cost series (averaged across all scenarios).
        
        Returns:
            pandas.Series: Series with dates as index and mean costs as values
        """
        df = self.load_newave()
        if df is None:
            return None
        
        return df['mean']
    
    def get_scenario_spread(self) -> Optional[pd.DataFrame]:
        """
        Calculate the spread (std deviation) across scenarios for each month.
        
        Returns:
            pandas.DataFrame: DataFrame with mean and std for each month
        """
        df = self.load_newave()
        if df is None:
            return None
        
        scenario_cols = [f'scenario_{i}' for i in range(1, 13)]
        
        result = pd.DataFrame({
            'mean': df['mean'],
            'std': df[scenario_cols].std(axis=1),
            'min': df[scenario_cols].min(axis=1),
            'max': df[scenario_cols].max(axis=1)
        })
        
        return result
    
    def filter_by_date_range(self, start_date: Optional[str] = None, 
                            end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Filter the data by date range.
        
        Args:
            start_date (str or datetime, optional): Start date for filtering
            end_date (str or datetime, optional): End date for filtering
            
        Returns:
            pandas.DataFrame: Filtered DataFrame
        """
        df = self.load_newave()
        if df is None:
            return None
        
        if start_date:
            start_date = pd.to_datetime(start_date)
            df = df[df.index >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            df = df[df.index <= end_date]
        
        return df
    
    def get_summary_statistics(self) -> Optional[Dict]:
        """
        Get summary statistics for the Newave data.
        
        Returns:
            Dictionary with summary statistics
        """
        df = self.load_newave()
        if df is None:
            return None
        
        scenario_cols = [f'scenario_{i}' for i in range(1, 13)]
        
        stats = {
            'date_range': {
                'start': df.index.min(),
                'end': df.index.max(),
                'n_months': len(df)
            },
            'mean_marginal_cost': {
                'overall_mean': df['mean'].mean(),
                'overall_std': df['mean'].std(),
                'min': df['mean'].min(),
                'max': df['mean'].max()
            },
            'scenario_variability': {
                'avg_std_across_scenarios': df[scenario_cols].std(axis=1).mean(),
                'max_std_across_scenarios': df[scenario_cols].std(axis=1).max()
            }
        }
        
        if hasattr(self, 'metadata'):
            stats['metadata'] = self.metadata
        
        return stats


# Example usage
if __name__ == "__main__":
    # Test the loader
    loader = NewaveLoader()
    
    print("Loading Newave data...")
    df = loader.load_newave()
    
    if df is not None:
        print("\n=== Data loaded successfully! ===")
        print(f"Shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())
        
        print("\n=== Summary Statistics ===")
        stats = loader.get_summary_statistics()
        if stats:
            print(f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
            print(f"Number of months: {stats['date_range']['n_months']}")
            print(f"Mean marginal cost: {stats['mean_marginal_cost']['overall_mean']:.2f}")
            print(f"Std deviation: {stats['mean_marginal_cost']['overall_std']:.2f}")
        
        print("\n=== Scenario Spread ===")
        spread = loader.get_scenario_spread()
        if spread is not None:
            print(spread.head())
    else:
        print("Failed to load Newave data")