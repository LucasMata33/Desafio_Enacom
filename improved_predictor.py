import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')


class ImprovedPredictor:
    """
    Modelo melhorado para previsãao de preços de energia.
    
    Melhorias implementadas:
    1. Média móvel exponencial (EMA) para capturar tendências
    2. Decomposição sazonal para capturar padrões mensais/trimestrais
    3. Correlação entre maturidades
    4. Ajuste por volatilidade recente
    5. Ensemble de múltiplas estratégias
    """
    
    def __init__(self, data: pd.DataFrame = None):
        """
        Initialize the ImprovedPredictor.
        
        Args:
            data (pd.DataFrame): Historical price data with datetime index
        """
        self.data = data
        self.models_fitted = False
        self.ema_params = {}
        self.seasonal_patterns = {}
        self.correlation_matrix = None
        
    def fit(self):
        """
        Fit the model to the historical data.
        Calculates EMA parameters, seasonal patterns, and correlations.
        """
        if self.data is None or self.data.empty:
            raise ValueError("No data available for fitting")
        
        # Calcular correlações entre vencimentos
        self.correlation_matrix = self.data.corr()
        
        # Ajustar os parâmetros EMA para cada coluna
        for column in self.data.columns:
            if column != 'SPREAD':
                self._fit_ema_params(column)
                self._fit_seasonal_pattern(column)
        
        self.models_fitted = True
        
    def _fit_ema_params(self, column: str, lookback_windows: List[int] = [4, 12, 26]):
        """
        Fit exponential moving average parameters.
        
        Args:
            column: Column name to fit
            lookback_windows: Windows for EMA calculation (weeks)
        """
        self.ema_params[column] = {}
        
        for window in lookback_windows:
            ema = self.data[column].ewm(span=window, adjust=False).mean()
            self.ema_params[column][f'ema_{window}'] = ema.iloc[-1]
    
    def _fit_seasonal_pattern(self, column: str):
        """
        Extract seasonal patterns (monthly averages).
        
        Args:
            column: Column name to analyze
        """
        if column not in self.data.columns:
            return
        
        # Agrupar por mês e calcular a média
        monthly_avg = self.data[column].groupby(self.data.index.month).mean()
        
        self.seasonal_patterns[column] = monthly_avg.to_dict()
    
    def _get_seasonal_adjustment(self, column: str, target_date: pd.Timestamp) -> float:
        """
        Get seasonal adjustment factor for a given date.
        
        Args:
            column: Column name
            target_date: Target prediction date
            
        Returns:
            Seasonal adjustment factor (multiplier)
        """
        if column not in self.seasonal_patterns:
            return 1.0
        
        month = target_date.month
        
        # Obter a média do mês atual
        current_month = self.data.index[-1].month
        current_avg = self.seasonal_patterns[column].get(current_month, 1.0)
        target_avg = self.seasonal_patterns[column].get(month, 1.0)
        
        # Evite divisão por zero
        if current_avg == 0 or pd.isna(current_avg):
            return 1.0
        
        return target_avg / current_avg
    
    def _predict_with_trend(self, column: str, steps: int = 1) -> np.ndarray:
        """
        Predict using trend analysis (EMA).
        
        Args:
            column: Column to predict
            steps: Number of steps ahead
            
        Returns:
            Array of predictions
        """
        if column not in self.data.columns:
            return np.array([np.nan] * steps)
        
        last_value = self.data[column].iloc[-1]
        
        if pd.isna(last_value):
            return np.array([np.nan] * steps)
        
        # Use várias EMAs para detectar a tendência
        if column in self.ema_params:
            ema_4 = self.ema_params[column].get('ema_4', last_value)
            ema_12 = self.ema_params[column].get('ema_12', last_value)
            
            # Calcular tendência
            trend = (ema_4 - ema_12) / ema_12 if ema_12 != 0 else 0
            
            # Aplicar tendência com decaimento
            predictions = []
            for i in range(1, steps + 1):
                decay_factor = 0.8 ** i  # Trend decays over time
                pred = last_value * (1 + trend * decay_factor)
                predictions.append(pred)
            
            return np.array(predictions)
        
        return np.array([last_value] * steps)
    
    def _predict_with_correlation(self, column: str, steps: int = 1) -> np.ndarray:
        """
        Predict using correlations with other maturities.
        
        Args:
            column: Column to predict
            steps: Number of steps ahead
            
        Returns:
            Array of predictions
        """
        if self.correlation_matrix is None or column not in self.data.columns:
            return np.array([np.nan] * steps)
        
        last_value = self.data[column].iloc[-1]
        
        if pd.isna(last_value):
            return np.array([np.nan] * steps)
        
        # Encontrar colunas mais correlacionadas
        correlations = self.correlation_matrix[column].drop(column)
        top_correlated = correlations.nlargest(3)
        
        # Calcular a mudança média ponderada de colunas correlacionadass
        weighted_change = 0
        total_weight = 0
        
        for corr_col, corr_value in top_correlated.items():
            if corr_col in self.data.columns and len(self.data[corr_col]) >= 2:
                recent_change = (self.data[corr_col].iloc[-1] - self.data[corr_col].iloc[-2]) / self.data[corr_col].iloc[-2]
                if not pd.isna(recent_change):
                    weight = abs(corr_value)
                    weighted_change += recent_change * weight
                    total_weight += weight
        
        if total_weight > 0:
            avg_change = weighted_change / total_weight
            predictions = [last_value * (1 + avg_change * (0.7 ** i)) for i in range(steps)]
            return np.array(predictions)
        
        return np.array([last_value] * steps)
    
    def _predict_with_seasonality(self, column: str, steps: int = 1) -> np.ndarray:
        """
        Predict using seasonal patterns.
        
        Args:
            column: Column to predict
            steps: Number of steps ahead
            
        Returns:
            Array of predictions
        """
        if column not in self.data.columns:
            return np.array([np.nan] * steps)
        
        last_value = self.data[column].iloc[-1]
        last_date = self.data.index[-1]
        
        if pd.isna(last_value):
            return np.array([np.nan] * steps)
        
        predictions = []
        for i in range(1, steps + 1):
            target_date = last_date + pd.DateOffset(weeks=i)
            seasonal_factor = self._get_seasonal_adjustment(column, target_date)
            predictions.append(last_value * seasonal_factor)
        
        return np.array(predictions)
    
    def predict_next_step(self, steps: int = 1) -> pd.DataFrame:
        """
        Predict next steps using ensemble of methods.
        
        Args:
            steps: Number of steps to predict
            
        Returns:
            DataFrame with predictions
        """
        if self.data is None or self.data.empty:
            raise ValueError("No data available for prediction")
        
        if not self.models_fitted:
            self.fit()
        
        last_date = self.data.index[-1]
        predictions = []
        
        for i in range(1, steps + 1):
            next_date = last_date + pd.DateOffset(weeks=i)
            prediction_row = {}
            
            for column in self.data.columns:
                if column == 'SPREAD':
                    # SPREAD é calculado a partir de outras colunas
                    prediction_row[column] = np.nan
                else:
                    # Previsão em conjunto: média de três métodos
                    pred_trend = self._predict_with_trend(column, i)[-1]
                    pred_corr = self._predict_with_correlation(column, i)[-1]
                    pred_season = self._predict_with_seasonality(column, i)[-1]
                    
                    # Weighted ensemble
                    weights = [0.4, 0.35, 0.25]  # Tendência, Correlação, Sazonalidade
                    valid_preds = []
                    valid_weights = []
                    
                    for pred, weight in zip([pred_trend, pred_corr, pred_season], weights):
                        if not pd.isna(pred):
                            valid_preds.append(pred)
                            valid_weights.append(weight)
                    
                    if valid_preds:
                        # Normalize weights
                        total_weight = sum(valid_weights)
                        normalized_weights = [w / total_weight for w in valid_weights]
                        
                        ensemble_pred = sum(p * w for p, w in zip(valid_preds, normalized_weights))
                        prediction_row[column] = ensemble_pred
                    else:
                        # Reverter para o valor anterior
                        prediction_row[column] = self.data[column].iloc[-1]
            
            # Calcule o SPREAD se M+0 e A+0 estiverem disponíveis
            if 'M + 0' in prediction_row and 'A + 0' in prediction_row:
                m0 = prediction_row['M + 0']
                a0 = prediction_row['A + 0']
                if not pd.isna(m0) and not pd.isna(a0) and a0 != 0:
                    prediction_row['SPREAD'] = (m0 - a0) / a0
                else:
                    prediction_row['SPREAD'] = self.data['SPREAD'].iloc[-1]
            
            prediction_row['date'] = next_date
            predictions.append(prediction_row)
        
        pred_df = pd.DataFrame(predictions)
        pred_df.set_index('date', inplace=True)
        
        return pred_df
    
    def evaluate_prediction(self, test_data: pd.DataFrame, steps: int = 1) -> Dict:
        """
        Evaluate prediction performance on test data.
        
        Args:
            test_data: Test dataset
            steps: Number of steps to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.data is None:
            raise ValueError("No training data available")
        
        if not self.models_fitted:
            self.fit()
        
        common_columns = set(self.data.columns) & set(test_data.columns)
        evaluation_results = {}
        
        for column in common_columns:
            if column == 'SPREAD':
                continue
            
            # Obter previsões
            predictions = self.predict_next_step(steps)[column].values
            
            # Obter valores reais
            actual_values = test_data[column].head(steps).values
            
            # Calcular métricas
            valid_mask = ~(np.isnan(predictions) | np.isnan(actual_values))
            
            if valid_mask.any():
                valid_pred = predictions[valid_mask]
                valid_actual = actual_values[valid_mask]
                
                mae = np.mean(np.abs(valid_pred - valid_actual))
                mse = np.mean((valid_pred - valid_actual) ** 2)
                rmse = np.sqrt(mse)
                
                # MAPE
                nonzero_mask = valid_actual != 0
                if nonzero_mask.any():
                    mape = np.mean(np.abs((valid_pred[nonzero_mask] - valid_actual[nonzero_mask]) / valid_actual[nonzero_mask])) * 100
                else:
                    mape = float('inf')
                
                evaluation_results[column] = {
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAPE': mape,
                    'predictions': predictions.tolist(),
                    'actual_values': actual_values.tolist()
                }
        
        return evaluation_results
    
    def get_feature_importance(self) -> Dict:
        """
        Get feature importance metrics.
        
        Returns:
            Dictionary with feature importance information
        """
        if not self.models_fitted:
            self.fit()
        
        importance = {
            'correlation_strength': {},
            'trend_strength': {},
            'seasonality_strength': {}
        }
        
        for column in self.data.columns:
            if column == 'SPREAD':
                continue
            
            # Força de correlação (média das 3 principais correlações)
            if self.correlation_matrix is not None and column in self.correlation_matrix:
                top_corr = self.correlation_matrix[column].drop(column).nlargest(3)
                importance['correlation_strength'][column] = top_corr.mean()
            
            # Força da tendência (diferença entre EMAs curta e longa)
            if column in self.ema_params:
                ema_4 = self.ema_params[column].get('ema_4', 0)
                ema_26 = self.ema_params[column].get('ema_26', 0)
                if ema_26 != 0:
                    importance['trend_strength'][column] = abs((ema_4 - ema_26) / ema_26)
            
            # Força da sazonalidade (coeficiente de variação das médias mensais)
            if column in self.seasonal_patterns:
                values = list(self.seasonal_patterns[column].values())
                if values and np.mean(values) != 0:
                    importance['seasonality_strength'][column] = np.std(values) / np.mean(values)
        
        return importance