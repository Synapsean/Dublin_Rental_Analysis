"""
Advanced Time Series Model Validation
====================================

Implements industry-standard validation techniques for time series forecasting:
- Walk-forward analysis (time series cross-validation)
- Backtesting with expanding/sliding windows
- Statistical significance testing
- Residual diagnostics
- Model comparison framework
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


class TimeSeriesValidator:
    """
    Robust time series validation framework.
    
    Key Principles:
    1. No future data leakage
    2. Respects temporal order
    3. Multiple evaluation periods
    4. Statistical significance testing
    """
    
    def __init__(self, ts_data, date_col='date', target_col='avg_rent'):
        self.ts = ts_data.copy()
        self.date_col = date_col
        self.target_col = target_col
        self.results = {}
    
    def walk_forward_validation(self, model_type='linear', test_size=4, step_size=2):
        """
        Walk-forward analysis - the gold standard for time series validation.
        
        How it works:
        1. Start with minimum training data
        2. Predict next N periods
        3. Move window forward by step_size
        4. Repeat until end of data
        
        Args:
            model_type: 'linear', 'prophet', or 'baseline'
            test_size: How many periods to predict each iteration
            step_size: How many periods to advance each step
        
        Returns:
            Dict with validation results and metrics
        """
        predictions = []
        actuals = []
        dates = []
        
        # Minimum training size (need enough data for meaningful model)
        min_train_size = 20  # ~5 years of quarterly data
        
        for i in range(min_train_size, len(self.ts) - test_size + 1, step_size):
            # Split data
            train = self.ts.iloc[:i].copy()
            test = self.ts.iloc[i:i+test_size].copy()
            
            if len(test) < test_size:
                break
                
            # Train model
            if model_type == 'linear':
                model, pred = self._train_linear(train, test)
            elif model_type == 'prophet' and PROPHET_AVAILABLE:
                model, pred = self._train_prophet(train, len(test))
            elif model_type == 'baseline':
                pred = [train[self.target_col].iloc[-1]] * len(test)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Store results
            predictions.extend(pred)
            actuals.extend(test[self.target_col].tolist())
            dates.extend(test[self.date_col].tolist())
        
        # Calculate metrics
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / np.array(actuals))) * 100
        
        results = {
            'model_type': model_type,
            'predictions': predictions,
            'actuals': actuals,
            'dates': dates,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'n_predictions': len(predictions)
        }
        
        self.results[model_type] = results
        return results
    
    def expanding_window_validation(self, model_type='linear', min_train_size=24):
        """
        Expanding window validation - train on all historical data.
        
        More realistic for production where you'd use all available data.
        """
        predictions = []
        actuals = []
        dates = []
        
        for i in range(min_train_size, len(self.ts) - 1):
            # Expanding window: use all data up to time i
            train = self.ts.iloc[:i].copy()
            test_point = self.ts.iloc[i].copy()
            
            # Train and predict next point
            if model_type == 'linear':
                model, pred = self._train_linear(train, pd.DataFrame([test_point]))
            elif model_type == 'prophet' and PROPHET_AVAILABLE:
                model, pred = self._train_prophet(train, 1)
            elif model_type == 'baseline':
                pred = [train[self.target_col].iloc[-1]]
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            predictions.append(pred[0])
            actuals.append(test_point[self.target_col])
            dates.append(test_point[self.date_col])
        
        # Calculate metrics
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / np.array(actuals))) * 100
        
        return {
            'model_type': f'{model_type}_expanding',
            'predictions': predictions,
            'actuals': actuals,
            'dates': dates,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'n_predictions': len(predictions)
        }
    
    def statistical_comparison(self, model1_results, model2_results):
        """
        Statistical significance test between two models.
        
        Uses Diebold-Mariano test for forecast accuracy comparison.
        """
        # Get absolute errors for both models
        errors1 = np.abs(np.array(model1_results['actuals']) - np.array(model1_results['predictions']))
        errors2 = np.abs(np.array(model2_results['actuals']) - np.array(model2_results['predictions']))
        
        # Ensure same length
        min_len = min(len(errors1), len(errors2))
        errors1 = errors1[:min_len]
        errors2 = errors2[:min_len]
        
        # Difference in errors
        error_diff = errors1 - errors2
        
        # Two-tailed t-test
        t_stat, p_value = stats.ttest_1samp(error_diff, 0)
        
        result = {
            'model1': model1_results['model_type'],
            'model2': model2_results['model_type'],
            'model1_mae': model1_results['mae'],
            'model2_mae': model2_results['mae'],
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'better_model': model1_results['model_type'] if model1_results['mae'] < model2_results['mae'] else model2_results['model_type']
        }
        
        return result
    
    def residual_diagnostics(self, model_results):
        """
        Check if model residuals satisfy time series assumptions.
        
        Tests:
        1. Normality (Shapiro-Wilk)
        2. Serial correlation (Ljung-Box)
        3. Heteroscedasticity
        """
        residuals = np.array(model_results['actuals']) - np.array(model_results['predictions'])
        
        # Normality test
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        
        # Serial correlation test (simplified)
        # In practice, would use statsmodels diagnostic_ljungbox
        lag1_corr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        
        # Heteroscedasticity (variance over time)
        mid_point = len(residuals) // 2
        early_var = np.var(residuals[:mid_point])
        late_var = np.var(residuals[mid_point:])
        var_ratio = late_var / early_var if early_var > 0 else np.inf
        
        return {
            'model_type': model_results['model_type'],
            'residual_mean': np.mean(residuals),
            'residual_std': np.std(residuals),
            'shapiro_test': {'statistic': shapiro_stat, 'p_value': shapiro_p, 'normal': shapiro_p > 0.05},
            'serial_correlation': {'lag1_correlation': lag1_corr, 'no_correlation': abs(lag1_corr) < 0.1},
            'heteroscedasticity': {'variance_ratio': var_ratio, 'homogeneous': 0.5 < var_ratio < 2.0}
        }
    
    def _train_linear(self, train_data, test_data):
        """Helper: Train linear regression model."""
        train = train_data.copy()
        train['time_idx'] = range(len(train))
        
        model = LinearRegression()
        model.fit(train[['time_idx']], train[self.target_col])
        
        # Predict
        test_indices = np.array([[len(train) + i] for i in range(len(test_data))])
        predictions = model.predict(test_indices)
        
        return model, predictions.tolist()
    
    def _train_prophet(self, train_data, periods):
        """Helper: Train Prophet model."""
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet not available")
        
        prophet_df = train_data[[self.date_col, self.target_col]].copy()
        prophet_df.columns = ['ds', 'y']
        
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(prophet_df)
        
        future = model.make_future_dataframe(periods=periods, freq='QS')
        forecast = model.predict(future)
        
        predictions = forecast['yhat'].tail(periods).tolist()
        return model, predictions


def comprehensive_validation_report(ts_data):
    """
    Generate complete validation report comparing all models.
    """
    validator = TimeSeriesValidator(ts_data)
    
    # Run validations
    print("Running comprehensive model validation...")
    
    # Walk-forward validation
    linear_wf = validator.walk_forward_validation('linear')
    baseline_wf = validator.walk_forward_validation('baseline')
    
    if PROPHET_AVAILABLE:
        prophet_wf = validator.walk_forward_validation('prophet')
    
    # Expanding window validation
    linear_exp = validator.expanding_window_validation('linear')
    
    # Statistical comparisons
    linear_vs_baseline = validator.statistical_comparison(linear_wf, baseline_wf)
    
    if PROPHET_AVAILABLE:
        prophet_vs_linear = validator.statistical_comparison(prophet_wf, linear_wf)
        prophet_vs_baseline = validator.statistical_comparison(prophet_wf, baseline_wf)
    
    # Residual diagnostics
    linear_diag = validator.residual_diagnostics(linear_wf)
    
    if PROPHET_AVAILABLE:
        prophet_diag = validator.residual_diagnostics(prophet_wf)
    
    # Compile report
    report = {
        'validation_results': {
            'linear_walkforward': linear_wf,
            'baseline_walkforward': baseline_wf,
            'linear_expanding': linear_exp
        },
        'statistical_tests': {
            'linear_vs_baseline': linear_vs_baseline
        },
        'diagnostics': {
            'linear': linear_diag
        }
    }
    
    if PROPHET_AVAILABLE:
        report['validation_results']['prophet_walkforward'] = prophet_wf
        report['statistical_tests']['prophet_vs_linear'] = prophet_vs_linear
        report['statistical_tests']['prophet_vs_baseline'] = prophet_vs_baseline
        report['diagnostics']['prophet'] = prophet_diag
    
    return report


if __name__ == "__main__":
    # Example usage
    from forecast_rents import load_data, prepare_aggregate_timeseries
    
    df = load_data()
    ts = prepare_aggregate_timeseries(df, "Dublin", "All property types", "All bedrooms")
    
    report = comprehensive_validation_report(ts)
    
    print("\\n=== MODEL VALIDATION RESULTS ===")
    for model_type, results in report['validation_results'].items():
        print(f"\\n{model_type.upper()}:")
        print(f"  MAE: €{results['mae']:.2f}")
        print(f"  RMSE: €{results['rmse']:.2f}")
        print(f"  MAPE: {results['mape']:.2f}%")
        print(f"  Predictions: {results['n_predictions']}")