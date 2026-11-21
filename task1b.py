from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from statsmodels.tsa.holtwinters import ExponentialSmoothing

VALIDATION_START = pd.Timestamp('2015-08-01')
TEST_START = pd.Timestamp('2016-02-01')
HORIZON = 3


def load_daily_sales():
    sales = pd.read_csv('Data/filtered_sales_evaluation.csv')
    calendar = pd.read_csv('Data/calendar.csv')
    calendar['date'] = pd.to_datetime(calendar['date'])

    day_columns = [column for column in sales if column.startswith('d_')]
    daily = sales.melt(
        id_vars=['item_id'], value_vars=day_columns,
        var_name='d', value_name='sales'
    ).merge(calendar, on='d', how='left')

    daily = daily[daily['date'] < TEST_START].copy()
    daily['weekday'] = daily['date'].dt.day_of_week
    daily['month'] = daily['date'].dt.month
    daily['holiday'] = (
        daily['event_name_1'].notna() | daily['event_name_2'].notna()
    ).astype(int)
    daily['snap'] = daily['snap_CA']
    return daily.sort_values(['item_id', 'date'])


def fit_initial_models(product_data, training_size):
    #Estimate regression and HW smoothing parameters on training data only.
    sales = product_data['sales'].to_numpy(dtype=float)
    features = pd.get_dummies(
        product_data[['weekday', 'month', 'snap', 'holiday']].astype(
            {'weekday': 'category', 'month': 'category'}
        ),
        columns=['weekday', 'month'], dtype=float
    )
    regression = Ridge(alpha=20).fit(features.iloc[:training_size], sales[:training_size])
    regression_predictions = np.maximum(0, regression.predict(features))

    hw_specs = {
        'HW additive': dict(trend=None, damped_trend=False),
        'HW damped': dict(trend='add', damped_trend=True),
    }
    hw_models = {}
    for name, spec in hw_specs.items():
        settings = dict(
            **spec, seasonal='add', seasonal_periods=7,
            initialization_method='estimated'
        )
        fitted = ExponentialSmoothing(sales[:training_size], **settings).fit(
            optimized=True, remove_bias=False
        )
        hw_models[name] = (settings, fitted.params)

    return regression_predictions, hw_models


def forecast_holt_winters(history, settings, fitted_params):
    # Update states with observed history, keeping training smoothing parameters fixed
    fit_args = dict(
        smoothing_level=fitted_params['smoothing_level'],
        smoothing_trend=fitted_params['smoothing_trend'],
        smoothing_seasonal=fitted_params['smoothing_seasonal'],
        optimized=False,
        remove_bias=False,
    )
    if settings['damped_trend']:
        fit_args['damping_trend'] = fitted_params['damping_trend']

    model = ExponentialSmoothing(history, **settings).fit(**fit_args)
    return float(np.maximum(0, model.forecast(HORIZON)).sum())


def forecast_all_methods(product_data, origin, training_size, regression_predictions, hw_models):
    # At origin, forecast sales on origin, origin+1 and origin+2.
    y = product_data['sales'].to_numpy(dtype=float)
    history = y[:origin]  # no sales from the forecast window
    future_weekdays = product_data['weekday'].iloc[origin:origin + HORIZON]

    predictions = {
        'Historical mean': HORIZON * y[:training_size].mean(),
        '28-day moving average': HORIZON * history[-28:].mean(),
        'Seasonal naive': sum(y[origin + h - 7] for h in range(HORIZON)),
        'Calendar regression': regression_predictions[origin:origin + HORIZON].sum(),
    }

    # Average the last eight observations for each corresponding weekday.
    recent = product_data.iloc[max(0, origin - 56):origin]
    predictions['Rolling weekday average'] = sum(
        recent.loc[recent['weekday'] == weekday, 'sales'].mean()
        for weekday in future_weekdays
    )

    for name, (settings, params) in hw_models.items():
        predictions[name] = forecast_holt_winters(history, settings, params)

    return predictions


def main():
    daily = load_daily_sales()
    predictions_rows = []

    for product, group in daily.groupby('item_id'):
        group = group.sort_values('date').reset_index(drop=True)
        y = group['sales'].to_numpy(dtype=float)
        training_size = int((group['date'] < VALIDATION_START).sum())
        regression_predictions, hw_models = fit_initial_models(group, training_size)

        # End early enough that every actual 3-day total stays before the test period.
        for origin in range(training_size, len(group) - HORIZON + 1):
            actual = float(y[origin:origin + HORIZON].sum())
            forecasts = forecast_all_methods(
                group, origin, training_size, regression_predictions, hw_models
            )
            for method, predicted in forecasts.items():
                predictions_rows.append({
                    'product': product,
                    'date': group.loc[origin, 'date'],
                    'model': method,
                    'actual_3day': actual,
                    'forecast_3day': float(predicted),
                    'error': actual - float(predicted),
                })
        print(f'Finished {product}')

    predictions = pd.DataFrame(predictions_rows)
    predictions.to_csv('Forecast_Results/three_day_predictions.csv', index=False)

    metrics = predictions.groupby(['product', 'model'])['error'].agg(
        MAE=lambda e: np.abs(e).mean(),
        RMSE=lambda e: np.sqrt(np.mean(e ** 2)),
        bias='mean',
        error_sd='std',
    ).reset_index().sort_values(['product', 'MAE'])
    metrics.to_csv('Forecast_Results/model_comparison.csv', index=False)

    print('\nThree-day demand forecast comparison (lower MAE is better):')
    for product, table in metrics.groupby('product'):
        print(f'\n{product}')
        print(table[['model', 'MAE', 'RMSE', 'bias', 'error_sd']].round(3).to_string(index=False))

    best = metrics.loc[metrics.groupby('product')['MAE'].idxmin()]
    print('\nBest method for each product:')
    print(best[['product', 'model', 'MAE', 'error_sd']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
