from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# -----------------------------------------------------------------------------
# 1. ASSUMPTIONS
# -----------------------------------------------------------------------------
VALIDATION_START = pd.Timestamp("2015-08-01")
VALIDATION_END = pd.Timestamp("2016-02-01")  # exclusive
LEAD_TIME_DAYS = 3
WARMUP_DAYS = 28

ORDER_COST = 1.00                         # dollars per SKU order line
PURCHASE_COST_FRACTION = 0.754            # fraction of selling price
ANNUAL_HOLDING_RATE = 0.25                # fraction of purchase cost
GOODWILL_COST = 0.25                      # dollars per lost sale

SERVICE_LEVELS = [0.90, 0.95, 0.99]
REVIEW_PERIODS = [1, 3, 7]

FORECAST_MODELS = {
    "FOODS_3_156": "moving_average",
    "FOODS_3_327": "holt_winters_seasonal",
    "FOODS_3_668": "holt_winters_seasonal",
    "FOODS_3_744": "holt_winters_damped",
    "HOUSEHOLD_1_532": "moving_average",
}


# -----------------------------------------------------------------------------
# 2. LOAD SALES AND PRICES
# -----------------------------------------------------------------------------
def load_inputs():
    sales = pd.read_csv("Data/filtered_sales_evaluation.csv")
    calendar = pd.read_csv("Data/calendar.csv")
    prices = pd.read_csv("Data/filtered_prices.csv")

    calendar["date"] = pd.to_datetime(calendar["date"])
    date_lookup = calendar.set_index("d")["date"]

    day_columns = [column for column in sales if column.startswith("d_")]
    sales_long = sales.melt(
        id_vars="item_id", value_vars=day_columns,
        var_name="day_code", value_name="demand"
    )
    sales_long["date"] = sales_long["day_code"].map(date_lookup)
    sales_long = sales_long[sales_long["date"] < VALIDATION_END]
    sales_long = sales_long.sort_values(["item_id", "date"])

    # Match the original comparison: average weekly prices during validation.
    validation_weeks = calendar.loc[
        (calendar["date"] >= VALIDATION_START)
        & (calendar["date"] < VALIDATION_END), "wm_yr_wk"
    ].unique()
    average_prices = (
        prices[prices["wm_yr_wk"].isin(validation_weeks)]
        .groupby("item_id")["sell_price"].mean().to_dict()
    )
    return sales_long, average_prices


# -----------------------------------------------------------------------------
# 3. FORECAST FUTURE DEMAND AND MEASURE FORECAST ERROR
# -----------------------------------------------------------------------------
def make_forecasts(product, product_sales):
    all_demand = product_sales["demand"].to_numpy(dtype=float)
    training_days = (product_sales["date"] < VALIDATION_START).sum()
    validation_demand = all_demand[training_days:].astype(int)

    horizons = sorted({LEAD_TIME_DAYS} | {
        LEAD_TIME_DAYS + period for period in REVIEW_PERIODS
    })
    max_horizon = max(horizons)
    forecasts = {days: np.zeros(len(validation_demand)) for days in horizons}

    model = FORECAST_MODELS[product]
    if model != "moving_average":
        damped = model == "holt_winters_damped"
        settings = dict(
            trend="add" if damped else None,
            damped_trend=damped,
            seasonal="add",
            seasonal_periods=7,
            initialization_method="estimated",
        )
        # Fit smoothing parameters on pre-validation observations only.
        initial_fit = ExponentialSmoothing(
            all_demand[:training_days], **settings
        ).fit(optimized=True, remove_bias=False)
        smoothing = initial_fit.params
        fixed_settings = dict(
            smoothing_level=smoothing["smoothing_level"],
            smoothing_trend=smoothing["smoothing_trend"],
            smoothing_seasonal=smoothing["smoothing_seasonal"],
            optimized=False,
            remove_bias=False,
        )
        if damped:
            fixed_settings["damping_trend"] = smoothing["damping_trend"]

    for day_index in range(len(validation_demand)):
        absolute_day = training_days + day_index
        history = all_demand[:absolute_day + 1]

        if model == "moving_average":
            daily_forecast = np.full(max_horizon, history[-28:].mean())
        else:
            fitted = ExponentialSmoothing(history, **settings).fit(**fixed_settings)
            daily_forecast = np.maximum(0, fitted.forecast(max_horizon))

        cumulative_forecast = np.cumsum(daily_forecast)
        for days in horizons:
            forecasts[days][day_index] = cumulative_forecast[days - 1]

    # Compare each forecast with the following days' actual demand.
    # This reproduces the original script's validation-period calibration.
    forecast_bias = {}
    forecast_sd = {}
    for days in horizons:
        valid_count = len(validation_demand) - days
        actual_future_demand = np.array([
            all_demand[training_days + t + 1:
                       training_days + t + 1 + days].sum()
            for t in range(valid_count)
        ])
        errors = actual_future_demand - forecasts[days][:valid_count]
        forecast_bias[days] = float(errors.mean())
        forecast_sd[days] = float(errors.std(ddof=1))

    return validation_demand, forecasts, forecast_bias, forecast_sd


# -----------------------------------------------------------------------------
# 4. COSTS AND EOQ
# -----------------------------------------------------------------------------
def calculate_costs(selling_price, training_daily_demand):
    purchase_cost = PURCHASE_COST_FRACTION * selling_price
    daily_holding_cost = ANNUAL_HOLDING_RATE * purchase_cost / 365
    lost_sale_cost = selling_price - purchase_cost + GOODWILL_COST

    # EOQ: balances ordering and holding costs; it does not model lost sales.
    eoq = np.sqrt(2 * training_daily_demand * ORDER_COST / daily_holding_cost)
    order_quantity = max(1, int(np.ceil(eoq)))
    return purchase_cost, daily_holding_cost, lost_sale_cost, order_quantity


# -----------------------------------------------------------------------------
# 5. SIMULATE POLICY
# -----------------------------------------------------------------------------
def simulate_policy(demand, forecasts, forecast_bias, forecast_sd,
                    holding_cost, lost_sale_cost, policy_type,
                    policy_parameter, safety_factor):
    if policy_type == "continuous":
        order_quantity = policy_parameter
        horizon = LEAD_TIME_DAYS
        initial_stock = np.ceil(
            forecasts[horizon][0] + forecast_bias[horizon]
            + safety_factor * forecast_sd[horizon] + order_quantity / 2
        )
    else:
        review_period = policy_parameter
        horizon = review_period + LEAD_TIME_DAYS
        initial_stock = np.ceil(
            forecasts[horizon][0] + forecast_bias[horizon]
            + safety_factor * forecast_sd[horizon]
        )

    on_hand = max(0, int(initial_stock))
    scheduled_arrivals = {}  # arrival day -> units due
    holding_total = shortage_total = 0.0
    orders_placed = lost_units = sold_units = 0
    sum_average_stock = 0.0

    for day, units_requested in enumerate(demand):
        # Receive any order placed three days earlier.
        on_hand += scheduled_arrivals.pop(day, 0)
        morning_stock = on_hand

        units_sold_today = min(on_hand, units_requested)
        units_lost_today = units_requested - units_sold_today
        on_hand -= units_sold_today
        average_stock_today = (morning_stock + on_hand) / 2

        # Warm-up days change inventory, but are excluded from reported costs.
        if day >= WARMUP_DAYS:
            holding_total += holding_cost * average_stock_today
            shortage_total += lost_sale_cost * units_lost_today
            sold_units += units_sold_today
            lost_units += units_lost_today
            sum_average_stock += average_stock_today

        inventory_position = on_hand + sum(scheduled_arrivals.values())
        order_units = 0

        if policy_type == "continuous":
            reorder_point = int(np.ceil(
                forecasts[horizon][day] + forecast_bias[horizon]
                + safety_factor * forecast_sd[horizon]
            ))
            if inventory_position <= reorder_point:
                # Order enough full Q-sized batches to exceed the reorder point.
                batches = max(1, int(np.ceil(
                    (reorder_point + 1 - inventory_position) / order_quantity
                )))
                order_units = batches * order_quantity

        else:  # periodic review
            if day % review_period == 0:
                order_up_to = int(np.ceil(
                    forecasts[horizon][day] + forecast_bias[horizon]
                    + safety_factor * forecast_sd[horizon]
                ))
                order_units = max(0, order_up_to - inventory_position)

        if order_units > 0:
            arrival_day = day + LEAD_TIME_DAYS
            scheduled_arrivals[arrival_day] = (
                scheduled_arrivals.get(arrival_day, 0) + order_units
            )
            if day >= WARMUP_DAYS:
                orders_placed += 1

    evaluated_days = len(demand) - WARMUP_DAYS
    ordering_total = orders_placed * ORDER_COST
    total_cost = ordering_total + holding_total + shortage_total

    return {
        "total_cost": total_cost,
        "ordering_cost": ordering_total,
        "holding_cost": holding_total,
        "shortage_cost": shortage_total,
        "orders": orders_placed,
        "orders_per_30_days": orders_placed * 30 / evaluated_days,
        "avg_inventory": sum_average_stock / evaluated_days,
        "lost_units": lost_units,
        "fill_rate": sold_units / (sold_units + lost_units),
        "mean_daily_cost": total_cost / evaluated_days,
    }


# -----------------------------------------------------------------------------
# 6. COMPARE CONTINUOUS AND PERIODIC POLICIES
# -----------------------------------------------------------------------------
def main():
    sales, prices = load_inputs()
    candidate_rows = []
    product_rows = []
    standard_normal = NormalDist()

    for product, product_sales in sales.groupby("item_id"):
        product_sales = product_sales.sort_values("date").reset_index(drop=True)
        selling_price = prices[product]
        training_mean = product_sales.loc[
            product_sales["date"] < VALIDATION_START, "demand"
        ].mean()

        purchase_cost, holding_cost, lost_sale_cost, eoq = calculate_costs(
            selling_price, training_mean
        )
        demand, forecasts, bias, error_sd = make_forecasts(product, product_sales)

        product_rows.append({
            "product": product,
            "mean_daily_demand_training": training_mean,
            "selling_price": selling_price,
            "purchase_cost": purchase_cost,
            "holding_cost_per_unit_day": holding_cost,
            "lost_sale_cost_per_unit": lost_sale_cost,
            "EOQ_units": eoq,
            "lead_time_error_sd": error_sd[LEAD_TIME_DAYS],
            "lead_time_error_bias": bias[LEAD_TIME_DAYS],
        })

        # Continuous review: one EOQ, three service-level targets.
        for service_level in SERVICE_LEVELS:
            z = standard_normal.inv_cdf(service_level)
            metrics = simulate_policy(
                demand, forecasts, bias, error_sd, holding_cost, lost_sale_cost,
                policy_type="continuous", policy_parameter=eoq, safety_factor=z
            )
            reorder_points = np.ceil(
                forecasts[LEAD_TIME_DAYS] + bias[LEAD_TIME_DAYS]
                + z * error_sd[LEAD_TIME_DAYS]
            )
            candidate_rows.append({
                "product": product,
                "policy": "continuous",
                "parameter": eoq,
                "target_cycle_service": service_level,
                "critical_ratio": np.nan,
                "z": z,
                "average_reorder_point": reorder_points[WARMUP_DAYS:].mean(),
                "nominal_safety_stock": z * error_sd[LEAD_TIME_DAYS],
                **metrics,
            })

        # Periodic review: critical ratio with approximate overage cost h * P.
        for review_period in REVIEW_PERIODS:
            protection_days = review_period + LEAD_TIME_DAYS
            overage_cost = holding_cost * review_period
            critical_ratio = lost_sale_cost / (lost_sale_cost + overage_cost)
            z = standard_normal.inv_cdf(critical_ratio)

            metrics = simulate_policy(
                demand, forecasts, bias, error_sd, holding_cost, lost_sale_cost,
                policy_type="periodic", policy_parameter=review_period,
                safety_factor=z
            )
            order_up_to_levels = np.ceil(
                forecasts[protection_days] + bias[protection_days]
                + z * error_sd[protection_days]
            )
            candidate_rows.append({
                "product": product,
                "policy": "periodic",
                "parameter": review_period,
                "target_cycle_service": np.nan,
                "critical_ratio": critical_ratio,
                "z": z,
                "average_order_up_to_level": order_up_to_levels[WARMUP_DAYS:].mean(),
                "nominal_safety_stock": z * error_sd[protection_days],
                **metrics,
            })

        print(f"Finished {product}", flush=True)

    all_candidates = pd.DataFrame(candidate_rows)
    cost_parameters = pd.DataFrame(product_rows)
    best_by_policy = (
        all_candidates.sort_values("total_cost")
        .groupby(["product", "policy"], as_index=False).first()
    )

    all_candidates.to_csv("Inventory_Policy_Results/all_policy_candidates.csv", index=False)
    cost_parameters.to_csv("Inventory_Policy_Results/cost_and_eoq_parameters.csv", index=False)
    best_by_policy.to_csv("Inventory_Policy_Results/best_policy_by_product.csv", index=False)

    columns = ["product", "policy", "parameter", "target_cycle_service",
               "critical_ratio", "total_cost", "fill_rate", "orders_per_30_days"]
    print("\nBEST POLICY SETTINGS FOR EACH PRODUCT\n")
    print(best_by_policy[columns].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
