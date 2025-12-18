"""Core biotech / agro financial modelling primitives."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Callable

import numpy as np
import pandas as pd


# =====================================
# 1. Configuration dataclasses (inputs)
# =====================================


@dataclass
class ModelConfig:
    first_year: int = 2024
    n_years: int = 25
    currency: str = "USD"
    tax_rate: float = 0.25
    discount_rate: float = 0.10
    ev_ebitda_multiple: float = 8.0
    working_capital_pct_sales: float = 0.08
    sales_ramp_factors: List[float] | None = None

    def __post_init__(self) -> None:
        if self.sales_ramp_factors is None:
            self.sales_ramp_factors = [0.20, 0.60, 1.00, 1.00, 1.00]

    @property
    def years(self) -> np.ndarray:
        return np.arange(self.first_year, self.first_year + self.n_years)


@dataclass
class ProductConfig:
    name: str
    stage: str
    success_prob: float
    include_in_consolidation: bool = True

    time_to_market: int = 3
    patent_years: int = 20
    preexisting_market: bool = False

    patent_revenue_target: float = 0.0
    post_patent_revenue_target: float = 0.0
    market_growth_patent: float = 0.005
    market_growth_post: float = 0.0

    cogs_patent: float = 0.30
    cogs_post: float = 0.50
    sales_marketing_pct: float = 0.15
    gna_pct: float = 0.10
    royalty_pct: float = 0.0

    rd_remaining_pre_launch: float = 0.0
    rd_annual_post_launch: float = 0.0

    capex_remaining_pre_launch: float = 0.0
    capex_annual_post_launch: float = 0.0

    rd_capitalization_ratio: float = 0.5
    rd_amort_years: int = 10
    capex_dep_years: int = 10


# ===========================
# 2. Core model classes
# ===========================


class Product:
    """Represents a single biotech (or agro) product/asset."""

    def __init__(self, config: ProductConfig, model_config: ModelConfig):
        self.config = config
        self.model_config = model_config

    def _launch_year(self) -> int:
        if self.config.preexisting_market:
            return self.model_config.first_year
        return self.model_config.first_year + max(self.config.time_to_market, 0)

    def _patent_end_year(self) -> int:
        return self._launch_year() + self.config.patent_years - 1

    @staticmethod
    def _rolling_amortization(additions: pd.Series, life: int) -> pd.Series:
        years = additions.index
        amort = pd.Series(0.0, index=years)
        if life <= 0:
            return amort
        for i in range(len(years)):
            add = additions.iloc[i]
            if add == 0:
                continue
            annual = add / life
            for j in range(i, min(i + life, len(years))):
                amort.iloc[j] += annual
        return amort

    def build_revenue_series(self) -> pd.Series:
        years = self.model_config.years
        cfg = self.config
        revenue = pd.Series(0.0, index=years, name=f"{cfg.name}_revenue")
        if not cfg.include_in_consolidation:
            return revenue

        launch_year = self._launch_year()
        patent_end = self._patent_end_year()

        for i, year in enumerate(years):
            if year < launch_year:
                continue
            years_since_launch = year - launch_year
            in_patent = year <= patent_end

            if in_patent:
                base_target = cfg.patent_revenue_target
                growth_rate = cfg.market_growth_patent
                years_since_growth_start = max(
                    0, years_since_launch - len(self.model_config.sales_ramp_factors)
                )
            else:
                base_target = cfg.post_patent_revenue_target
                growth_rate = cfg.market_growth_post
                years_since_growth_start = max(0, year - (patent_end + 1))

            ramp = (
                self.model_config.sales_ramp_factors[years_since_launch]
                if years_since_launch < len(self.model_config.sales_ramp_factors)
                else 1.0
            )

            target_with_growth = base_target * ((1 + growth_rate) ** years_since_growth_start)
            revenue.iloc[i] = ramp * target_with_growth

        return revenue

    def build_cashflow_table(self) -> pd.DataFrame:
        years = self.model_config.years
        cfg = self.config
        df = pd.DataFrame(index=years)
        df["revenue"] = self.build_revenue_series()

        patent_end = self._patent_end_year()
        cogs_vals: List[float] = []
        for year, rev in df["revenue"].items():
            if rev == 0:
                cogs_vals.append(0.0)
                continue
            pct = cfg.cogs_patent if year <= patent_end else cfg.cogs_post
            cogs_vals.append(-pct * rev)
        df["cogs"] = cogs_vals

        df["sales_marketing"] = -cfg.sales_marketing_pct * df["revenue"]
        df["gna"] = -cfg.gna_pct * df["revenue"]
        df["royalty"] = -cfg.royalty_pct * df["revenue"]

        rd_cash = pd.Series(0.0, index=years)
        if cfg.rd_remaining_pre_launch > 0 and not cfg.preexisting_market:
            pre_years = max(1, cfg.time_to_market)
            annual_pre = cfg.rd_remaining_pre_launch / pre_years
            for i in range(pre_years):
                rd_cash.iloc[i] -= annual_pre

        launch_year = self._launch_year()
        for i, year in enumerate(years):
            if year >= launch_year:
                rd_cash.iloc[i] -= cfg.rd_annual_post_launch
        df["rd_cash"] = rd_cash

        rd_cap_add = rd_cash * cfg.rd_capitalization_ratio
        rd_expensed_current = rd_cash * (1 - cfg.rd_capitalization_ratio)
        rd_amort = self._rolling_amortization(rd_cap_add, cfg.rd_amort_years)
        df["rd_cap_add"] = rd_cap_add
        df["rd_amort"] = rd_amort
        df["rd_expense_pnl"] = rd_expensed_current + rd_amort

        capex_cash = pd.Series(0.0, index=years)
        if cfg.capex_remaining_pre_launch > 0 and not cfg.preexisting_market:
            pre_years = max(1, cfg.time_to_market)
            annual_pre_cx = cfg.capex_remaining_pre_launch / pre_years
            for i in range(pre_years):
                capex_cash.iloc[i] -= annual_pre_cx

        for i, year in enumerate(years):
            if year >= launch_year:
                capex_cash.iloc[i] -= cfg.capex_annual_post_launch
        df["capex_cash"] = capex_cash

        depreciation = self._rolling_amortization(capex_cash, cfg.capex_dep_years)
        df["depreciation"] = depreciation

        df["ebit"] = (
            df["revenue"]
            + df["cogs"]
            + df["sales_marketing"]
            + df["gna"]
            + df["royalty"]
            + df["rd_expense_pnl"]
        )
        df["da"] = -(df["rd_amort"] + df["depreciation"])
        df["ebitda"] = df["ebit"] + df["da"]

        tax_rate = self.model_config.tax_rate
        df["tax"] = 0.0
        positive_ebit = df["ebit"] > 0
        df.loc[positive_ebit, "tax"] = -tax_rate * df.loc[positive_ebit, "ebit"]
        df["nopat"] = df["ebit"] + df["tax"]

        df["fcff"] = df["nopat"] + df["da"] + df["capex_cash"] + df["rd_cap_add"]
        return df

    def build_probability_weighted_table(self) -> pd.DataFrame:
        df = self.build_cashflow_table().copy()
        p = self.config.success_prob
        for col in ["revenue", "ebit", "ebitda", "nopat", "fcff"]:
            df[col] = df[col] * p
        return df


class Portfolio:
    """A collection of products that can be valued together."""

    def __init__(self, products: List[Product], model_config: ModelConfig):
        self.products = products
        self.model_config = model_config

    def consolidated_table(self) -> Dict[str, pd.DataFrame | pd.Series]:
        years = self.model_config.years
        base_cols = [
            "revenue",
            "cogs",
            "sales_marketing",
            "gna",
            "royalty",
            "rd_cash",
            "rd_cap_add",
            "rd_amort",
            "rd_expense_pnl",
            "capex_cash",
            "depreciation",
            "ebit",
            "da",
            "ebitda",
            "tax",
            "nopat",
            "fcff",
        ]
        cons_df = pd.DataFrame(0.0, index=years, columns=base_cols)

        per_product: Dict[str, pd.DataFrame] = {}
        per_product_prob: Dict[str, pd.DataFrame] = {}

        for prod in self.products:
            cfg = prod.config
            if not cfg.include_in_consolidation:
                continue
            df = prod.build_cashflow_table()
            per_product[cfg.name] = df
            wdf = prod.build_probability_weighted_table()
            per_product_prob[cfg.name] = wdf
            cons_df = cons_df.add(wdf[base_cols], fill_value=0.0)

        wc = self.model_config.working_capital_pct_sales * cons_df["revenue"]
        wc_diff = wc.diff().fillna(wc)
        cons_df["delta_wc"] = -wc_diff
        cons_df["fcff_after_wc"] = cons_df["fcff"] + cons_df["delta_wc"]

        return {
            "per_product": per_product,
            "per_product_prob": per_product_prob,
            "consolidated": cons_df,
        }


# ======================
# 3. Valuation engine
# ======================


@dataclass
class ValuationResult:
    portfolio: Portfolio
    rnpv: float
    dcf_table: pd.DataFrame
    consolidated: pd.DataFrame
    per_product: Dict[str, pd.DataFrame]
    per_product_prob: Dict[str, pd.DataFrame]


class ValuationEngine:
    """Runs DCF valuation (rNPV, terminal value) on a Portfolio."""

    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.model_config = portfolio.model_config

    def _discounted_cash_flows(self, fcff: pd.Series) -> pd.DataFrame:
        years = fcff.index.values
        t = np.arange(len(years))
        df = pd.DataFrame(index=years)
        df["t"] = t
        df["fcff"] = fcff.values
        df["discount_factor"] = 1.0 / ((1 + self.model_config.discount_rate) ** t)
        df["discounted_fcff"] = df["fcff"] * df["discount_factor"]
        return df

    def _add_terminal_value(self, dcf_df: pd.DataFrame, cons_df: pd.DataFrame) -> float:
        last_year = cons_df.index[-1]
        last_ebitda = cons_df.loc[last_year, "ebitda"]
        multiple = self.model_config.ev_ebitda_multiple
        terminal_ev = multiple * last_ebitda

        t_last = dcf_df.loc[last_year, "t"]
        dcf_df.loc[last_year, "terminal_value"] = terminal_ev
        dcf_df.loc[last_year, "discounted_terminal_value"] = terminal_ev / (
            (1 + self.model_config.discount_rate) ** t_last
        )
        rnpv = dcf_df["discounted_fcff"].sum() + dcf_df["discounted_terminal_value"].sum()
        return rnpv

    def run(self) -> ValuationResult:
        agg = self.portfolio.consolidated_table()
        cons = agg["consolidated"]
        dcf_df = self._discounted_cash_flows(cons["fcff_after_wc"])
        rnpv = self._add_terminal_value(dcf_df, cons)
        return ValuationResult(
            portfolio=self.portfolio,
            rnpv=rnpv,
            dcf_table=dcf_df,
            consolidated=cons,
            per_product=agg["per_product"],
            per_product_prob=agg["per_product_prob"],
        )


# ======================
# 4. VC-style valuation
# ======================


@dataclass
class VCInputs:
    exit_year: int
    target_irr: float
    investor_ownership_at_exit: float
    new_money: float


class VCValuator:
    def __init__(self, valuation_result: ValuationResult):
        self.result = valuation_result
        self.model_config = valuation_result.portfolio.model_config

    def compute_exit_ev(self, exit_year: int, multiple: Optional[float] = None) -> float:
        if multiple is None:
            multiple = self.model_config.ev_ebitda_multiple
        cons = self.result.consolidated
        if exit_year not in cons.index:
            raise ValueError("Exit year not in consolidated index")
        exit_ebitda = cons.loc[exit_year, "ebitda"]
        return float(exit_ebitda * multiple)

    def vc_method(self, vc_inputs: VCInputs, exit_multiple: Optional[float] = None) -> Dict[str, float]:
        exit_ev = self.compute_exit_ev(vc_inputs.exit_year, exit_multiple)
        years_to_exit = vc_inputs.exit_year - self.model_config.first_year
        if years_to_exit <= 0:
            raise ValueError("Exit year must be after first model year")

        investor_exit_value = exit_ev * vc_inputs.investor_ownership_at_exit
        investor_pv_required = investor_exit_value / ((1 + vc_inputs.target_irr) ** years_to_exit)
        implied_post_money = investor_pv_required / vc_inputs.investor_ownership_at_exit
        implied_pre_money = implied_post_money - vc_inputs.new_money
        investor_irr_actual = (investor_exit_value / vc_inputs.new_money) ** (1 / years_to_exit) - 1

        return {
            "exit_enterprise_value": exit_ev,
            "investor_exit_value": investor_exit_value,
            "investor_pv_required": investor_pv_required,
            "implied_post_money": implied_post_money,
            "implied_pre_money": implied_pre_money,
            "investor_irr_if_pay_new_money": investor_irr_actual,
        }


# ============================
# 5. Scenario & stress testing
# ============================


@dataclass
class Scenario:
    name: str
    revenue_multiplier: float = 1.0
    cost_multiplier: float = 1.0
    discount_rate_shift: float = 0.0
    success_prob_multiplier: float = 1.0


class ScenarioEngine:
    def __init__(self, base_portfolio: Portfolio):
        self.base_portfolio = base_portfolio
        self.base_model_config = base_portfolio.model_config

    def _apply_scenario(self, scenario: Scenario) -> Portfolio:
        new_model_cfg = ModelConfig(**asdict(self.base_model_config))
        new_model_cfg.discount_rate += scenario.discount_rate_shift

        new_products: List[Product] = []
        for prod in self.base_portfolio.products:
            cfg = prod.config
            cfg_dict = asdict(cfg)
            cfg_dict["patent_revenue_target"] *= scenario.revenue_multiplier
            cfg_dict["post_patent_revenue_target"] *= scenario.revenue_multiplier
            cfg_dict["cogs_patent"] *= scenario.cost_multiplier
            cfg_dict["cogs_post"] *= scenario.cost_multiplier
            cfg_dict["success_prob"] = max(
                0.0, min(1.0, cfg_dict["success_prob"] * scenario.success_prob_multiplier)
            )
            new_cfg = ProductConfig(**cfg_dict)
            new_products.append(Product(new_cfg, new_model_cfg))

        return Portfolio(new_products, new_model_cfg)

    def run_scenarios(self, scenarios: List[Scenario], ebitda_year_offset: int = 0) -> pd.DataFrame:
        rows = []

        base_val = ValuationEngine(self.base_portfolio).run()
        base_cons = base_val.consolidated
        base_year = self.base_model_config.first_year + ebitda_year_offset
        rows.append(
            {
                "scenario": "Base",
                "discount_rate": self.base_model_config.discount_rate,
                "rnpv": base_val.rnpv,
                "ebitda_year": base_year,
                "ebitda_value": float(base_cons.loc[base_year, "ebitda"]),
            }
        )

        for sc in scenarios:
            port_sc = self._apply_scenario(sc)
            val = ValuationEngine(port_sc).run()
            cons = val.consolidated
            year = port_sc.model_config.first_year + ebitda_year_offset
            rows.append(
                {
                    "scenario": sc.name,
                    "discount_rate": port_sc.model_config.discount_rate,
                    "rnpv": val.rnpv,
                    "ebitda_year": year,
                    "ebitda_value": float(cons.loc[year, "ebitda"]),
                }
            )

        return pd.DataFrame(rows)


# ==========================
# 6. Monte Carlo & risk
# ==========================


class MonteCarloEngine:
    def __init__(self, base_portfolio: Portfolio):
        self.base_portfolio = base_portfolio

    def simulate(
        self,
        n_sims: int = 1000,
        revenue_sigma: float = 0.1,
        cost_sigma: float = 0.05,
        random_seed: Optional[int] = None,
    ) -> pd.Series:
        rng = np.random.default_rng(random_seed)
        vals = []

        for _ in range(n_sims):
            model_cfg = self.base_portfolio.model_config
            new_products: List[Product] = []
            rev_scale = rng.normal(1.0, revenue_sigma)
            cogs_scale = rng.normal(1.0, cost_sigma)

            for prod in self.base_portfolio.products:
                cfg_dict = asdict(prod.config)
                cfg_dict["patent_revenue_target"] *= rev_scale
                cfg_dict["post_patent_revenue_target"] *= rev_scale
                cfg_dict["cogs_patent"] *= cogs_scale
                cfg_dict["cogs_post"] *= cogs_scale
                new_cfg = ProductConfig(**cfg_dict)
                new_products.append(Product(new_cfg, model_cfg))

            sim_portfolio = Portfolio(new_products, model_cfg)
            val = ValuationEngine(sim_portfolio).run()
            vals.append(val.rnpv)

        return pd.Series(vals, name="rnpv_sim")

    @staticmethod
    def value_at_risk(simulated_rnpv: pd.Series, alpha: float = 0.95) -> float:
        return float(simulated_rnpv.quantile(1 - alpha))

    @staticmethod
    def conditional_value_at_risk(simulated_rnpv: pd.Series, alpha: float = 0.95) -> float:
        var = simulated_rnpv.quantile(1 - alpha)
        tail = simulated_rnpv[simulated_rnpv <= var]
        if len(tail) == 0:
            return float(var)
        return float(tail.mean())


# ======================================
# 8. ForecastEngine: ARIMA / Prophet / LSTM
# ======================================


class ForecastEngine:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config

    def forecast_arima(
        self,
        series: pd.Series,
        order: tuple[int, int, int] = (1, 1, 1),
        steps: Optional[int] = None,
    ) -> pd.Series:
        from statsmodels.tsa.arima.model import ARIMA

        if steps is None:
            steps = self.model_config.n_years

        model = ARIMA(series, order=order)
        fitted = model.fit()
        forecast = fitted.forecast(steps=steps)
        forecast.name = f"{series.name}_arima_forecast"
        return forecast

    def forecast_prophet(
        self,
        df: pd.DataFrame,
        periods: Optional[int] = None,
        freq: str = "Y",
    ) -> pd.DataFrame:
        from prophet import Prophet

        if periods is None:
            periods = self.model_config.n_years

        m = Prophet()
        m.fit(df)
        future = m.make_future_dataframe(periods=periods, freq=freq)
        forecast = m.predict(future)
        return forecast

    def forecast_lstm(
        self,
        series: pd.Series,
        lookback: int = 12,
        steps_ahead: Optional[int] = None,
        epochs: int = 50,
        batch_size: int = 16,
    ) -> np.ndarray:
        import numpy as np
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense

        if steps_ahead is None:
            steps_ahead = self.model_config.n_years

        values = series.values.astype("float32")
        X, y = [], []
        for i in range(len(values) - lookback):
            X.append(values[i : i + lookback])
            y.append(values[i + lookback])
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = Sequential()
        model.add(LSTM(32, input_shape=(lookback, 1)))
        model.add(Dense(1))
        model.compile(loss="mse", optimizer="adam")
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

        history = values[-lookback:].copy()
        forecasts = []
        for _ in range(steps_ahead):
            x_input = history[-lookback:].reshape((1, lookback, 1))
            yhat = model.predict(x_input, verbose=0)[0, 0]
            forecasts.append(yhat)
            history = np.append(history, yhat)

        return np.array(forecasts)

    @staticmethod
    def _implied_cagr_from_forecast(base_value: float, forecast_values: np.ndarray) -> float:
        if base_value <= 0:
            return 0.0
        horizon = len(forecast_values)
        avg_forecast = float(np.mean(forecast_values))
        if avg_forecast <= 0:
            return 0.0
        return (avg_forecast / base_value) ** (1 / max(horizon, 1)) - 1

    def apply_price_forecast_to_products(
        self,
        products: List[Product],
        price_forecast: pd.Series,
        base_price: float,
        mode: str = "growth",
        growth_scale: float = 1.0,
        revenue_scale_max: float = 2.0,
    ) -> List[Product]:
        forecast_values = price_forecast.values
        out: List[Product] = []

        if mode == "growth":
            implied_cagr = self._implied_cagr_from_forecast(base_price, forecast_values)
            adj = growth_scale * implied_cagr

            for prod in products:
                cfg_dict = asdict(prod.config)
                cfg_dict["market_growth_patent"] += adj
                cfg_dict["market_growth_post"] += adj
                new_cfg = ProductConfig(**cfg_dict)
                out.append(Product(new_cfg, prod.model_config))

        elif mode == "revenue_scale":
            ratio = float(np.mean(forecast_values)) / base_price if base_price > 0 else 1.0
            ratio = min(max(ratio, 0.0), revenue_scale_max)

            for prod in products:
                cfg_dict = asdict(prod.config)
                cfg_dict["patent_revenue_target"] *= ratio
                cfg_dict["post_patent_revenue_target"] *= ratio
                new_cfg = ProductConfig(**cfg_dict)
                out.append(Product(new_cfg, prod.model_config))

        else:
            raise ValueError("mode must be 'growth' or 'revenue_scale'")

        return out


# ===========================================================
# 9. Forecast → Scenario bridge (Prophet-driven stress tests)
# ===========================================================


class ForecastScenarioBridge:
    def __init__(self, base_portfolio: Portfolio, forecast_engine: ForecastEngine):
        self.base_portfolio = base_portfolio
        self.forecast_engine = forecast_engine
        self.model_config = base_portfolio.model_config

    @staticmethod
    def _avg_ratio(series: pd.Series, base_value: float, cap: float = 3.0) -> float:
        if base_value <= 0:
            return 1.0
        ratio = float(series.mean() / base_value)
        return max(0.0, min(ratio, cap))

    def build_price_scenarios_from_prophet(
        self,
        hist_df: pd.DataFrame,
        periods: Optional[int] = None,
        freq: str = "Y",
        pessimistic_discount_uplift: float = 0.02,
        optimistic_discount_reduction: float = 0.01,
        cost_sensitivity: float = 0.05,
        ebitda_year_offset: int = 0,
    ) -> pd.DataFrame:
        forecast = self.forecast_engine.forecast_prophet(hist_df, periods=periods, freq=freq)
        last_hist_date = hist_df["ds"].max()
        future_fc = forecast[forecast["ds"] > last_hist_date].copy()
        n = self.model_config.n_years
        future_fc = future_fc.head(n)

        base_series = future_fc["yhat"]
        pess_series = future_fc["yhat_lower"]
        opt_series = future_fc["yhat_upper"]

        base_price = float(hist_df["y"].iloc[-1])
        base_ratio = self._avg_ratio(base_series, base_price)
        pess_ratio = self._avg_ratio(pess_series, base_price)
        opt_ratio = self._avg_ratio(opt_series, base_price)

        scenarios: List[Scenario] = []
        scenarios.append(
            Scenario(
                name="Prophet Base",
                revenue_multiplier=base_ratio,
                cost_multiplier=1.0,
                discount_rate_shift=0.0,
                success_prob_multiplier=1.0,
            )
        )
        scenarios.append(
            Scenario(
                name="Prophet Pessimistic",
                revenue_multiplier=pess_ratio,
                cost_multiplier=1.0 + cost_sensitivity,
                discount_rate_shift=pessimistic_discount_uplift,
                success_prob_multiplier=0.95,
            )
        )
        scenarios.append(
            Scenario(
                name="Prophet Optimistic",
                revenue_multiplier=opt_ratio,
                cost_multiplier=max(0.0, 1.0 - cost_sensitivity),
                discount_rate_shift=-optimistic_discount_reduction,
                success_prob_multiplier=1.05,
            )
        )

        scen_engine = ScenarioEngine(self.base_portfolio)
        scen_results = scen_engine.run_scenarios(
            scenarios=scenarios,
            ebitda_year_offset=ebitda_year_offset,
        )
        return scen_results


__all__ = [
    "ModelConfig",
    "ProductConfig",
    "Product",
    "Portfolio",
    "ValuationEngine",
    "ValuationResult",
    "VCInputs",
    "VCValuator",
    "Scenario",
    "ScenarioEngine",
    "MonteCarloEngine",
    "ForecastEngine",
    "ForecastScenarioBridge",
]


if __name__ == "__main__":
    model_cfg = ModelConfig()
    moonshine_cfg = ProductConfig(
        name="Vaccine_Moonshine",
        stage="Market",
        success_prob=1.0,
        include_in_consolidation=True,
        preexisting_market=True,
        time_to_market=-20,
        patent_years=20,
        patent_revenue_target=11_250_000.0,
        post_patent_revenue_target=6_300_000.0,
        market_growth_patent=0.005,
        market_growth_post=0.0,
        cogs_patent=0.30,
        cogs_post=0.50,
        sales_marketing_pct=0.15,
        gna_pct=0.10,
        rd_annual_post_launch=500_000.0,
        capex_annual_post_launch=100_000.0,
    )

    moonshine = Product(moonshine_cfg, model_cfg)
    portfolio = Portfolio([moonshine], model_cfg)

    engine = ValuationEngine(portfolio)
    val_res = engine.run()
    print("rNPV:", round(val_res.rnpv, 2), model_cfg.currency)

    vc_inputs = VCInputs(
        exit_year=model_cfg.first_year + 8,
        target_irr=0.40,
        investor_ownership_at_exit=0.25,
        new_money=20_000_000.0,
    )
    vc_val = VCValuator(val_res)
    vc_result = vc_val.vc_method(vc_inputs, exit_multiple=10.0)
    print("VC implied pre-money:", round(vc_result["implied_pre_money"], 2))

    drought = Scenario(
        name="Drought",
        revenue_multiplier=0.8,
        cost_multiplier=1.1,
        discount_rate_shift=0.02,
        success_prob_multiplier=0.9,
    )
    disease = Scenario(
        name="Disease",
        revenue_multiplier=0.7,
        cost_multiplier=1.15,
        discount_rate_shift=0.03,
        success_prob_multiplier=0.8,
    )

    scen_engine = ScenarioEngine(portfolio)
    scen_df = scen_engine.run_scenarios([drought, disease], ebitda_year_offset=5)
    print("\nScenario comparison:")
    print(scen_df)

    mc = MonteCarloEngine(portfolio)
    sims = mc.simulate(n_sims=200, revenue_sigma=0.15, cost_sigma=0.10, random_seed=42)
    print("\nMonte Carlo mean rNPV:", sims.mean())
    print("95% VaR:", MonteCarloEngine.value_at_risk(sims, alpha=0.95))
    print("95% CVaR:", MonteCarloEngine.conditional_value_at_risk(sims, alpha=0.95))

    idx = pd.period_range("2015", "2023", freq="Y").to_timestamp()
    hist_prices = pd.Series(
        [100, 102, 105, 108, 110, 115, 118, 120, 123],
        index=idx,
        name="price",
    )
    fe = ForecastEngine(model_cfg)
    price_fc = fe.forecast_arima(hist_prices, order=(1, 1, 1), steps=model_cfg.n_years)
    products_forecasted = fe.apply_price_forecast_to_products(
        portfolio.products,
        price_fc,
        base_price=hist_prices.iloc[-1],
        mode="growth",
        growth_scale=1.0,
    )
    port_fc = Portfolio(products_forecasted, model_cfg)
    val_engine_fc = ValuationEngine(port_fc)
    val_res_fc = val_engine_fc.run()
    print("rNPV with price-linked growth:", round(val_res_fc.rnpv, 2), model_cfg.currency)

    dates = pd.date_range("2015-01-01", periods=9, freq="Y")
    hist_prices_df = pd.DataFrame({"ds": dates, "y": [100, 102, 105, 108, 110, 115, 118, 120, 123]})
    bridge = ForecastScenarioBridge(portfolio, fe)
    scen_results = bridge.build_price_scenarios_from_prophet(hist_prices_df, freq="Y", ebitda_year_offset=5)
    print("\nProphet-driven scenario comparison:")
    print(scen_results)
