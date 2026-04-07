"""
TA-Lib Technical Analysis Microservice
FastAPI server providing professional-grade technical analysis via TA-Lib.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
import pandas as pd
import talib
import json

app = FastAPI(title="TA-Lib Technical Analysis Service", version="1.0.0")

# Allow requests from Next.js dev server
import os

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    *[o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()],
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Models ──────────────────────────────────────────────────

class OHLCVData(BaseModel):
    open: list[float]
    high: list[float]
    low: list[float]
    close: list[float]
    volume: list[float]
    times: Optional[list] = None  # timestamps (passthrough for alignment)


class IndicatorRequest(BaseModel):
    name: str
    params: dict = Field(default_factory=dict)


class IndicatorsRequest(BaseModel):
    ohlcv: OHLCVData
    indicators: list[IndicatorRequest]


class AnalyzeRequest(BaseModel):
    ohlcv: OHLCVData
    include_patterns: bool = True


class TrendlineRequest(BaseModel):
    ohlcv: OHLCVData
    dates: list[str]  # YYYY-MM-DD for each candle
    trend_type: str = "both"  # "support", "resistance", or "both"
    max_results_per_type: int = 2  # max trendlines per type (1 primary + 1 secondary)
    min_span_pct: float = 0.20  # trendline must span at least 20% of the data range


class SupplyDemandRequest(BaseModel):
    ohlcv: OHLCVData
    dates: list[str]
    max_zones: int = 6  # max total zones to return


class ScanRequest(BaseModel):
    ohlcv: OHLCVData


# ─── Helpers ────────────────────────────────────────────────────────────────────

def to_np(lst: list[float]) -> np.ndarray:
    """Convert list to numpy float64 array, replacing None with NaN."""
    return np.array(lst, dtype=np.float64)


def clean(arr) -> list:
    """Convert numpy array to JSON-safe list, replacing NaN with None."""
    if arr is None:
        return []
    return [None if np.isnan(v) else round(float(v), 4) for v in arr]


def last_valid(arr) -> Optional[float]:
    """Get the last non-NaN value from an array."""
    if arr is None or len(arr) == 0:
        return None
    for v in reversed(arr):
        if not np.isnan(v):
            return round(float(v), 4)
    return None


# Map of all supported single-output indicators and their required inputs
INDICATOR_MAP = {
    # Overlap Studies
    "SMA": {"func": talib.SMA, "inputs": ["close"], "default_params": {"timeperiod": 20}},
    "EMA": {"func": talib.EMA, "inputs": ["close"], "default_params": {"timeperiod": 20}},
    "WMA": {"func": talib.WMA, "inputs": ["close"], "default_params": {"timeperiod": 20}},
    "DEMA": {"func": talib.DEMA, "inputs": ["close"], "default_params": {"timeperiod": 20}},
    "TEMA": {"func": talib.TEMA, "inputs": ["close"], "default_params": {"timeperiod": 20}},
    "TRIMA": {"func": talib.TRIMA, "inputs": ["close"], "default_params": {"timeperiod": 20}},
    "KAMA": {"func": talib.KAMA, "inputs": ["close"], "default_params": {"timeperiod": 20}},
    "T3": {"func": talib.T3, "inputs": ["close"], "default_params": {"timeperiod": 5, "vfactor": 0.7}},
    "MIDPOINT": {"func": talib.MIDPOINT, "inputs": ["close"], "default_params": {"timeperiod": 14}},
    "HT_TRENDLINE": {"func": talib.HT_TRENDLINE, "inputs": ["close"], "default_params": {}},
    "SAR": {"func": talib.SAR, "inputs": ["high", "low"], "default_params": {"acceleration": 0.02, "maximum": 0.2}},

    # Momentum (single output)
    "RSI": {"func": talib.RSI, "inputs": ["close"], "default_params": {"timeperiod": 14}},
    "MOM": {"func": talib.MOM, "inputs": ["close"], "default_params": {"timeperiod": 10}},
    "ROC": {"func": talib.ROC, "inputs": ["close"], "default_params": {"timeperiod": 10}},
    "WILLR": {"func": talib.WILLR, "inputs": ["high", "low", "close"], "default_params": {"timeperiod": 14}},
    "CCI": {"func": talib.CCI, "inputs": ["high", "low", "close"], "default_params": {"timeperiod": 14}},
    "ADX": {"func": talib.ADX, "inputs": ["high", "low", "close"], "default_params": {"timeperiod": 14}},
    "ADXR": {"func": talib.ADXR, "inputs": ["high", "low", "close"], "default_params": {"timeperiod": 14}},
    "APO": {"func": talib.APO, "inputs": ["close"], "default_params": {"fastperiod": 12, "slowperiod": 26}},
    "CMO": {"func": talib.CMO, "inputs": ["close"], "default_params": {"timeperiod": 14}},
    "DX": {"func": talib.DX, "inputs": ["high", "low", "close"], "default_params": {"timeperiod": 14}},
    "MFI": {"func": talib.MFI, "inputs": ["high", "low", "close", "volume"], "default_params": {"timeperiod": 14}},
    "MINUS_DI": {"func": talib.MINUS_DI, "inputs": ["high", "low", "close"], "default_params": {"timeperiod": 14}},
    "PLUS_DI": {"func": talib.PLUS_DI, "inputs": ["high", "low", "close"], "default_params": {"timeperiod": 14}},
    "TRIX": {"func": talib.TRIX, "inputs": ["close"], "default_params": {"timeperiod": 30}},
    "ULTOSC": {"func": talib.ULTOSC, "inputs": ["high", "low", "close"], "default_params": {"timeperiod1": 7, "timeperiod2": 14, "timeperiod3": 28}},
    "BOP": {"func": talib.BOP, "inputs": ["open", "high", "low", "close"], "default_params": {}},
    "PPO": {"func": talib.PPO, "inputs": ["close"], "default_params": {"fastperiod": 12, "slowperiod": 26}},

    # Volume
    "AD": {"func": talib.AD, "inputs": ["high", "low", "close", "volume"], "default_params": {}},
    "OBV": {"func": talib.OBV, "inputs": ["close", "volume"], "default_params": {}},

    # Volatility
    "ATR": {"func": talib.ATR, "inputs": ["high", "low", "close"], "default_params": {"timeperiod": 14}},
    "NATR": {"func": talib.NATR, "inputs": ["high", "low", "close"], "default_params": {"timeperiod": 14}},
    "TRANGE": {"func": talib.TRANGE, "inputs": ["high", "low", "close"], "default_params": {}},

    # Price Transform
    "AVGPRICE": {"func": talib.AVGPRICE, "inputs": ["open", "high", "low", "close"], "default_params": {}},
    "MEDPRICE": {"func": talib.MEDPRICE, "inputs": ["high", "low"], "default_params": {}},
    "TYPPRICE": {"func": talib.TYPPRICE, "inputs": ["high", "low", "close"], "default_params": {}},
    "WCLPRICE": {"func": talib.WCLPRICE, "inputs": ["high", "low", "close"], "default_params": {}},

    # Cycle Indicators
    "HT_DCPERIOD": {"func": talib.HT_DCPERIOD, "inputs": ["close"], "default_params": {}},
    "HT_DCPHASE": {"func": talib.HT_DCPHASE, "inputs": ["close"], "default_params": {}},
    "HT_TRENDMODE": {"func": talib.HT_TRENDMODE, "inputs": ["close"], "default_params": {}},

    # Statistics
    "BETA": {"func": talib.BETA, "inputs": ["high", "low"], "default_params": {"timeperiod": 5}},
    "CORREL": {"func": talib.CORREL, "inputs": ["high", "low"], "default_params": {"timeperiod": 30}},
    "STDDEV": {"func": talib.STDDEV, "inputs": ["close"], "default_params": {"timeperiod": 5, "nbdev": 1}},
    "VAR": {"func": talib.VAR, "inputs": ["close"], "default_params": {"timeperiod": 5, "nbdev": 1}},
    "LINEARREG": {"func": talib.LINEARREG, "inputs": ["close"], "default_params": {"timeperiod": 14}},
    "LINEARREG_SLOPE": {"func": talib.LINEARREG_SLOPE, "inputs": ["close"], "default_params": {"timeperiod": 14}},
    "LINEARREG_ANGLE": {"func": talib.LINEARREG_ANGLE, "inputs": ["close"], "default_params": {"timeperiod": 14}},
    "TSF": {"func": talib.TSF, "inputs": ["close"], "default_params": {"timeperiod": 14}},
}

# Multi-output indicators
MULTI_OUTPUT_MAP = {
    "MACD": {
        "func": talib.MACD, "inputs": ["close"],
        "default_params": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
        "outputs": ["macd", "signal", "histogram"],
    },
    "MACDEXT": {
        "func": talib.MACDEXT, "inputs": ["close"],
        "default_params": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
        "outputs": ["macd", "signal", "histogram"],
    },
    "STOCH": {
        "func": talib.STOCH, "inputs": ["high", "low", "close"],
        "default_params": {"fastk_period": 5, "slowk_period": 3, "slowd_period": 3},
        "outputs": ["slowk", "slowd"],
    },
    "STOCHF": {
        "func": talib.STOCHF, "inputs": ["high", "low", "close"],
        "default_params": {"fastk_period": 5, "fastd_period": 3},
        "outputs": ["fastk", "fastd"],
    },
    "STOCHRSI": {
        "func": talib.STOCHRSI, "inputs": ["close"],
        "default_params": {"timeperiod": 14, "fastk_period": 5, "fastd_period": 3},
        "outputs": ["fastk", "fastd"],
    },
    "BBANDS": {
        "func": talib.BBANDS, "inputs": ["close"],
        "default_params": {"timeperiod": 20, "nbdevup": 2.0, "nbdevdn": 2.0},
        "outputs": ["upper", "middle", "lower"],
    },
    "AROON": {
        "func": talib.AROON, "inputs": ["high", "low"],
        "default_params": {"timeperiod": 14},
        "outputs": ["aroondown", "aroonup"],
    },
    "AROONOSC": {
        "func": talib.AROONOSC, "inputs": ["high", "low"],
        "default_params": {"timeperiod": 14},
        "outputs": None,  # single output
    },
    "MAMA": {
        "func": talib.MAMA, "inputs": ["close"],
        "default_params": {"fastlimit": 0.5, "slowlimit": 0.05},
        "outputs": ["mama", "fama"],
    },
    "HT_PHASOR": {
        "func": talib.HT_PHASOR, "inputs": ["close"],
        "default_params": {},
        "outputs": ["inphase", "quadrature"],
    },
    "HT_SINE": {
        "func": talib.HT_SINE, "inputs": ["close"],
        "default_params": {},
        "outputs": ["sine", "leadsine"],
    },
    "MIDPRICE": {
        "func": talib.MIDPRICE, "inputs": ["high", "low"],
        "default_params": {"timeperiod": 14},
        "outputs": None,
    },
    "ADOSC": {
        "func": talib.ADOSC, "inputs": ["high", "low", "close", "volume"],
        "default_params": {"fastperiod": 3, "slowperiod": 10},
        "outputs": None,
    },
}

# All 61 candlestick pattern functions
CANDLE_PATTERNS = {name: getattr(talib, name) for name in talib.get_function_groups()["Pattern Recognition"]}


def compute_indicator(name: str, params: dict, arrays: dict) -> dict:
    """Compute a single indicator and return results."""
    name_upper = name.upper()

    # Check single-output indicators first
    if name_upper in INDICATOR_MAP:
        cfg = INDICATOR_MAP[name_upper]
        merged = {**cfg["default_params"], **params}
        input_arrays = [arrays[k] for k in cfg["inputs"]]
        result = cfg["func"](*input_arrays, **merged)
        key = f"{name_upper}_{merged.get('timeperiod', '')}" if 'timeperiod' in merged else name_upper
        return {key: clean(result)}

    # Check multi-output indicators
    if name_upper in MULTI_OUTPUT_MAP:
        cfg = MULTI_OUTPUT_MAP[name_upper]
        merged = {**cfg["default_params"], **params}
        input_arrays = [arrays[k] for k in cfg["inputs"]]
        result = cfg["func"](*input_arrays, **merged)
        outputs = cfg["outputs"]
        if outputs and isinstance(result, tuple):
            key_base = name_upper
            if "timeperiod" in merged:
                key_base = f"{name_upper}_{merged['timeperiod']}"
            return {key_base: {k: clean(v) for k, v in zip(outputs, result)}}
        else:
            key = f"{name_upper}_{merged.get('timeperiod', '')}" if 'timeperiod' in merged else name_upper
            return {key: clean(result if not isinstance(result, tuple) else result[0])}

    raise ValueError(f"Unknown indicator: {name}")


# ─── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/api/indicators")
async def compute_indicators(req: IndicatorsRequest):
    """Compute one or more indicators on provided OHLCV data."""
    try:
        arrays = {
            "open": to_np(req.ohlcv.open),
            "high": to_np(req.ohlcv.high),
            "low": to_np(req.ohlcv.low),
            "close": to_np(req.ohlcv.close),
            "volume": to_np(req.ohlcv.volume),
        }

        results = {}
        for ind in req.indicators:
            try:
                result = compute_indicator(ind.name, ind.params, arrays)
                results.update(result)
            except Exception as e:
                results[ind.name] = {"error": str(e)}

        return {"results": results, "length": len(req.ohlcv.close)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze")
async def full_analysis(req: AnalyzeRequest):
    """
    Comprehensive technical analysis: computes all major indicators
    and optionally detects candlestick patterns.
    """
    try:
        o = to_np(req.ohlcv.open)
        h = to_np(req.ohlcv.high)
        l = to_np(req.ohlcv.low)
        c = to_np(req.ohlcv.close)
        v = to_np(req.ohlcv.volume)

        current_price = float(c[-1]) if len(c) > 0 else 0

        # ── Trend Indicators ────────────────────────────────
        sma_20 = talib.SMA(c, timeperiod=20)
        sma_50 = talib.SMA(c, timeperiod=50)
        sma_200 = talib.SMA(c, timeperiod=200)
        ema_9 = talib.EMA(c, timeperiod=9)
        ema_21 = talib.EMA(c, timeperiod=21)
        ema_50 = talib.EMA(c, timeperiod=50)

        # ── Momentum ────────────────────────────────────────
        rsi = talib.RSI(c, timeperiod=14)
        macd, macd_signal, macd_hist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
        stoch_k, stoch_d = talib.STOCH(h, l, c, fastk_period=14, slowk_period=3, slowd_period=3)
        adx = talib.ADX(h, l, c, timeperiod=14)
        cci = talib.CCI(h, l, c, timeperiod=14)
        willr = talib.WILLR(h, l, c, timeperiod=14)
        mfi = talib.MFI(h, l, c, v, timeperiod=14)
        plus_di = talib.PLUS_DI(h, l, c, timeperiod=14)
        minus_di = talib.MINUS_DI(h, l, c, timeperiod=14)
        mom = talib.MOM(c, timeperiod=10)
        roc = talib.ROC(c, timeperiod=10)

        # ── Volatility ──────────────────────────────────────
        atr = talib.ATR(h, l, c, timeperiod=14)
        natr = talib.NATR(h, l, c, timeperiod=14)
        bb_upper, bb_middle, bb_lower = talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2)

        # ── Volume ──────────────────────────────────────────
        obv = talib.OBV(c, v)
        ad = talib.AD(h, l, c, v)
        adosc = talib.ADOSC(h, l, c, v, fastperiod=3, slowperiod=10)

        # ── Support / Resistance (pivot-based) ──────────────
        recent_highs = h[-20:] if len(h) >= 20 else h
        recent_lows = l[-20:] if len(l) >= 20 else l
        pivot_high = float(np.nanmax(recent_highs))
        pivot_low = float(np.nanmin(recent_lows))
        pivot = (pivot_high + pivot_low + current_price) / 3
        r1 = 2 * pivot - pivot_low
        s1 = 2 * pivot - pivot_high
        r2 = pivot + (pivot_high - pivot_low)
        s2 = pivot - (pivot_high - pivot_low)

        # ── Trend Determination ─────────────────────────────
        rsi_val = last_valid(rsi)
        adx_val = last_valid(adx)
        macd_val = last_valid(macd)
        macd_sig = last_valid(macd_signal)
        macd_hist_val = last_valid(macd_hist)
        sma20_val = last_valid(sma_20)
        sma50_val = last_valid(sma_50)
        sma200_val = last_valid(sma_200)

        def rsi_signal(v):
            if v is None: return "UNKNOWN"
            if v > 70: return "OVERBOUGHT"
            if v > 60: return "BULLISH"
            if v < 30: return "OVERSOLD"
            if v < 40: return "BEARISH"
            return "NEUTRAL"

        def macd_signal_str(m, s):
            if m is None or s is None: return "UNKNOWN"
            if m > s and m > 0: return "STRONG_BULLISH"
            if m > s: return "BULLISH"
            if m < s and m < 0: return "STRONG_BEARISH"
            if m < s: return "BEARISH"
            return "NEUTRAL"

        def trend_str(price, sma20, sma50, sma200):
            bullish = 0
            if sma20 and price > sma20: bullish += 1
            if sma50 and price > sma50: bullish += 1
            if sma200 and price > sma200: bullish += 1
            if bullish >= 3: return "STRONG_UPTREND"
            if bullish == 2: return "UPTREND"
            if bullish == 1: return "SIDEWAYS"
            return "DOWNTREND"

        # ── Candlestick Patterns ────────────────────────────
        patterns_found = []
        if req.include_patterns:
            for pattern_name, pattern_func in CANDLE_PATTERNS.items():
                try:
                    result = pattern_func(o, h, l, c)
                    # Check last 5 candles for patterns
                    for i in range(-1, max(-6, -len(result)), -1):
                        val = int(result[i])
                        if val != 0:
                            idx = len(result) + i
                            nice_name = pattern_name.replace("CDL", "").replace("_", " ").title()
                            patterns_found.append({
                                "name": nice_name,
                                "function": pattern_name,
                                "index": idx,
                                "time": req.ohlcv.times[idx] if req.ohlcv.times and idx < len(req.ohlcv.times) else None,
                                "signal": "BULLISH" if val > 0 else "BEARISH",
                                "strength": abs(val),  # 100 = single pattern, 200 = double confirmation
                            })
                except Exception:
                    pass

        # ── Volume Analysis ─────────────────────────────────
        vol_arr = v[-20:] if len(v) >= 20 else v
        avg_volume = float(np.nanmean(vol_arr))
        latest_volume = float(v[-1]) if len(v) > 0 else 0
        vol_ratio = round(latest_volume / avg_volume, 2) if avg_volume > 0 else 0

        # ── Build Response ──────────────────────────────────
        return {
            "symbol": "COMPUTED",
            "currentPrice": round(current_price, 2),
            "indicators": {
                "SMA_20": last_valid(sma_20),
                "SMA_50": last_valid(sma_50),
                "SMA_200": last_valid(sma_200),
                "EMA_9": last_valid(ema_9),
                "EMA_21": last_valid(ema_21),
                "EMA_50": last_valid(ema_50),
                "RSI_14": rsi_val,
                "MACD": {
                    "line": macd_val,
                    "signal": macd_sig,
                    "histogram": macd_hist_val,
                },
                "Stochastic": {
                    "K": last_valid(stoch_k),
                    "D": last_valid(stoch_d),
                },
                "ADX_14": adx_val,
                "CCI_14": last_valid(cci),
                "Williams_R": last_valid(willr),
                "MFI_14": last_valid(mfi),
                "Plus_DI": last_valid(plus_di),
                "Minus_DI": last_valid(minus_di),
                "Momentum": last_valid(mom),
                "ROC": last_valid(roc),
                "bollingerBands": {
                    "upper": last_valid(bb_upper),
                    "middle": last_valid(bb_middle),
                    "lower": last_valid(bb_lower),
                },
                "ATR_14": last_valid(atr),
                "NATR_14": last_valid(natr),
                "OBV": last_valid(obv),
            },
            "levels": {
                "resistance_R2": round(r2, 2),
                "resistance_R1": round(r1, 2),
                "pivot": round(pivot, 2),
                "support_S1": round(s1, 2),
                "support_S2": round(s2, 2),
                "recent_high": round(pivot_high, 2),
                "recent_low": round(pivot_low, 2),
                "SMA50": sma50_val,
                "SMA200": sma200_val,
            },
            "volume": {
                "latest": latest_volume,
                "avg20": round(avg_volume),
                "ratio": vol_ratio,
                "trend": "HIGH" if vol_ratio > 1.5 else "LOW" if vol_ratio < 0.5 else "NORMAL",
            },
            "trend": {
                "direction": trend_str(current_price, sma20_val, sma50_val, sma200_val),
                "aboveSMA20": current_price > (sma20_val or 0),
                "aboveSMA50": current_price > (sma50_val or 0),
                "aboveSMA200": current_price > (sma200_val or 0),
                "rsiSignal": rsi_signal(rsi_val),
                "macdSignal": macd_signal_str(macd_val, macd_sig),
                "adxStrength": "STRONG" if (adx_val or 0) > 25 else "WEAK",
                "stochasticSignal": (
                    "OVERBOUGHT" if (last_valid(stoch_k) or 50) > 80
                    else "OVERSOLD" if (last_valid(stoch_k) or 50) < 20
                    else "NEUTRAL"
                ),
            },
            "patterns": patterns_found,
            "overlays": {
                "SMA_20": clean(sma_20),
                "SMA_50": clean(sma_50),
                "EMA_9": clean(ema_9),
                "EMA_21": clean(ema_21),
                "BB_upper": clean(bb_upper),
                "BB_middle": clean(bb_middle),
                "BB_lower": clean(bb_lower),
            },
            "subcharts": {
                "RSI": clean(rsi),
                "MACD": {"macd": clean(macd), "signal": clean(macd_signal), "histogram": clean(macd_hist)},
                "Stochastic": {"K": clean(stoch_k), "D": clean(stoch_d)},
                "ADX": clean(adx),
                "ATR": clean(atr),
                "OBV": clean(obv),
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/scan")
async def scan_patterns(req: ScanRequest):
    """Scan for all 61 candlestick patterns in the provided OHLCV data."""
    try:
        o = to_np(req.ohlcv.open)
        h = to_np(req.ohlcv.high)
        l = to_np(req.ohlcv.low)
        c = to_np(req.ohlcv.close)

        patterns_found = []
        for pattern_name, pattern_func in CANDLE_PATTERNS.items():
            try:
                result = pattern_func(o, h, l, c)
                for i in range(len(result)):
                    val = int(result[i])
                    if val != 0:
                        nice_name = pattern_name.replace("CDL", "").replace("_", " ").title()
                        patterns_found.append({
                            "name": nice_name,
                            "function": pattern_name,
                            "index": i,
                            "time": req.ohlcv.times[i] if req.ohlcv.times and i < len(req.ohlcv.times) else None,
                            "signal": "BULLISH" if val > 0 else "BEARISH",
                            "strength": abs(val),
                        })
            except Exception:
                pass

        # Sort by index descending (most recent first)
        patterns_found.sort(key=lambda p: p["index"], reverse=True)

        return {"patterns": patterns_found, "total": len(patterns_found)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/available")
async def list_available():
    """List all available indicators and candlestick patterns."""
    single = []
    for name, cfg in INDICATOR_MAP.items():
        single.append({
            "name": name,
            "inputs": cfg["inputs"],
            "defaultParams": cfg["default_params"],
            "type": "single",
        })

    multi = []
    for name, cfg in MULTI_OUTPUT_MAP.items():
        multi.append({
            "name": name,
            "inputs": cfg["inputs"],
            "defaultParams": cfg["default_params"],
            "outputs": cfg["outputs"],
            "type": "multi",
        })

    patterns = [
        {"name": name.replace("CDL", "").replace("_", " ").title(), "function": name}
        for name in CANDLE_PATTERNS.keys()
    ]

    return {
        "indicators": single + multi,
        "patterns": patterns,
        "totalIndicators": len(single) + len(multi),
        "totalPatterns": len(patterns),
    }


def _find_pivots(prices: np.ndarray, order: int = 5):
    """
    Find pivot highs and pivot lows.
    A pivot high at index i means prices[i] is the max in [i-order, i+order].
    A pivot low at index i means prices[i] is the min in [i-order, i+order].
    """
    n = len(prices)
    pivot_highs = []
    pivot_lows = []
    for i in range(order, n - order):
        window = prices[i - order: i + order + 1]
        if prices[i] == np.max(window):
            pivot_highs.append(i)
        if prices[i] == np.min(window):
            pivot_lows.append(i)
    return pivot_lows, pivot_highs


def _fit_lines_through_pivots(
    prices: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    pivot_indices: list[int],
    n: int,
    line_type: str,  # 'support' or 'resistance'
    max_lines: int = 2,
    min_span_pct: float = 0.20,
):
    """
    For each pair of pivots, fit a line and validate:
    - Support: line must stay at or below the lows (no candle low pierces below)
    - Resistance: line must stay at or above the highs (no candle high pierces above)
    - Count how many pivots the line touches (within ATR tolerance)
    - Line must span at least min_span_pct of total data
    """
    min_span = int(n * min_span_pct)
    atr_arr = talib.ATR(highs, lows, prices, timeperiod=14)
    avg_atr = float(np.nanmean(atr_arr[~np.isnan(atr_arr)])) if np.any(~np.isnan(atr_arr)) else 1.0
    tolerance = avg_atr * 0.15  # how close a pivot must be to "touch" the line
    violation_tolerance = avg_atr * 0.05  # small allowance for minor wicks

    candidates = []

    for i in range(len(pivot_indices)):
        for j in range(i + 1, len(pivot_indices)):
            idx_a = pivot_indices[i]
            idx_b = pivot_indices[j]

            if idx_b - idx_a < min_span:
                continue

            # Use actual prices at pivots (lows for support, highs for resistance)
            if line_type == 'support':
                price_a = float(lows[idx_a])
                price_b = float(lows[idx_b])
            else:
                price_a = float(highs[idx_a])
                price_b = float(highs[idx_b])

            slope = (price_b - price_a) / (idx_b - idx_a)
            intercept = price_a - slope * idx_a

            # Validate: check that the line doesn't cut through candles
            violations = 0
            touches = 0
            for k in range(idx_a, min(idx_b + 1, n)):
                line_price = slope * k + intercept
                if line_type == 'support':
                    # Support line should be at or below the lows
                    if lows[k] < line_price - violation_tolerance:
                        violations += 1
                    if abs(lows[k] - line_price) < tolerance:
                        touches += 1
                else:
                    # Resistance line should be at or above the highs
                    if highs[k] > line_price + violation_tolerance:
                        violations += 1
                    if abs(highs[k] - line_price) < tolerance:
                        touches += 1

            span = idx_b - idx_a
            max_violations = max(2, int(span * 0.08))  # allow up to 8% violations

            if violations > max_violations:
                continue
            if touches < 2:
                continue

            # Score: more touches + longer span + fewer violations = better
            score = (touches ** 2) * (span / n) * (1.0 / (1 + violations))

            candidates.append({
                'idx_a': idx_a,
                'idx_b': idx_b,
                'price_a': price_a,
                'price_b': price_b,
                'slope': slope,
                'intercept': intercept,
                'touches': touches,
                'violations': violations,
                'span': span,
                'score': score,
            })

    # Sort by score descending
    candidates.sort(key=lambda c: c['score'], reverse=True)

    # Deduplicate: skip lines too similar to already-selected ones
    selected = []
    for c in candidates:
        if len(selected) >= max_lines:
            break
        is_dup = False
        for s in selected:
            # Similar if slope is close AND they overlap significantly
            slope_diff = abs(c['slope'] - s['slope'])
            price_diff = abs(c['price_a'] - s['price_a'])
            if slope_diff < avg_atr * 0.001 and price_diff < avg_atr * 0.5:
                is_dup = True
                break
        if not is_dup:
            selected.append(c)

    return selected


@app.post("/api/trendlines")
async def detect_trendlines_endpoint(req: TrendlineRequest):
    """
    Detect support/resistance trendlines using pivot detection + linear regression.
    Draws lines through actual swing highs/lows, validated to not cut through candles.
    """
    try:
        h = to_np(req.ohlcv.high)
        l = to_np(req.ohlcv.low)
        c = to_np(req.ohlcv.close)
        n = len(c)

        if n < 20:
            return {"trendlines": [], "total": 0}

        # Adaptive pivot order: larger window for more data
        order = max(3, n // 25)
        pivot_lows, pivot_highs = _find_pivots(c, order=order)

        trendlines = []
        trend_type = req.trend_type.lower()

        if trend_type in ('both', 'support') and len(pivot_lows) >= 2:
            support_lines = _fit_lines_through_pivots(
                c, h, l, pivot_lows, n,
                line_type='support',
                max_lines=req.max_results_per_type,
                min_span_pct=req.min_span_pct,
            )
            for line in support_lines:
                trendlines.append({
                    "type": "support",
                    "startIndex": line['idx_a'],
                    "startDate": req.dates[line['idx_a']],
                    "startPrice": round(line['price_a'], 2),
                    "endIndex": line['idx_b'],
                    "endDate": req.dates[line['idx_b']],
                    "endPrice": round(line['price_b'], 2),
                    "numPoints": line['touches'],
                    "slope": round(line['slope'], 6),
                    "score": round(line['score'], 4),
                    "violations": line['violations'],
                })

        if trend_type in ('both', 'resistance') and len(pivot_highs) >= 2:
            resistance_lines = _fit_lines_through_pivots(
                c, h, l, pivot_highs, n,
                line_type='resistance',
                max_lines=req.max_results_per_type,
                min_span_pct=req.min_span_pct,
            )
            for line in resistance_lines:
                trendlines.append({
                    "type": "resistance",
                    "startIndex": line['idx_a'],
                    "startDate": req.dates[line['idx_a']],
                    "startPrice": round(line['price_a'], 2),
                    "endIndex": line['idx_b'],
                    "endDate": req.dates[line['idx_b']],
                    "endPrice": round(line['price_b'], 2),
                    "numPoints": line['touches'],
                    "slope": round(line['slope'], 6),
                    "score": round(line['score'], 4),
                    "violations": line['violations'],
                })

        # Sort by score
        trendlines.sort(key=lambda t: t["score"], reverse=True)

        return {"trendlines": trendlines, "total": len(trendlines)}

    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


@app.post("/api/supply-demand")
async def detect_supply_demand(req: SupplyDemandRequest):
    """
    Detect supply and demand zones from OHLCV data.

    Algorithm:
    1. Find swing highs/lows using ATR-based pivot detection
    2. At each swing point, look for the explosive move (big candle) that started the rally/drop
    3. The body of that base candle defines the zone (open-close range)
    4. Score zones by: strength of the move away, number of retests, whether zone is fresh
    """
    try:
        o = to_np(req.ohlcv.open)
        h = to_np(req.ohlcv.high)
        l = to_np(req.ohlcv.low)
        c = to_np(req.ohlcv.close)
        n = len(c)

        if n < 20:
            return {"zones": [], "total": 0}

        # ATR for adaptive thresholds
        atr_arr = talib.ATR(h, l, c, timeperiod=14)
        avg_atr = float(np.nanmean(atr_arr[-50:])) if n >= 50 else float(np.nanmean(atr_arr[~np.isnan(atr_arr)]))

        # Find swing highs and lows (local extrema within lookback window)
        lookback = 10
        swing_highs = []
        swing_lows = []

        for i in range(lookback, n - lookback):
            # Swing high: highest high in window
            window_h = h[i - lookback:i + lookback + 1]
            if h[i] == np.max(window_h):
                swing_highs.append(i)
            # Swing low: lowest low in window
            window_l = l[i - lookback:i + lookback + 1]
            if l[i] == np.min(window_l):
                swing_lows.append(i)

        zones = []

        # SUPPLY ZONES: form at swing highs where price dropped sharply
        for idx in swing_highs:
            # Find the base candle — the candle at or just before the swing high
            # that initiated the move up (look for the biggest body in the 3 candles before)
            best_base = idx
            best_body = 0
            for j in range(max(0, idx - 3), idx + 1):
                body = abs(c[j] - o[j])
                if body > best_body:
                    best_body = body
                    best_base = j

            # Zone boundaries: the body range of the base candle, extended slightly
            zone_high = float(max(o[best_base], c[best_base]))
            zone_low = float(min(o[best_base], c[best_base]))
            # Extend zone slightly with wicks
            zone_high = max(zone_high, float(h[best_base]))
            zone_low = min(zone_low, zone_high - max(best_body, avg_atr * 0.5))

            if zone_high - zone_low < avg_atr * 0.2:
                continue  # zone too thin

            # Score: how far price dropped from this zone
            drop_after = float(h[idx]) - float(np.min(l[idx:min(idx + 30, n)]))
            if drop_after < avg_atr * 1.0:
                continue  # weak move away, not a real supply zone

            # Check freshness: has price come back to this zone?
            retests = 0
            broken = False
            for k in range(idx + 1, n):
                if h[k] > zone_high + avg_atr * 0.5:
                    broken = True
                    break
                if l[k] <= zone_high and h[k] >= zone_low:
                    retests += 1

            score = drop_after / avg_atr * (1 if not broken else 0.3) * (1 + retests * 0.2)

            # Zone extends from formation to end of chart (or until broken)
            end_idx = n - 1
            zones.append({
                "type": "supply",
                "priceHigh": round(zone_high, 2),
                "priceLow": round(zone_low, 2),
                "startDate": req.dates[best_base],
                "endDate": req.dates[end_idx],
                "formationIndex": int(idx),
                "formationDate": req.dates[idx],
                "score": round(score, 2),
                "retests": retests,
                "broken": broken,
                "moveAway": round(drop_after, 2),
            })

        # DEMAND ZONES: form at swing lows where price rallied sharply
        for idx in swing_lows:
            best_base = idx
            best_body = 0
            for j in range(max(0, idx - 3), idx + 1):
                body = abs(c[j] - o[j])
                if body > best_body:
                    best_body = body
                    best_base = j

            zone_high = float(max(o[best_base], c[best_base]))
            zone_low = float(min(o[best_base], c[best_base]))
            zone_low = min(zone_low, float(l[best_base]))
            zone_high = max(zone_high, zone_low + max(best_body, avg_atr * 0.5))

            if zone_high - zone_low < avg_atr * 0.2:
                continue

            rally_after = float(np.max(h[idx:min(idx + 30, n)])) - float(l[idx])
            if rally_after < avg_atr * 1.0:
                continue

            retests = 0
            broken = False
            for k in range(idx + 1, n):
                if l[k] < zone_low - avg_atr * 0.5:
                    broken = True
                    break
                if l[k] <= zone_high and h[k] >= zone_low:
                    retests += 1

            score = rally_after / avg_atr * (1 if not broken else 0.3) * (1 + retests * 0.2)

            end_idx = n - 1
            zones.append({
                "type": "demand",
                "priceHigh": round(zone_high, 2),
                "priceLow": round(zone_low, 2),
                "startDate": req.dates[best_base],
                "endDate": req.dates[end_idx],
                "formationIndex": int(idx),
                "formationDate": req.dates[idx],
                "score": round(score, 2),
                "retests": retests,
                "broken": broken,
                "moveAway": round(rally_after, 2),
            })

        # Sort by score, prefer unbroken zones, limit results
        zones.sort(key=lambda z: (not z["broken"], z["score"]), reverse=True)
        zones = zones[:req.max_zones]

        return {"zones": zones, "total": len(zones)}

    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


@app.get("/")
async def root():
    return {"status": "ok", "service": "finance-talib-service"}


@app.get("/health")
async def health():
    return {"status": "ok", "talib_version": talib.__ta_version__.decode() if hasattr(talib, '__ta_version__') else "unknown"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
