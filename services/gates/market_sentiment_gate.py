from __future__ import annotations
"""
market_sentiment_gate.py
------------------------
Comprehensive market sentiment analysis that provides setup biases based on:
  • Nifty/BankNifty momentum alignment
  • Sector strength rankings
  • Market breadth indicators (advance/decline, new highs/lows)
  • Fear/Greed sentiment (VIX-like indicators)
  • FII/DII flow sentiment

This module enhances setup selection by applying market context filters
and biases to improve win rates in different market conditions.

Public API
----------
class MarketSentimentGate:
    def analyze_sentiment(self, nifty_df5, banknifty_df5, breadth_data=None) -> SentimentReading
    def get_setup_bias(self, setup_type: str, sentiment: SentimentReading) -> float
    def should_trade_setup(self, setup_type: str, sentiment: SentimentReading) -> bool
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np

class SentimentLevel(Enum):
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"

class MarketTrend(Enum):
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"

@dataclass(frozen=True)
class SentimentReading:
    """Complete market sentiment snapshot"""
    # Index momentum
    nifty_momentum: float
    banknifty_momentum: float
    combined_momentum: float
    market_trend: MarketTrend
    
    # Breadth indicators
    advance_decline_ratio: float
    new_highs_lows_ratio: float
    
    # Fear/Greed indicators
    sentiment_level: SentimentLevel
    fear_greed_score: float  # 0-100 scale
    
    # Sector data
    sector_strengths: Dict[str, float]
    top_sectors: List[str]
    
    # Combined signals
    bullish_bias: float
    bearish_bias: float
    mean_reversion_bias: float
    momentum_bias: float

class MarketSentimentGate:
    """
    Analyzes market sentiment and provides setup biases for better trade selection
    """
    
    def __init__(self, cfg: Dict, log=None):
        self.log = log
        self.cfg = cfg.get("market_sentiment", {})
        self.enabled = self.cfg.get("enabled", True)
        
        # Nifty momentum thresholds
        nifty_cfg = self.cfg.get("nifty_momentum", {})
        self.nifty_bull_threshold = nifty_cfg.get("bullish_threshold", 0.3)
        self.nifty_bear_threshold = nifty_cfg.get("bearish_threshold", -0.3)
        self.nifty_lookback = nifty_cfg.get("lookback_bars", 10)
        self.nifty_weight = nifty_cfg.get("weight_factor", 1.5)
        
        # BankNifty momentum thresholds  
        bank_cfg = self.cfg.get("banknifty_momentum", {})
        self.bank_bull_threshold = bank_cfg.get("bullish_threshold", 0.4)
        self.bank_bear_threshold = bank_cfg.get("bearish_threshold", -0.4)
        self.bank_lookback = bank_cfg.get("lookback_bars", 10)
        self.bank_weight = bank_cfg.get("weight_factor", 1.3)
        
        # Breadth indicators
        breadth_cfg = self.cfg.get("breadth_indicators", {})
        self.ad_bull_ratio = breadth_cfg.get("advance_decline_ratio_bullish", 1.2)
        self.ad_bear_ratio = breadth_cfg.get("advance_decline_ratio_bearish", 0.8)
        self.hl_bull_ratio = breadth_cfg.get("new_highs_lows_ratio_bullish", 1.5)
        self.hl_bear_ratio = breadth_cfg.get("new_highs_lows_ratio_bearish", 0.7)
        
        # VIX/Fear-Greed
        vix_cfg = self.cfg.get("vix_sentiment", {})
        self.fear_threshold = vix_cfg.get("fear_threshold", 20.0)
        self.greed_threshold = vix_cfg.get("greed_threshold", 12.0)
        self.extreme_fear_threshold = vix_cfg.get("extreme_fear_threshold", 30.0)
        self.extreme_greed_threshold = vix_cfg.get("extreme_greed_threshold", 10.0)
        
        # Sentiment biases
        bias_cfg = self.cfg.get("sentiment_bias", {})
        self.bullish_bias_mult = bias_cfg.get("bullish_bias_multiplier", 1.2)
        self.bearish_bias_mult = bias_cfg.get("bearish_bias_multiplier", 1.2)
        self.contrarian_mode = bias_cfg.get("contrarian_mode_enabled", True)
        self.extreme_fade = bias_cfg.get("extreme_sentiment_fade", True)

    def analyze_sentiment(self, 
                         nifty_df5: pd.DataFrame, 
                         banknifty_df5: Optional[pd.DataFrame] = None,
                         breadth_data: Optional[Dict] = None,
                         vix_level: Optional[float] = None) -> SentimentReading:
        """
        Comprehensive sentiment analysis from multiple data sources
        """
        if not self.enabled:
            return self._neutral_sentiment()
            
        # Index momentum analysis
        nifty_momentum = self._calculate_momentum(nifty_df5, self.nifty_lookback)
        banknifty_momentum = 0.0
        if banknifty_df5 is not None and not banknifty_df5.empty:
            banknifty_momentum = self._calculate_momentum(banknifty_df5, self.bank_lookback)
        
        # Combined momentum (weighted)
        combined_momentum = (nifty_momentum * self.nifty_weight + 
                           banknifty_momentum * self.bank_weight) / (self.nifty_weight + self.bank_weight)
        
        market_trend = self._classify_trend(combined_momentum)
        
        # Breadth indicators (mock implementation for now)
        ad_ratio = breadth_data.get("advance_decline_ratio", 1.0) if breadth_data else 1.0
        hl_ratio = breadth_data.get("new_highs_lows_ratio", 1.0) if breadth_data else 1.0
        
        # Fear/Greed sentiment
        fear_greed_score = self._calculate_fear_greed(vix_level, combined_momentum)
        sentiment_level = self._classify_sentiment(fear_greed_score)
        
        # Sector strengths (mock for now - would need sector data)
        sector_strengths = self._calculate_sector_strengths(breadth_data)
        top_sectors = sorted(sector_strengths.keys(), key=sector_strengths.get, reverse=True)[:5]
        
        # Calculate biases
        bullish_bias, bearish_bias = self._calculate_directional_biases(
            combined_momentum, sentiment_level, ad_ratio, hl_ratio
        )
        mean_reversion_bias = self._calculate_mean_reversion_bias(sentiment_level, combined_momentum)
        momentum_bias = self._calculate_momentum_bias(combined_momentum, sentiment_level)
        
        return SentimentReading(
            nifty_momentum=nifty_momentum,
            banknifty_momentum=banknifty_momentum,
            combined_momentum=combined_momentum,
            market_trend=market_trend,
            advance_decline_ratio=ad_ratio,
            new_highs_lows_ratio=hl_ratio,
            sentiment_level=sentiment_level,
            fear_greed_score=fear_greed_score,
            sector_strengths=sector_strengths,
            top_sectors=top_sectors,
            bullish_bias=bullish_bias,
            bearish_bias=bearish_bias,
            mean_reversion_bias=mean_reversion_bias,
            momentum_bias=momentum_bias
        )
    
    def get_setup_bias(self, setup_type: str, sentiment: SentimentReading) -> float:
        """
        SIMPLIFIED (Dec 2024): Always returns 1.0 (no bias).

        Setup-specific sentiment bias has been moved to pipeline config.
        This method is kept for backwards compatibility with trade_decision_gate.py.

        Pipelines can access sentiment via analyze_sentiment() and apply their own
        setup-specific logic in their config (gates.sentiment_rules.<setup>).

        Returns:
            1.0 - neutral bias (no adjustment)
        """
        return 1.0

    def should_trade_setup(self, setup_type: str, sentiment: SentimentReading) -> bool:
        """
        SIMPLIFIED (Dec 2024): Always returns True (never blocks).

        Setup-specific sentiment blocking has been moved to pipeline config.
        This method is kept for backwards compatibility with trade_decision_gate.py.

        Pipelines can implement their own sentiment-based blocking using:
        - gates.blocked_sentiment_levels: [extreme_fear, extreme_greed]
        - Or custom logic in validate_gates()

        Returns:
            True - never blocks based on sentiment
        """
        return True
    
    def _calculate_momentum(self, df: pd.DataFrame, lookback: int) -> float:
        """Calculate momentum score from -1 to +1"""
        if df is None or df.empty or len(df) < lookback:
            return 0.0
            
        closes = df['close'].tail(lookback).astype(float)
        if len(closes) < 2:
            return 0.0
            
        # Simple momentum: (current - lookback_start) / lookback_start
        momentum = (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0]
        return np.clip(momentum * 10, -1.0, 1.0)  # Scale and clip
    
    def _classify_trend(self, momentum: float) -> MarketTrend:
        """Classify market trend based on combined momentum"""
        if momentum > 0.5:
            return MarketTrend.STRONG_BULLISH
        elif momentum > 0.2:
            return MarketTrend.BULLISH
        elif momentum < -0.5:
            return MarketTrend.STRONG_BEARISH
        elif momentum < -0.2:
            return MarketTrend.BEARISH
        else:
            return MarketTrend.NEUTRAL
    
    def _calculate_fear_greed(self, vix_level: Optional[float], momentum: float) -> float:
        """Calculate fear/greed score 0-100 (0=extreme fear, 100=extreme greed)"""
        if vix_level is not None:
            # VIX-based calculation
            if vix_level >= self.extreme_fear_threshold:
                return 10.0
            elif vix_level >= self.fear_threshold:
                return 25.0
            elif vix_level <= self.extreme_greed_threshold:
                return 90.0
            elif vix_level <= self.greed_threshold:
                return 75.0
            else:
                return 50.0
        else:
            # Momentum-based approximation
            return 50.0 + (momentum * 30.0)  # -1 to +1 momentum -> 20 to 80 score
    
    def _classify_sentiment(self, score: float) -> SentimentLevel:
        """Classify sentiment level from fear/greed score"""
        if score <= 20:
            return SentimentLevel.EXTREME_FEAR
        elif score <= 35:
            return SentimentLevel.FEAR
        elif score >= 80:
            return SentimentLevel.EXTREME_GREED
        elif score >= 65:
            return SentimentLevel.GREED
        else:
            return SentimentLevel.NEUTRAL
    
    def _calculate_sector_strengths(self, breadth_data: Optional[Dict]) -> Dict[str, float]:
        """Calculate relative sector strengths (mock implementation)"""
        # In real implementation, this would analyze sector indices
        default_sectors = {
            "BANK": 1.0, "IT": 1.0, "PHARMA": 1.0, "FMCG": 1.0, 
            "AUTO": 1.0, "METAL": 1.0, "ENERGY": 1.0, "REALTY": 1.0
        }
        
        if breadth_data and "sector_data" in breadth_data:
            return breadth_data["sector_data"]
        
        return default_sectors
    
    def _calculate_directional_biases(self, momentum: float, sentiment: SentimentLevel, 
                                    ad_ratio: float, hl_ratio: float) -> Tuple[float, float]:
        """Calculate bullish and bearish biases"""
        base_bull = 1.0
        base_bear = 1.0
        
        # Momentum bias
        if momentum > 0.3:
            base_bull *= 1.2
            base_bear *= 0.8
        elif momentum < -0.3:
            base_bull *= 0.8
            base_bear *= 1.2
            
        # Breadth bias
        if ad_ratio > self.ad_bull_ratio and hl_ratio > self.hl_bull_ratio:
            base_bull *= 1.15
        elif ad_ratio < self.ad_bear_ratio and hl_ratio < self.hl_bear_ratio:
            base_bear *= 1.15
            
        # Contrarian bias for extreme sentiment
        if self.contrarian_mode and sentiment in [SentimentLevel.EXTREME_FEAR, SentimentLevel.EXTREME_GREED]:
            if sentiment == SentimentLevel.EXTREME_FEAR:
                base_bull *= 1.3  # Buy fear
                base_bear *= 0.7
            else:  # EXTREME_GREED
                base_bull *= 0.7
                base_bear *= 1.3  # Sell greed
                
        return base_bull, base_bear
    
    def _calculate_mean_reversion_bias(self, sentiment: SentimentLevel, momentum: float) -> float:
        """Calculate bias for mean reversion setups"""
        bias = 1.0
        
        # Mean reversion works better in extreme sentiment
        if sentiment in [SentimentLevel.EXTREME_FEAR, SentimentLevel.EXTREME_GREED]:
            bias *= 1.4
        elif sentiment in [SentimentLevel.FEAR, SentimentLevel.GREED]:
            bias *= 1.2
            
        # Less effective in strong trends
        if abs(momentum) > 0.5:
            bias *= 0.8
            
        return bias
    
    def _calculate_momentum_bias(self, momentum: float, sentiment: SentimentLevel) -> float:
        """Calculate bias for momentum/breakout setups"""
        bias = 1.0
        
        # Momentum works better in trending markets
        if abs(momentum) > 0.3:
            bias *= 1.2
        elif abs(momentum) < 0.1:
            bias *= 0.8
            
        # Less effective in extreme sentiment (reversals likely)
        if sentiment in [SentimentLevel.EXTREME_FEAR, SentimentLevel.EXTREME_GREED]:
            bias *= 0.7
            
        return bias
    
    def _neutral_sentiment(self) -> SentimentReading:
        """Return neutral sentiment when disabled"""
        return SentimentReading(
            nifty_momentum=0.0,
            banknifty_momentum=0.0,
            combined_momentum=0.0,
            market_trend=MarketTrend.NEUTRAL,
            advance_decline_ratio=1.0,
            new_highs_lows_ratio=1.0,
            sentiment_level=SentimentLevel.NEUTRAL,
            fear_greed_score=50.0,
            sector_strengths={},
            top_sectors=[],
            bullish_bias=1.0,
            bearish_bias=1.0,
            mean_reversion_bias=1.0,
            momentum_bias=1.0
        )