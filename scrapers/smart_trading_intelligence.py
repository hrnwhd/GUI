# ===== SMART TRADING INTELLIGENCE SYSTEM - PREFERENCES APPROACH =====
# Provides intelligent guidance and preferences instead of hard blocks
# Generates actionable insights for optimized trading decisions

import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import schedule
import threading
import time

# ===== CONFIGURATION =====
CONFIG = {
    'input_files': {
        'sentiment': 'sentiment_signals.json',
        'correlation': 'correlation_data.json',
        'cot': 'cot_consolidated_data.json',
        'calendar': 'test_calendar_output.json'
    },
    'output_file': 'trading_intelligence.json',
    'log_file': 'smart_intelligence.log',
    'update_interval_minutes': 15,
    'risk_thresholds': {
        'high_correlation': 70,
        'extreme_cot': 25,
        'high_impact_hours': 24
    }
}

# Trading pairs to monitor
TRADING_PAIRS = [
    'AUDUSD', 'USDCAD', 'XAUUSD', 'EURUSD', 'GBPUSD',
    'AUDCAD', 'USDCHF', 'GBPCAD', 'AUDNZD', 'NZDCAD', 
    'US500', 'BTCUSD'
]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - SMART_INTEL - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG['log_file'], encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SmartTradingIntelligence:
    """Generates intelligent trading preferences and guidance"""
    
    def __init__(self):
        self.pair_mappings = {
            'GOLD': 'XAUUSD',
            'SPX500': 'US500',
            'BITCOIN': 'BTCUSD',
            'AUD': 'AUDUSD',
            'EUR': 'EURUSD',
            'GBP': 'GBPUSD',
            'CAD': 'USDCAD',
            'CHF': 'USDCHF'
        }
        
    def generate_intelligence(self) -> Dict[str, Any]:
        """Generate smart trading intelligence with preferences approach"""
        try:
            logger.info("🧠 Generating smart trading intelligence...")
            
            # Load all raw data
            raw_data = self._load_raw_data()
            
            # Create intelligence structure with preferences focus
            intelligence = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'version': '2.0',
                    'approach': 'PREFERENCES_BASED',
                    'valid_until': (datetime.now() + timedelta(minutes=30)).isoformat(),
                    'data_quality': self._assess_data_quality(raw_data)
                },
                'trading_preferences': {},  # Main preferences for each pair
                'market_intelligence': {},  # Overall market insights
                'correlation_awareness': {},# Correlation insights
                'event_calendar': {},       # Event timing intelligence
                'risk_insights': []         # Risk awareness (not blocks)
            }
            
            # Process each trading pair for preferences
            for pair in TRADING_PAIRS:
                pair_preferences = self._analyze_pair_preferences(pair, raw_data)
                if pair_preferences:
                    intelligence['trading_preferences'][pair] = pair_preferences
            
            # Generate market intelligence
            intelligence['market_intelligence'] = self._generate_market_intelligence(raw_data)
            
            # Generate correlation awareness
            intelligence['correlation_awareness'] = self._generate_correlation_awareness(raw_data)
            
            # Generate event calendar intelligence
            intelligence['event_calendar'] = self._generate_event_intelligence(raw_data)
            
            # Generate risk insights
            intelligence['risk_insights'] = self._generate_risk_insights(raw_data, intelligence['trading_preferences'])
            
            logger.info(f"✅ Generated preferences for {len(intelligence['trading_preferences'])} pairs")
            return intelligence
            
        except Exception as e:
            logger.error(f"❌ Error generating intelligence: {e}")
            return self._create_safe_fallback()
    
    def _analyze_pair_preferences(self, pair: str, raw_data: Dict) -> Dict[str, Any]:
        """Analyze trading preferences for a specific pair"""
        try:
            preferences = {
                'pair': pair,
                'timestamp': datetime.now().isoformat(),
                
                # 🎯 DIRECTION PREFERENCES
                'preferred_direction': 'NEUTRAL',         # LONG/SHORT/NEUTRAL
                'preference_strength': 'WEAK',            # STRONG/MODERATE/WEAK
                'alternative_direction': 'ACCEPTABLE',    # ACCEPTABLE/CAUTION/AVOID
                'directional_confidence': 0.5,           # 0.0 to 1.0
                
                # 📊 POSITION SIZING INTELLIGENCE
                'position_size_modifier': 1.0,           # 0.5 to 1.5x normal size
                'max_exposure_pct': 100,                 # Maximum exposure percentage
                'risk_adjustment': 'NORMAL',             # AGGRESSIVE/NORMAL/CONSERVATIVE
                'size_adjustment_reason': [],            # Why size should be adjusted
                
                # 🔗 CORRELATION AWARENESS
                'correlation_risk_level': 'LOW',         # LOW/MEDIUM/HIGH
                'correlated_pairs': [],                  # List of correlated instruments
                'diversification_score': 100,           # 0-100% (higher = better)
                'correlation_impact': 'MINIMAL',        # MINIMAL/MODERATE/SIGNIFICANT
                
                # ⏰ TIMING INTELLIGENCE
                'event_risk_hours': 0,                  # Hours until high-impact events
                'optimal_entry_window': 'ANYTIME',      # ANYTIME/BEFORE_EVENTS/AFTER_EVENTS
                'hold_duration_guidance': 'NORMAL',     # SHORT/NORMAL/EXTENDED
                'timing_factors': [],                   # Factors affecting timing
                
                # 🏛️ COT INSIGHTS
                'cot_sentiment': 'NEUTRAL',             # BULLISH/BEARISH/NEUTRAL/EXTREME
                'contrarian_signal': False,             # True if extreme positioning
                'institutional_bias': 'NEUTRAL',        # What smart money is doing
                'cot_strength': 0.0,                    # Strength of COT signal
                
                # 📈 OVERALL ASSESSMENT
                'overall_opportunity': 'NORMAL',        # EXCELLENT/GOOD/NORMAL/POOR
                'key_factors': [],                      # Main factors driving assessment
                'trading_notes': '',                    # Human-readable summary
                'last_updated': datetime.now().isoformat()
            }
            
            # Analyze each component
            self._analyze_sentiment_preferences(pair, raw_data.get('sentiment'), preferences)
            self._analyze_correlation_impact(pair, raw_data.get('correlation'), preferences)
            self._analyze_event_timing(pair, raw_data.get('calendar'), preferences)
            self._analyze_cot_insights(pair, raw_data.get('cot'), preferences)
            
            # Calculate final preferences
            self._calculate_final_preferences(preferences)
            
            return preferences
            
        except Exception as e:
            logger.error(f"❌ Error analyzing preferences for {pair}: {e}")
            return self._create_safe_pair_preferences(pair)
    
    def _analyze_sentiment_preferences(self, pair: str, sentiment_data: Dict, preferences: Dict):
        """Analyze sentiment for directional preferences"""
        try:
            if not sentiment_data or 'pairs' not in sentiment_data:
                return
            
            pair_sentiment = sentiment_data['pairs'].get(pair)
            if not pair_sentiment:
                return
            
            # Get sentiment scores
            sentiment_scores = pair_sentiment.get('sentiment', {})
            short_pct = sentiment_scores.get('short', 50)
            long_pct = sentiment_scores.get('long', 50)
            
            # Calculate directional preference
            sentiment_bias = long_pct - short_pct  # Positive = bullish, negative = bearish
            
            if sentiment_bias > 30:
                preferences['preferred_direction'] = 'LONG'
                preferences['preference_strength'] = 'STRONG' if sentiment_bias > 50 else 'MODERATE'
                preferences['alternative_direction'] = 'CAUTION' if sentiment_bias > 40 else 'ACCEPTABLE'
                preferences['directional_confidence'] = min(0.9, 0.5 + abs(sentiment_bias) / 100)
            elif sentiment_bias < -30:
                preferences['preferred_direction'] = 'SHORT'
                preferences['preference_strength'] = 'STRONG' if sentiment_bias < -50 else 'MODERATE'
                preferences['alternative_direction'] = 'CAUTION' if sentiment_bias < -40 else 'ACCEPTABLE'
                preferences['directional_confidence'] = min(0.9, 0.5 + abs(sentiment_bias) / 100)
            else:
                preferences['preferred_direction'] = 'NEUTRAL'
                preferences['preference_strength'] = 'WEAK'
                preferences['alternative_direction'] = 'ACCEPTABLE'
                preferences['directional_confidence'] = 0.5
            
            # Check for blocked directions (convert to preferences)
            blocked = pair_sentiment.get('blocked_directions', [])
            if 'long' in blocked:
                if preferences['preferred_direction'] == 'LONG':
                    preferences['preferred_direction'] = 'NEUTRAL'
                preferences['alternative_direction'] = 'AVOID' if preferences['preferred_direction'] != 'SHORT' else preferences['alternative_direction']
            if 'short' in blocked:
                if preferences['preferred_direction'] == 'SHORT':
                    preferences['preferred_direction'] = 'NEUTRAL'
                preferences['alternative_direction'] = 'AVOID' if preferences['preferred_direction'] != 'LONG' else preferences['alternative_direction']
            
            preferences['key_factors'].append(f"Sentiment: {long_pct}% Long, {short_pct}% Short")
            
            logger.debug(f"{pair} sentiment preference: {preferences['preferred_direction']} ({preferences['preference_strength']})")
            
        except Exception as e:
            logger.error(f"❌ Error analyzing sentiment for {pair}: {e}")
    
    def _analyze_correlation_impact(self, pair: str, correlation_data: Dict, preferences: Dict):
        """Analyze correlation impact on position sizing and diversification"""
        try:
            if not correlation_data:
                return
            
            matrix = correlation_data.get('correlation_matrix', {})
            normalized_pair = self._normalize_pair_name(pair)
            
            # Find correlations for this pair
            pair_correlations = {}
            for row_symbol, correlations in matrix.items():
                if self._normalize_pair_name(row_symbol) == normalized_pair:
                    pair_correlations = correlations
                    break
            
            # Analyze correlation strength and impact
            high_correlations = []
            correlation_risk_score = 0
            
            for symbol, data in pair_correlations.items():
                correlation_value = data.get('value', 0)
                abs_corr = abs(correlation_value)
                
                if abs_corr >= 70:
                    high_correlations.append({
                        'pair': symbol,
                        'correlation': correlation_value,
                        'type': 'POSITIVE' if correlation_value > 0 else 'NEGATIVE'
                    })
                    correlation_risk_score += abs_corr
            
            # Set correlation risk level
            if len(high_correlations) >= 4:
                preferences['correlation_risk_level'] = 'HIGH'
                preferences['correlation_impact'] = 'SIGNIFICANT'
                preferences['position_size_modifier'] = 0.7
                preferences['diversification_score'] = 30
            elif len(high_correlations) >= 2:
                preferences['correlation_risk_level'] = 'MEDIUM'
                preferences['correlation_impact'] = 'MODERATE'
                preferences['position_size_modifier'] = 0.85
                preferences['diversification_score'] = 60
            else:
                preferences['correlation_risk_level'] = 'LOW'
                preferences['correlation_impact'] = 'MINIMAL'
                preferences['position_size_modifier'] = 1.0
                preferences['diversification_score'] = 90
            
            preferences['correlated_pairs'] = high_correlations[:5]  # Top 5
            
            if high_correlations:
                preferences['size_adjustment_reason'].append(f"High correlation with {len(high_correlations)} pairs")
                preferences['key_factors'].append(f"Correlation risk: {preferences['correlation_risk_level']}")
            
            logger.debug(f"{pair} correlation: {len(high_correlations)} high correlations, risk = {preferences['correlation_risk_level']}")
            
        except Exception as e:
            logger.error(f"❌ Error analyzing correlation for {pair}: {e}")
    
    def _analyze_event_timing(self, pair: str, calendar_data: Dict, preferences: Dict):
        """Analyze event timing for optimal entry/exit windows"""
        try:
            if not calendar_data or 'events' not in calendar_data:
                return
            
            events = calendar_data['events']
            relevant_events = self._filter_events_for_pair(pair, events)
            
            # Analyze upcoming events
            now = datetime.now()
            high_impact_soon = []
            medium_impact_soon = []
            
            for event in relevant_events:
                impact = event.get('impact', '').lower()
                if impact == 'high':
                    high_impact_soon.append(event)
                elif impact == 'medium':
                    medium_impact_soon.append(event)
            
            # Calculate event risk timing
            total_events = len(high_impact_soon) + len(medium_impact_soon)
            high_impact_count = len(high_impact_soon)
            
            if high_impact_count >= 3:
                preferences['event_risk_hours'] = 24
                preferences['optimal_entry_window'] = 'AFTER_EVENTS'
                preferences['hold_duration_guidance'] = 'SHORT'
                preferences['position_size_modifier'] *= 0.8
                preferences['risk_adjustment'] = 'CONSERVATIVE'
                preferences['timing_factors'].append(f"{high_impact_count} high-impact events in 24h")
            elif high_impact_count >= 1:
                preferences['event_risk_hours'] = 12
                preferences['optimal_entry_window'] = 'BEFORE_EVENTS'
                preferences['hold_duration_guidance'] = 'SHORT'
                preferences['timing_factors'].append(f"{high_impact_count} high-impact event today")
            elif total_events >= 3:
                preferences['event_risk_hours'] = 6
                preferences['optimal_entry_window'] = 'ANYTIME'
                preferences['hold_duration_guidance'] = 'NORMAL'
                preferences['timing_factors'].append(f"{total_events} medium-impact events")
            else:
                preferences['event_risk_hours'] = 0
                preferences['optimal_entry_window'] = 'ANYTIME'
                preferences['hold_duration_guidance'] = 'NORMAL'
            
            if total_events > 0:
                preferences['key_factors'].append(f"Events: {high_impact_count} high, {len(medium_impact_soon)} medium impact")
            
            logger.debug(f"{pair} events: {high_impact_count} high impact, entry window = {preferences['optimal_entry_window']}")
            
        except Exception as e:
            logger.error(f"❌ Error analyzing events for {pair}: {e}")
    
    def _analyze_cot_insights(self, pair: str, cot_data: Dict, preferences: Dict):
        """Analyze COT data for institutional bias and contrarian signals"""
        try:
            if not cot_data or 'data' not in cot_data:
                return
            
            # Map pair to COT instrument
            cot_mapping = {
                'AUDUSD': 'AUD', 'EURUSD': 'EUR', 'GBPUSD': 'GBP',
                'USDCAD': 'CAD', 'USDCHF': 'CHF', 'XAUUSD': 'Gold'
            }
            
            cot_instrument = cot_mapping.get(pair)
            if not cot_instrument:
                return
            
            # Find latest COT data
            cot_records = []
            for data_type in ['Financial', 'Commodity']:
                if data_type in cot_data['data']:
                    for record in cot_data['data'][data_type]:
                        if record.get('Name') == cot_instrument:
                            cot_records.append(record)
            
            if not cot_records:
                return
            
            # Get latest record
            latest_record = max(cot_records, key=lambda x: x.get('Date', ''))
            net_pct = latest_record.get('Non_Commercial_Net_Pct', 0)
            
            # Analyze COT positioning
            abs_net = abs(net_pct)
            preferences['cot_strength'] = min(1.0, abs_net / 50)  # Normalize to 0-1
            
            if net_pct > 40:
                preferences['cot_sentiment'] = 'EXTREME'
                preferences['institutional_bias'] = 'BULLISH'
                preferences['contrarian_signal'] = True
                preferences['key_factors'].append(f"COT: Extreme bullish positioning ({net_pct:.1f}%)")
                # Extreme positioning suggests potential reversal
                if preferences['preferred_direction'] == 'LONG':
                    preferences['preference_strength'] = 'WEAK'
                    preferences['alternative_direction'] = 'CAUTION'
            elif net_pct < -40:
                preferences['cot_sentiment'] = 'EXTREME'
                preferences['institutional_bias'] = 'BEARISH'
                preferences['contrarian_signal'] = True
                preferences['key_factors'].append(f"COT: Extreme bearish positioning ({net_pct:.1f}%)")
                # Extreme positioning suggests potential reversal
                if preferences['preferred_direction'] == 'SHORT':
                    preferences['preference_strength'] = 'WEAK'
                    preferences['alternative_direction'] = 'CAUTION'
            elif net_pct > 15:
                preferences['cot_sentiment'] = 'BULLISH'
                preferences['institutional_bias'] = 'BULLISH'
                preferences['key_factors'].append(f"COT: Bullish positioning ({net_pct:.1f}%)")
            elif net_pct < -15:
                preferences['cot_sentiment'] = 'BEARISH'
                preferences['institutional_bias'] = 'BEARISH'
                preferences['key_factors'].append(f"COT: Bearish positioning ({net_pct:.1f}%)")
            else:
                preferences['cot_sentiment'] = 'NEUTRAL'
                preferences['institutional_bias'] = 'NEUTRAL'
            
            logger.debug(f"{pair} COT: {net_pct:.1f}% → {preferences['cot_sentiment']}, contrarian: {preferences['contrarian_signal']}")
            
        except Exception as e:
            logger.error(f"❌ Error analyzing COT for {pair}: {e}")
    
    def _calculate_final_preferences(self, preferences: Dict):
        """Calculate final trading preferences and opportunity assessment"""
        try:
            # Calculate overall opportunity score
            opportunity_score = 0
            scoring_factors = []
            
            # Factor 1: Directional confidence
            conf_score = preferences['directional_confidence'] * 25
            opportunity_score += conf_score
            scoring_factors.append(f"Direction confidence: {conf_score:.1f}/25")
            
            # Factor 2: Preference strength
            strength_map = {'STRONG': 25, 'MODERATE': 15, 'WEAK': 5}
            strength_score = strength_map.get(preferences['preference_strength'], 5)
            opportunity_score += strength_score
            scoring_factors.append(f"Preference strength: {strength_score}/25")
            
            # Factor 3: Diversification score (correlation)
            div_score = preferences['diversification_score'] * 0.2  # Max 20 points
            opportunity_score += div_score
            scoring_factors.append(f"Diversification: {div_score:.1f}/20")
            
            # Factor 4: Event timing
            timing_map = {
                'ANYTIME': 20, 'BEFORE_EVENTS': 15, 'AFTER_EVENTS': 10
            }
            timing_score = timing_map.get(preferences['optimal_entry_window'], 10)
            opportunity_score += timing_score
            scoring_factors.append(f"Entry timing: {timing_score}/20")
            
            # Factor 5: COT alignment (bonus/penalty)
            if preferences['contrarian_signal']:
                cot_score = -10  # Penalty for extreme positioning
            elif preferences['cot_sentiment'] in ['BULLISH', 'BEARISH']:
                cot_score = 10   # Bonus for clear institutional bias
            else:
                cot_score = 0
            opportunity_score += cot_score
            if cot_score != 0:
                scoring_factors.append(f"COT factor: {cot_score}/10")
            
            # Determine opportunity level
            if opportunity_score >= 80:
                preferences['overall_opportunity'] = 'EXCELLENT'
            elif opportunity_score >= 65:
                preferences['overall_opportunity'] = 'GOOD'
            elif opportunity_score >= 40:
                preferences['overall_opportunity'] = 'NORMAL'
            else:
                preferences['overall_opportunity'] = 'POOR'
            
            # Adjust position size based on opportunity
            opportunity_multipliers = {
                'EXCELLENT': 1.2, 'GOOD': 1.1, 'NORMAL': 1.0, 'POOR': 0.8
            }
            preferences['position_size_modifier'] *= opportunity_multipliers[preferences['overall_opportunity']]
            
            # Cap position size modifier
            preferences['position_size_modifier'] = max(0.5, min(1.5, preferences['position_size_modifier']))
            
            # Set max exposure based on risk factors
            if preferences['correlation_risk_level'] == 'HIGH':
                preferences['max_exposure_pct'] = 60
            elif preferences['correlation_risk_level'] == 'MEDIUM':
                preferences['max_exposure_pct'] = 80
            else:
                preferences['max_exposure_pct'] = 100
            
            # Generate trading notes
            notes = []
            notes.append(f"Overall opportunity: {preferences['overall_opportunity']} (score: {opportunity_score:.1f}/100)")
            notes.append(f"Preferred direction: {preferences['preferred_direction']} ({preferences['preference_strength']})")
            notes.append(f"Position size: {preferences['position_size_modifier']:.2f}x normal")
            
            if preferences['contrarian_signal']:
                notes.append("⚠️ Contrarian signal: Extreme COT positioning suggests potential reversal")
            
            if preferences['correlation_risk_level'] != 'LOW':
                notes.append(f"📊 {preferences['correlation_risk_level']} correlation risk - consider diversification")
            
            if preferences['event_risk_hours'] > 0:
                notes.append(f"⏰ {preferences['event_risk_hours']}h until high-impact events")
            
            preferences['trading_notes'] = " | ".join(notes)
            
            logger.debug(f"Final opportunity: {preferences['overall_opportunity']} (score: {opportunity_score:.1f})")
            
        except Exception as e:
            logger.error(f"❌ Error calculating final preferences: {e}")
    
    def _filter_events_for_pair(self, pair: str, events: List[Dict]) -> List[Dict]:
        """Filter events relevant to a trading pair"""
        currency_map = {
            'AUDUSD': ['AUD', 'USD'], 'EURUSD': ['EUR', 'USD'], 'GBPUSD': ['GBP', 'USD'],
            'USDCAD': ['USD', 'CAD'], 'USDCHF': ['USD', 'CHF'], 'AUDCAD': ['AUD', 'CAD'],
            'GBPCAD': ['GBP', 'CAD'], 'AUDNZD': ['AUD', 'NZD'], 'NZDCAD': ['NZD', 'CAD'],
            'XAUUSD': ['USD'], 'US500': ['USD'], 'BTCUSD': ['USD']
        }
        
        target_currencies = currency_map.get(pair, [])
        return [event for event in events if event.get('currency', '') in target_currencies]
    
    def _normalize_pair_name(self, pair: str) -> str:
        """Normalize pair name"""
        return self.pair_mappings.get(pair.upper(), pair.upper())
    
    def _generate_market_intelligence(self, raw_data: Dict) -> Dict:
        """Generate overall market intelligence"""
        try:
            intelligence = {
                'timestamp': datetime.now().isoformat(),
                'market_regime': 'NORMAL',                 # TRENDING/RANGING/VOLATILE/NORMAL
                'volatility_environment': 'NORMAL',       # HIGH/ELEVATED/NORMAL/LOW
                'directional_bias': 'NEUTRAL',            # RISK_ON/RISK_OFF/NEUTRAL
                'correlation_environment': 'NORMAL',      # HIGH/ELEVATED/NORMAL/LOW
                'event_density': 'NORMAL',                # HIGH/ELEVATED/NORMAL/LOW
                'recommended_approach': 'BALANCED',       # AGGRESSIVE/BALANCED/DEFENSIVE
                'key_themes': [],                         # Main market themes
                'overall_confidence': 0.5                 # Confidence in market assessment
            }
            
            # Analyze sentiment patterns
            if raw_data.get('sentiment'):
                pairs = raw_data['sentiment'].get('pairs', {})
                strong_signals = 0
                total_pairs = len(pairs)
                risk_on_count = 0
                risk_off_count = 0
                
                for pair, data in pairs.items():
                    signal = data.get('signal_strength', '')
                    if 'Strong' in signal:
                        strong_signals += 1
                    
                    # Check for risk-on/risk-off patterns
                    if pair in ['AUDUSD', 'NZDUSD', 'EURUSD'] and 'Long' in signal:
                        risk_on_count += 1
                    elif pair in ['USDCAD', 'USDCHF', 'USDJPY'] and 'Long' in signal:
                        risk_off_count += 1
                
                if total_pairs > 0:
                    strong_ratio = strong_signals / total_pairs
                    if strong_ratio > 0.6:
                        intelligence['market_regime'] = 'TRENDING'
                        intelligence['volatility_environment'] = 'HIGH'
                    elif strong_ratio > 0.3:
                        intelligence['market_regime'] = 'VOLATILE'
                    
                    # Determine directional bias
                    if risk_on_count > risk_off_count + 1:
                        intelligence['directional_bias'] = 'RISK_ON'
                        intelligence['key_themes'].append('Risk-on sentiment dominant')
                    elif risk_off_count > risk_on_count + 1:
                        intelligence['directional_bias'] = 'RISK_OFF'
                        intelligence['key_themes'].append('Risk-off sentiment dominant')
            
            # Analyze correlation environment
            if raw_data.get('correlation'):
                warnings = raw_data['correlation'].get('warnings', [])
                if len(warnings) > 20:
                    intelligence['correlation_environment'] = 'HIGH'
                    intelligence['key_themes'].append('High correlation environment - reduced diversification')
                elif len(warnings) > 10:
                    intelligence['correlation_environment'] = 'ELEVATED'
            
            # Analyze event density
            if raw_data.get('calendar'):
                events = raw_data['calendar'].get('events', [])
                high_impact = [e for e in events if e.get('impact', '').lower() == 'high']
                if len(high_impact) > 8:
                    intelligence['event_density'] = 'HIGH'
                    intelligence['key_themes'].append(f'{len(high_impact)} high-impact events this week')
                elif len(high_impact) > 4:
                    intelligence['event_density'] = 'ELEVATED'
            
            # Determine recommended approach
            risk_factors = 0
            if intelligence['volatility_environment'] in ['HIGH', 'ELEVATED']:
                risk_factors += 1
            if intelligence['correlation_environment'] in ['HIGH', 'ELEVATED']:
                risk_factors += 1
            if intelligence['event_density'] in ['HIGH', 'ELEVATED']:
                risk_factors += 1
            
            if risk_factors >= 2:
                intelligence['recommended_approach'] = 'DEFENSIVE'
            elif risk_factors == 1:
                intelligence['recommended_approach'] = 'BALANCED'
            elif intelligence['market_regime'] == 'TRENDING':
                intelligence['recommended_approach'] = 'AGGRESSIVE'
            
            # Calculate overall confidence
            confidence_factors = []
            if raw_data.get('sentiment'):
                confidence_factors.append(0.8)
            if raw_data.get('correlation'):
                confidence_factors.append(0.7)
            if raw_data.get('calendar'):
                confidence_factors.append(0.6)
            if raw_data.get('cot'):
                confidence_factors.append(0.9)
            
            intelligence['overall_confidence'] = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.3
            
            return intelligence
            
        except Exception as e:
            logger.error(f"❌ Error generating market intelligence: {e}")
            return {'error': str(e)}
    
    def _generate_correlation_awareness(self, raw_data: Dict) -> Dict:
        """Generate correlation awareness insights"""
        try:
            awareness = {
                'timestamp': datetime.now().isoformat(),
                'correlation_clusters': [],               # Groups of highly correlated pairs
                'diversification_opportunities': [],      # Pairs with low correlation
                'risk_concentrations': [],               # Areas of high correlation risk
                'correlation_strength': 'NORMAL',        # Overall correlation environment
                'recommendations': []                     # Specific recommendations
            }
            
            if not raw_data.get('correlation'):
                return awareness
            
            matrix = raw_data['correlation'].get('correlation_matrix', {})
            warnings = raw_data['correlation'].get('warnings', [])
            
            # Identify correlation clusters
            clusters = {}
            for warning in warnings:
                if warning.get('type') == 'HIGH_CORRELATION':
                    pair = warning.get('pair', '')
                    correlation = warning.get('value', 0)
                    if abs(correlation) >= 75:
                        cluster_key = f"cluster_{len(clusters)}"
                        if cluster_key not in clusters:
                            clusters[cluster_key] = []
                        clusters[cluster_key].append({
                            'pair': pair,
                            'correlation': correlation
                        })
            
            awareness['correlation_clusters'] = list(clusters.values())
            
            # Find diversification opportunities (pairs with low correlation)
            diversification_pairs = []
            for row_symbol, correlations in matrix.items():
                low_corr_count = 0
                for symbol, data in correlations.items():
                    if abs(data.get('value', 0)) < 30:
                        low_corr_count += 1
                
                if low_corr_count >= 3:  # Has multiple low correlations
                    diversification_pairs.append({
                        'pair': row_symbol,
                        'low_correlation_count': low_corr_count
                    })
            
            awareness['diversification_opportunities'] = diversification_pairs[:5]
            
            # Identify risk concentrations
            high_risk_pairs = []
            for warning in warnings:
                if abs(warning.get('value', 0)) >= 85:
                    high_risk_pairs.append(warning)
            
            awareness['risk_concentrations'] = high_risk_pairs[:10]
            
            # Determine overall correlation strength
            if len(warnings) > 25:
                awareness['correlation_strength'] = 'VERY_HIGH'
            elif len(warnings) > 15:
                awareness['correlation_strength'] = 'HIGH'
            elif len(warnings) > 8:
                awareness['correlation_strength'] = 'ELEVATED'
            else:
                awareness['correlation_strength'] = 'NORMAL'
            
            # Generate recommendations
            if awareness['correlation_strength'] in ['HIGH', 'VERY_HIGH']:
                awareness['recommendations'].append("Reduce overall position sizes due to high correlation environment")
                awareness['recommendations'].append("Focus on pairs with low correlation to improve diversification")
            
            if len(awareness['risk_concentrations']) > 5:
                awareness['recommendations'].append("Avoid simultaneous positions in highly correlated pairs")
            
            if awareness['diversification_opportunities']:
                top_diversifier = awareness['diversification_opportunities'][0]['pair']
                awareness['recommendations'].append(f"Consider {top_diversifier} for portfolio diversification")
            
            return awareness
            
        except Exception as e:
            logger.error(f"❌ Error generating correlation awareness: {e}")
            return {'error': str(e)}
    
    def _generate_event_intelligence(self, raw_data: Dict) -> Dict:
        """Generate event calendar intelligence"""
        try:
            event_intel = {
                'timestamp': datetime.now().isoformat(),
                'high_impact_events': [],                # Next 48 hours high impact
                'medium_impact_events': [],              # Next 48 hours medium impact
                'currency_risk_levels': {},              # Risk level by currency
                'optimal_trading_windows': {},           # Best times to trade by pair
                'event_clusters': [],                    # Time periods with multiple events
                'recommendations': []                    # Trading recommendations
            }
            
            if not raw_data.get('calendar') or 'events' not in raw_data['calendar']:
                return event_intel
            
            events = raw_data['calendar']['events']
            
            # Categorize events by impact
            for event in events:
                impact = event.get('impact', '').lower()
                event_info = {
                    'name': event.get('event_name', 'Unknown'),
                    'currency': event.get('currency', ''),
                    'time': event.get('time', ''),
                    'date': event.get('event_date', ''),
                    'previous': event.get('previous', ''),
                    'consensus': event.get('consensus', '')
                }
                
                if impact == 'high':
                    event_intel['high_impact_events'].append(event_info)
                elif impact == 'medium':
                    event_intel['medium_impact_events'].append(event_info)
            
            # Calculate currency risk levels
            currency_events = {}
            for event in events:
                currency = event.get('currency', '')
                impact = event.get('impact', '').lower()
                
                if currency not in currency_events:
                    currency_events[currency] = {'high': 0, 'medium': 0, 'low': 0}
                
                currency_events[currency][impact] = currency_events[currency].get(impact, 0) + 1
            
            for currency, counts in currency_events.items():
                risk_score = counts.get('high', 0) * 3 + counts.get('medium', 0) * 2 + counts.get('low', 0)
                
                if risk_score >= 8:
                    risk_level = 'VERY_HIGH'
                elif risk_score >= 5:
                    risk_level = 'HIGH'
                elif risk_score >= 3:
                    risk_level = 'ELEVATED'
                else:
                    risk_level = 'NORMAL'
                
                event_intel['currency_risk_levels'][currency] = {
                    'risk_level': risk_level,
                    'event_count': sum(counts.values()),
                    'high_impact_count': counts.get('high', 0)
                }
            
            # Generate optimal trading windows for major pairs
            major_pairs = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'XAUUSD']
            
            for pair in major_pairs:
                relevant_events = self._filter_events_for_pair(pair, events)
                high_impact_count = len([e for e in relevant_events if e.get('impact', '').lower() == 'high'])
                
                if high_impact_count >= 3:
                    window = 'AVOID_NEXT_24H'
                elif high_impact_count >= 1:
                    window = 'CAUTION_BEFORE_EVENTS'
                else:
                    window = 'NORMAL_TRADING'
                
                event_intel['optimal_trading_windows'][pair] = window
            
            # Identify event clusters (multiple events in short timeframes)
            # This is a simplified version - in reality you'd parse actual times
            if len(event_intel['high_impact_events']) >= 3:
                event_intel['event_clusters'].append({
                    'period': 'Next 24 hours',
                    'event_count': len(event_intel['high_impact_events']),
                    'risk_level': 'HIGH'
                })
            
            # Generate recommendations
            high_risk_currencies = [curr for curr, data in event_intel['currency_risk_levels'].items() 
                                  if data['risk_level'] in ['HIGH', 'VERY_HIGH']]
            
            if high_risk_currencies:
                event_intel['recommendations'].append(f"High event risk for: {', '.join(high_risk_currencies)}")
                event_intel['recommendations'].append("Consider reduced position sizes before major events")
            
            if len(event_intel['high_impact_events']) > 5:
                event_intel['recommendations'].append("Heavy event week - prioritize risk management")
            
            if event_intel['event_clusters']:
                event_intel['recommendations'].append("Multiple events clustered - expect increased volatility")
            
            return event_intel
            
        except Exception as e:
            logger.error(f"❌ Error generating event intelligence: {e}")
            return {'error': str(e)}
    
    def _generate_risk_insights(self, raw_data: Dict, trading_preferences: Dict) -> List[Dict]:
        """Generate risk insights (not blocking alerts, but awareness)"""
        insights = []
        
        try:
            # High correlation insight
            high_corr_pairs = [pair for pair, prefs in trading_preferences.items() 
                             if prefs.get('correlation_risk_level') == 'HIGH']
            if high_corr_pairs:
                insights.append({
                    'type': 'CORRELATION_AWARENESS',
                    'priority': 'HIGH',
                    'title': 'High Correlation Environment',
                    'message': f'{len(high_corr_pairs)} pairs showing high correlation risk',
                    'affected_pairs': high_corr_pairs,
                    'recommendation': 'Consider reducing individual position sizes and improving diversification',
                    'impact': 'Portfolio concentration risk increased',
                    'timestamp': datetime.now().isoformat()
                })
            
            # Contrarian signal insight
            contrarian_pairs = [pair for pair, prefs in trading_preferences.items() 
                              if prefs.get('contrarian_signal')]
            if contrarian_pairs:
                insights.append({
                    'type': 'CONTRARIAN_OPPORTUNITY',
                    'priority': 'MEDIUM',
                    'title': 'Extreme Positioning Detected',
                    'message': f'{len(contrarian_pairs)} pairs showing extreme COT positioning',
                    'affected_pairs': contrarian_pairs,
                    'recommendation': 'Watch for potential trend reversals - extreme positioning often precedes turns',
                    'impact': 'Potential trend reversal opportunity',
                    'timestamp': datetime.now().isoformat()
                })
            
            # Poor opportunity insight
            poor_pairs = [pair for pair, prefs in trading_preferences.items() 
                        if prefs.get('overall_opportunity') == 'POOR']
            if poor_pairs:
                insights.append({
                    'type': 'LIMITED_OPPORTUNITY',
                    'priority': 'LOW',
                    'title': 'Limited Trading Opportunities',
                    'message': f'{len(poor_pairs)} pairs showing poor opportunity scores',
                    'affected_pairs': poor_pairs,
                    'recommendation': 'Focus capital on higher-opportunity pairs or wait for better setups',
                    'impact': 'Lower expected returns on these pairs',
                    'timestamp': datetime.now().isoformat()
                })
            
            # Event risk insight
            event_risk_pairs = [pair for pair, prefs in trading_preferences.items() 
                              if prefs.get('event_risk_hours', 0) > 12]
            if event_risk_pairs:
                insights.append({
                    'type': 'EVENT_RISK_AWARENESS',
                    'priority': 'MEDIUM',
                    'title': 'Upcoming High-Impact Events',
                    'message': f'{len(event_risk_pairs)} pairs have significant events in next 24 hours',
                    'affected_pairs': event_risk_pairs,
                    'recommendation': 'Consider timing entries around events or reducing size before volatility',
                    'impact': 'Increased short-term volatility expected',
                    'timestamp': datetime.now().isoformat()
                })
            
            # Excellent opportunity insight
            excellent_pairs = [pair for pair, prefs in trading_preferences.items() 
                             if prefs.get('overall_opportunity') == 'EXCELLENT']
            if excellent_pairs:
                insights.append({
                    'type': 'HIGH_OPPORTUNITY',
                    'priority': 'HIGH',
                    'title': 'Excellent Trading Opportunities',
                    'message': f'{len(excellent_pairs)} pairs showing excellent opportunity scores',
                    'affected_pairs': excellent_pairs,
                    'recommendation': 'Consider concentrating capital on these high-opportunity setups',
                    'impact': 'Higher expected returns potential',
                    'timestamp': datetime.now().isoformat()
                })
            
            return sorted(insights, key=lambda x: {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}[x['priority']], reverse=True)
            
        except Exception as e:
            logger.error(f"❌ Error generating risk insights: {e}")
            return []
    
    def _load_raw_data(self) -> Dict[str, Any]:
        """Load and validate raw data from all sources"""
        raw_data = {}
        
        for source, filename in CONFIG['input_files'].items():
            try:
                if os.path.exists(filename):
                    with open(filename, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    raw_data[source] = data
                    logger.info(f"📖 Loaded {source} data from {filename}")
                else:
                    logger.warning(f"⚠️ {filename} not found")
                    raw_data[source] = None
            except Exception as e:
                logger.error(f"❌ Error loading {source}: {e}")
                raw_data[source] = None
        
        return raw_data
    
    def _assess_data_quality(self, raw_data: Dict) -> Dict:
        """Assess quality of available data"""
        quality = {
            'overall_score': 0,
            'available_sources': [],
            'missing_sources': [],
            'data_age_status': 'UNKNOWN'
        }
        
        try:
            total_sources = len(CONFIG['input_files'])
            available_count = 0
            
            for source_name in CONFIG['input_files'].keys():
                if raw_data.get(source_name):
                    quality['available_sources'].append(source_name)
                    available_count += 1
                else:
                    quality['missing_sources'].append(source_name)
            
            quality['overall_score'] = round(available_count / total_sources, 2)
            
            if available_count >= 3:
                quality['data_age_status'] = 'FRESH'
            elif available_count >= 2:
                quality['data_age_status'] = 'ACCEPTABLE'
            else:
                quality['data_age_status'] = 'STALE'
            
            return quality
            
        except Exception as e:
            logger.error(f"❌ Error assessing data quality: {e}")
            return quality
    
    def _create_safe_fallback(self) -> Dict:
        """Create safe fallback when system fails"""
        return {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'version': '2.0',
                'approach': 'PREFERENCES_BASED',
                'status': 'FALLBACK_MODE',
                'valid_until': (datetime.now() + timedelta(minutes=30)).isoformat(),
                'data_quality': {'overall_score': 0, 'data_age_status': 'UNAVAILABLE'}
            },
            'trading_preferences': {pair: self._create_safe_pair_preferences(pair) for pair in TRADING_PAIRS},
            'market_intelligence': {
                'market_regime': 'UNCERTAIN',
                'recommended_approach': 'DEFENSIVE',
                'overall_confidence': 0.1
            },
            'correlation_awareness': {'correlation_strength': 'UNKNOWN'},
            'event_calendar': {'recommendations': ['Data unavailable - use manual analysis']},
            'risk_insights': [{
                'type': 'SYSTEM_FALLBACK',
                'priority': 'HIGH',
                'title': 'Intelligence System in Fallback Mode',
                'message': 'Using conservative defaults due to data unavailability',
                'recommendation': 'Verify all signals manually before trading',
                'timestamp': datetime.now().isoformat()
            }]
        }
    
    def _create_safe_pair_preferences(self, pair: str) -> Dict:
        """Create safe conservative preferences when analysis fails"""
        return {
            'pair': pair,
            'timestamp': datetime.now().isoformat(),
            'preferred_direction': 'NEUTRAL',
            'preference_strength': 'WEAK',
            'alternative_direction': 'ACCEPTABLE',
            'directional_confidence': 0.5,
            'position_size_modifier': 0.8,  # Conservative
            'max_exposure_pct': 60,         # Conservative
            'risk_adjustment': 'CONSERVATIVE',
            'size_adjustment_reason': ['Data unavailable - using conservative defaults'],
            'correlation_risk_level': 'MEDIUM',  # Conservative assumption
            'correlated_pairs': [],
            'diversification_score': 50,
            'correlation_impact': 'UNKNOWN',
            'event_risk_hours': 24,         # Assume events
            'optimal_entry_window': 'AFTER_EVENTS',
            'hold_duration_guidance': 'SHORT',
            'timing_factors': ['Data unavailable'],
            'cot_sentiment': 'NEUTRAL',
            'contrarian_signal': False,
            'institutional_bias': 'NEUTRAL',
            'cot_strength': 0.0,
            'overall_opportunity': 'POOR',  # Conservative
            'key_factors': ['Data unavailable - conservative mode'],
            'trading_notes': 'FALLBACK MODE: Data unavailable, using conservative preferences',
            'last_updated': datetime.now().isoformat()
        }

class IntelligenceManager:
    """Manages the intelligence generation and scheduling"""
    
    def __init__(self):
        self.intelligence_generator = SmartTradingIntelligence()
        self.running = False
        self.thread = None
        
    def start_continuous_updates(self):
        """Start continuous intelligence updates"""
        try:
            logger.info("🚀 Starting Smart Trading Intelligence Manager (Preferences Mode)...")
            
            schedule.every(CONFIG['update_interval_minutes']).minutes.do(self._update_intelligence)
            
            logger.info("🧠 Generating initial intelligence...")
            self._update_intelligence()
            
            self.running = True
            self.thread = threading.Thread(target=self._run_scheduler, daemon=True)
            self.thread.start()
            
            logger.info(f"✅ Intelligence manager started - updates every {CONFIG['update_interval_minutes']} minutes")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error starting intelligence manager: {e}")
            return False
    
    def stop(self):
        """Stop the intelligence manager"""
        try:
            self.running = False
            schedule.clear()
            
            if self.thread:
                self.thread.join(timeout=5)
            
            logger.info("🛑 Intelligence manager stopped")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error stopping intelligence manager: {e}")
            return False
    
    def _run_scheduler(self):
        """Run the scheduler loop"""
        logger.info("⚙️ Intelligence scheduler thread started")
        
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"❌ Error in scheduler loop: {e}")
                time.sleep(60)
        
        logger.info("⚙️ Intelligence scheduler thread stopped")
    
    def _update_intelligence(self):
        """Update trading intelligence"""
        try:
            logger.info("🔄 Updating trading intelligence...")
            
            intelligence = self.intelligence_generator.generate_intelligence()
            self._save_intelligence(intelligence)
            self._log_intelligence_summary(intelligence)
            
            logger.info("✅ Trading intelligence updated successfully")
            
        except Exception as e:
            logger.error(f"❌ Error updating intelligence: {e}")
    
    def _save_intelligence(self, intelligence: Dict):
        """Save intelligence to JSON file"""
        try:
            output_file = CONFIG['output_file']
            if os.path.exists(output_file):
                backup_file = f"{output_file}.backup"
                try:
                    import shutil
                    shutil.copy2(output_file, backup_file)
                except Exception as e:
                    logger.warning(f"Could not create backup: {e}")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(intelligence, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"💾 Intelligence saved to {output_file}")
            
        except Exception as e:
            logger.error(f"❌ Error saving intelligence: {e}")
    
    def _log_intelligence_summary(self, intelligence: Dict):
        """Log summary of generated intelligence"""
        try:
            trading_preferences = intelligence.get('trading_preferences', {})
            risk_insights = intelligence.get('risk_insights', [])
            market_intel = intelligence.get('market_intelligence', {})
            
            logger.info("📊 INTELLIGENCE SUMMARY (PREFERENCES MODE):")
            logger.info(f"   Pairs analyzed: {len(trading_preferences)}")
            logger.info(f"   Risk insights: {len(risk_insights)}")
            logger.info(f"   Market regime: {market_intel.get('market_regime', 'UNKNOWN')}")
            logger.info(f"   Recommended approach: {market_intel.get('recommended_approach', 'UNKNOWN')}")
            
            # Count opportunities
            opportunities = {}
            directions = {}
            for prefs in trading_preferences.values():
                opp = prefs.get('overall_opportunity', 'UNKNOWN')
                opportunities[opp] = opportunities.get(opp, 0) + 1
                
                direction = prefs.get('preferred_direction', 'NEUTRAL')
                directions[direction] = directions.get(direction, 0) + 1
            
            logger.info(f"   Opportunities: {opportunities}")
            logger.info(f"   Directional bias: {directions}")
            
            # Log high-priority insights
            high_insights = [i for i in risk_insights if i.get('priority') == 'HIGH']
            for insight in high_insights[:3]:
                logger.info(f"   🎯 {insight.get('title', 'INSIGHT')}: {insight.get('message', '')}")
            
        except Exception as e:
            logger.error(f"❌ Error logging summary: {e}")
    
    def force_update(self):
        """Force immediate intelligence update"""
        try:
            logger.info("🔥 Forcing immediate intelligence update...")
            self._update_intelligence()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in force update: {e}")
            return False
    
    def get_current_intelligence(self) -> Optional[Dict]:
        """Get current intelligence from file"""
        try:
            if os.path.exists(CONFIG['output_file']):
                with open(CONFIG['output_file'], 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning("⚠️ Intelligence file not found")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error loading intelligence: {e}")
            return None

# ===== UTILITY FUNCTIONS =====

def create_sample_intelligence():
    """Create sample trading intelligence for testing"""
    logger.info("🔧 Creating sample trading intelligence...")
    
    generator = SmartTradingIntelligence()
    intelligence = generator.generate_intelligence()
    
    try:
        with open(CONFIG['output_file'], 'w', encoding='utf-8') as f:
            json.dump(intelligence, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info(f"✅ Sample intelligence created: {CONFIG['output_file']}")
        
        trading_preferences = intelligence.get('trading_preferences', {})
        risk_insights = intelligence.get('risk_insights', [])
        
        logger.info(f"   📊 Analyzed {len(trading_preferences)} pairs")
        logger.info(f"   🎯 Generated {len(risk_insights)} insights")
        
        # Show some examples
        for pair, prefs in list(trading_preferences.items())[:3]:
            opp = prefs.get('overall_opportunity', 'UNKNOWN')
            direction = prefs.get('preferred_direction', 'NEUTRAL')
            strength = prefs.get('preference_strength', 'WEAK')
            logger.info(f"   📈 {pair}: {direction} ({strength}) - {opp} opportunity")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating sample intelligence: {e}")
        return False

def test_intelligence_system():
    """Test the preferences-based intelligence system"""
    logger.info("🧪 Testing Smart Trading Intelligence System (Preferences Mode)...")
    
    try:
        generator = SmartTradingIntelligence()
        intelligence = generator.generate_intelligence()
        
        if intelligence and 'trading_preferences' in intelligence:
            logger.info("✅ Intelligence generation successful")
            
            manager = IntelligenceManager()
            if manager.force_update():
                logger.info("✅ Manager force update successful")
            
            if os.path.exists(CONFIG['output_file']):
                logger.info(f"✅ Output file created: {CONFIG['output_file']}")
                
                size = os.path.getsize(CONFIG['output_file'])
                logger.info(f"   File size: {size} bytes")
                
                return True
            else:
                logger.error("❌ Output file not created")
                return False
        else:
            logger.error("❌ Intelligence generation failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in system test: {e}")
        return False

def run_daemon():
    """Run intelligence manager as daemon"""
    manager = IntelligenceManager()
    
    try:
        if not manager.start_continuous_updates():
            logger.error("❌ Failed to start intelligence manager")
            return
        
        logger.info("🎯 Smart Trading Intelligence running (Preferences Mode). Press Ctrl+C to stop.")
        
        while manager.running:
            try:
                time.sleep(300)
                
                if datetime.now().minute == 0:
                    intelligence = manager.get_current_intelligence()
                    if intelligence:
                        market_regime = intelligence.get('market_intelligence', {}).get('market_regime', 'UNKNOWN')
                        logger.info(f"📊 Market Regime: {market_regime}")
                        
            except KeyboardInterrupt:
                logger.info("🛑 Stop signal received")
                break
                
    except Exception as e:
        logger.error(f"❌ Error in daemon mode: {e}")
        
    finally:
        manager.stop()

def main():
    """Main function"""
    import sys
    
    logger.info("="*60)
    logger.info("SMART TRADING INTELLIGENCE SYSTEM - PREFERENCES MODE")
    logger.info("="*60)
    logger.info(f"Output file: {CONFIG['output_file']}")
    logger.info(f"Update interval: {CONFIG['update_interval_minutes']} minutes")
    logger.info(f"Monitored pairs: {len(TRADING_PAIRS)}")
    logger.info("Approach: Preferences & guidance instead of hard blocks")
    logger.info("="*60)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'test':
            test_intelligence_system()
        elif command == 'sample':
            create_sample_intelligence()
        elif command == 'daemon':
            run_daemon()
        elif command == 'once':
            manager = IntelligenceManager()
            success = manager.force_update()
            sys.exit(0 if success else 1)
        else:
            print("Usage: python smart_trading_intelligence.py [test|sample|daemon|once]")
            print("")
            print("Commands:")
            print("  test   - Test the intelligence system")
            print("  sample - Create sample intelligence file")
            print("  daemon - Run continuous updates")
            print("  once   - Run single update and exit")
            sys.exit(1)
    else:
        manager = IntelligenceManager()
        manager.force_update()

if __name__ == "__main__":
    main()