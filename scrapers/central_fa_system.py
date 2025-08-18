# ===== CENTRAL FUNDAMENTAL ANALYSIS INTELLIGENCE SYSTEM =====
# Orchestrates all FA data scrapers and generates unified intelligence JSON
# Provides consolidated FA signals for trading robots

import os
import sys
import json
import logging
import threading
import time
import schedule
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import subprocess
import importlib.util
import warnings

warnings.filterwarnings("ignore")

# ===== CONFIGURATION =====
SYSTEM_CONFIG = {
    'system_name': 'Central_FA_Intelligence',
    'version': '1.0.0',
    'update_intervals': {
        'sentiment': 30,      # minutes
        'correlation': 30,    # minutes  
        'calendar': 240,      # minutes (4 hours)
        'cot': 10080         # minutes (weekly - Fridays)
    },
    'output_file': 'fa_intelligence.json',
    'log_file': 'central_fa_system.log',
    'data_freshness_limits': {
        'sentiment': 60,      # minutes
        'correlation': 90,    # minutes
        'calendar': 720,      # minutes (12 hours)
        'cot': 20160         # minutes (2 weeks)
    }
}

# Scraper module configurations
SCRAPER_MODULES = {
    'calendar': {
        'file': 'calendar_scraper.py',
        'class': 'FixedEconomicCalendarScraper',
        'output_file': 'test_calendar_output.json',
        'data_key': 'events'
    },
    'correlation': {
        'file': 'correlation_scraper.py', 
        'class': 'CorrelationSignalManager',
        'output_file': 'correlation_data.json',
        'data_key': 'correlation_matrix'
    },
    'cot': {
        'file': 'cot_scraper.py',
        'class': 'COTDataManager', 
        'output_file': 'cot_consolidated_data.json',
        'data_key': 'data'
    },
    'sentiment': {
        'file': 'sentiment_scraper.py',
        'class': 'SentimentSignalManager',
        'output_file': 'sentiment_signals.json', 
        'data_key': 'pairs'
    }
}

# Trading pairs to monitor
MONITORED_PAIRS = [
    'AUDUSD', 'USDCAD', 'XAUUSD', 'EURUSD', 'GBPUSD',
    'AUDCAD', 'USDCHF', 'GBPCAD', 'AUDNZD', 'NZDCAD', 
    'US500', 'BTCUSD'
]

# ===== LOGGING SETUP =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - CENTRAL_FA - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(SYSTEM_CONFIG['log_file'], encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===== MAIN SYSTEM CLASSES =====

class DataSourceManager:
    """Manages individual data source scrapers"""
    
    def __init__(self):
        self.scrapers = {}
        self.last_updates = {}
        
    def initialize_scrapers(self):
        """Initialize all scraper modules"""
        try:
            logger.info("🔧 Initializing scraper modules...")
            
            for source_name, config in SCRAPER_MODULES.items():
                try:
                    # Import module dynamically
                    spec = importlib.util.spec_from_file_location(
                        source_name, config['file']
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Get the manager class
                    scraper_class = getattr(module, config['class'])
                    self.scrapers[source_name] = scraper_class()
                    
                    logger.info(f"✅ Initialized {source_name} scraper")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to initialize {source_name}: {e}")
                    self.scrapers[source_name] = None
                    
            successful_scrapers = sum(1 for s in self.scrapers.values() if s is not None)
            logger.info(f"🎯 Initialized {successful_scrapers}/{len(SCRAPER_MODULES)} scrapers")
            
            return successful_scrapers > 0
            
        except Exception as e:
            logger.error(f"❌ Error initializing scrapers: {e}")
            return False
    
    def update_data_source(self, source_name: str) -> bool:
        """Update a specific data source"""
        try:
            if source_name not in self.scrapers:
                logger.error(f"❌ Unknown data source: {source_name}")
                return False
                
            scraper = self.scrapers[source_name]
            if scraper is None:
                logger.error(f"❌ {source_name} scraper not initialized")
                return False
            
            logger.info(f"🔄 Updating {source_name} data...")
            
            # Call appropriate update method based on source
            if source_name == 'calendar':
                # Generate target dates for calendar
                target_dates = self._generate_calendar_dates()
                success = len(scraper.scrape_calendar_data(target_dates)) > 0
                
            elif source_name == 'correlation':
                success = scraper.update_correlation_data()
                
            elif source_name == 'cot':
                success = scraper.update_cot_data()
                
            elif source_name == 'sentiment':
                success = scraper.update_sentiment_signals()
                
            else:
                logger.error(f"❌ Unknown update method for {source_name}")
                return False
            
            if success:
                self.last_updates[source_name] = datetime.now()
                logger.info(f"✅ {source_name} data updated successfully")
            else:
                logger.error(f"❌ {source_name} data update failed")
                
            return success
            
        except Exception as e:
            logger.error(f"❌ Error updating {source_name}: {e}")
            return False
    
    def _generate_calendar_dates(self) -> List[str]:
        """Generate target dates for calendar scraper"""
        target_dates = []
        current_date = datetime.now()
        
        # Get next 7 trading days
        for i in range(10):  # Check 10 days to get 7 trading days
            check_date = current_date + timedelta(days=i)
            if check_date.weekday() < 5:  # Monday = 0, Friday = 4
                formatted_date = check_date.strftime("%A, %b %d, %Y")
                target_dates.append(formatted_date)
                if len(target_dates) >= 7:
                    break
                    
        return target_dates
    
    def get_source_status(self, source_name: str) -> Dict[str, Any]:
        """Get status of a specific data source"""
        try:
            config = SCRAPER_MODULES.get(source_name, {})
            output_file = config.get('output_file', '')
            
            status = {
                'source': source_name,
                'initialized': source_name in self.scrapers and self.scrapers[source_name] is not None,
                'last_update': self.last_updates.get(source_name),
                'file_exists': os.path.exists(output_file),
                'file_age_minutes': None,
                'data_fresh': False
            }
            
            # Check file age
            if status['file_exists']:
                file_time = datetime.fromtimestamp(os.path.getmtime(output_file))
                age_minutes = (datetime.now() - file_time).total_seconds() / 60
                status['file_age_minutes'] = age_minutes
                
                freshness_limit = SYSTEM_CONFIG['data_freshness_limits'].get(source_name, 60)
                status['data_fresh'] = age_minutes <= freshness_limit
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Error getting {source_name} status: {e}")
            return {'source': source_name, 'error': str(e)}

class IntelligenceProcessor:
    """Processes raw FA data into unified intelligence"""
    
    def __init__(self):
        self.processing_rules = {
            'sentiment_weight': 0.3,
            'correlation_weight': 0.25, 
            'calendar_weight': 0.25,
            'cot_weight': 0.2
        }
    
    def process_all_data(self) -> Dict[str, Any]:
        """Process all FA data sources into unified intelligence"""
        try:
            logger.info("🧠 Processing FA data into unified intelligence...")
            
            # Load all data sources
            raw_data = self._load_all_sources()
            
            # Create intelligence structure
            intelligence = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'system_version': SYSTEM_CONFIG['version'],
                    'data_sources_used': list(raw_data.keys()),
                    'processing_rules': self.processing_rules
                },
                'pair_analysis': {},
                'market_overview': {},
                'alerts': [],
                'system_health': {}
            }
            
            # Process each monitored pair
            for pair in MONITORED_PAIRS:
                pair_intel = self._process_pair_intelligence(pair, raw_data)
                if pair_intel:
                    intelligence['pair_analysis'][pair] = pair_intel
            
            # Generate market overview
            intelligence['market_overview'] = self._generate_market_overview(raw_data)
            
            # Generate alerts
            intelligence['alerts'] = self._generate_alerts(raw_data, intelligence['pair_analysis'])
            
            # System health
            intelligence['system_health'] = self._assess_system_health(raw_data)
            
            logger.info(f"✅ Processed intelligence for {len(intelligence['pair_analysis'])} pairs")
            return intelligence
            
        except Exception as e:
            logger.error(f"❌ Error processing intelligence: {e}")
            return self._create_fallback_intelligence()
    
    def _load_all_sources(self) -> Dict[str, Any]:
        """Load data from all sources"""
        raw_data = {}
        
        for source_name, config in SCRAPER_MODULES.items():
            try:
                output_file = config['output_file']
                if os.path.exists(output_file):
                    with open(output_file, 'r') as f:
                        data = json.load(f)
                    raw_data[source_name] = data
                    logger.debug(f"📖 Loaded {source_name} data")
                else:
                    logger.warning(f"⚠️ {source_name} data file not found")
                    raw_data[source_name] = None
                    
            except Exception as e:
                logger.error(f"❌ Error loading {source_name}: {e}")
                raw_data[source_name] = None
        
        return raw_data
    
    def _process_pair_intelligence(self, pair: str, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process intelligence for a specific trading pair"""
        try:
            pair_intel = {
                'pair': pair,
                'timestamp': datetime.now().isoformat(),
                'sentiment_analysis': {},
                'correlation_analysis': {},
                'fundamental_events': [],
                'cot_analysis': {},
                'overall_signal': 'NEUTRAL',
                'risk_factors': [],
                'confidence_score': 0.0
            }
            
            # Process sentiment data
            if raw_data.get('sentiment') and raw_data['sentiment'].get('pairs', {}).get(pair):
                sentiment_data = raw_data['sentiment']['pairs'][pair]
                pair_intel['sentiment_analysis'] = {
                    'allowed_directions': sentiment_data.get('allowed_directions', []),
                    'blocked_directions': sentiment_data.get('blocked_directions', []),
                    'sentiment_scores': sentiment_data.get('sentiment', {}),
                    'signal_strength': sentiment_data.get('signal_strength', 'Unknown')
                }
            
            # Process correlation data
            if raw_data.get('correlation'):
                corr_matrix = raw_data['correlation'].get('correlation_matrix', {})
                warnings = raw_data['correlation'].get('warnings', [])
                
                # Find correlations for this pair
                pair_correlations = {}
                for row_symbol, correlations in corr_matrix.items():
                    if self._normalize_pair_name(row_symbol) == self._normalize_pair_name(pair):
                        pair_correlations = correlations
                        break
                
                # Find high correlation warnings for this pair
                pair_warnings = [w for w in warnings if pair in w.get('pair', '')]
                
                pair_intel['correlation_analysis'] = {
                    'high_correlations': self._extract_high_correlations(pair_correlations),
                    'correlation_warnings': pair_warnings,
                    'correlation_risk_level': 'HIGH' if pair_warnings else 'LOW'
                }
            
            # Process calendar events
            if raw_data.get('calendar') and raw_data['calendar'].get('events'):
                events = raw_data['calendar']['events']
                pair_events = self._filter_events_for_pair(events, pair)
                pair_intel['fundamental_events'] = pair_events
            
            # Process COT data
            if raw_data.get('cot'):
                cot_data = raw_data['cot'].get('data', {})
                cot_analysis = self._analyze_cot_for_pair(cot_data, pair)
                pair_intel['cot_analysis'] = cot_analysis
            
            # Calculate overall signal and confidence
            overall_signal, confidence = self._calculate_overall_signal(pair_intel)
            pair_intel['overall_signal'] = overall_signal
            pair_intel['confidence_score'] = confidence
            
            # Identify risk factors
            pair_intel['risk_factors'] = self._identify_risk_factors(pair_intel)
            
            return pair_intel
            
        except Exception as e:
            logger.error(f"❌ Error processing {pair} intelligence: {e}")
            return None
    
    def _normalize_pair_name(self, pair: str) -> str:
        """Normalize pair names for comparison"""
        mapping = {
            'GOLD': 'XAUUSD',
            'XAUUSD': 'XAUUSD',
            'SPX500': 'US500',
            'US500': 'US500',
            'BITCOIN': 'BTCUSD',
            'BTCUSD': 'BTCUSD',
            'AUD': 'AUDUSD',
            'EUR': 'EURUSD',
            'GBP': 'GBPUSD',
            'CAD': 'USDCAD',
            'CHF': 'USDCHF'
        }
        return mapping.get(pair.upper(), pair.upper())
    
    def _extract_high_correlations(self, correlations: Dict) -> List[Dict]:
        """Extract high correlation relationships"""
        high_corrs = []
        for symbol, data in correlations.items():
            correlation_value = data.get('value', 0)
            if abs(correlation_value) >= 70:  # High correlation threshold
                high_corrs.append({
                    'symbol': symbol,
                    'correlation': correlation_value,
                    'strength': 'Strong Positive' if correlation_value > 70 else 'Strong Negative'
                })
        return sorted(high_corrs, key=lambda x: abs(x['correlation']), reverse=True)
    
    def _filter_events_for_pair(self, events: List[Dict], pair: str) -> List[Dict]:
        """Filter calendar events relevant to a trading pair"""
        relevant_events = []
        
        # Currency mapping for events
        currency_map = {
            'AUDUSD': ['AUD', 'USD'],
            'EURUSD': ['EUR', 'USD'], 
            'GBPUSD': ['GBP', 'USD'],
            'USDCAD': ['USD', 'CAD'],
            'USDCHF': ['USD', 'CHF'],
            'XAUUSD': ['USD'],  # Gold affected by USD
            'US500': ['USD'],   # US500 affected by USD
            'BTCUSD': ['USD']   # Bitcoin affected by USD
        }
        
        target_currencies = currency_map.get(pair, [])
        
        for event in events:
            event_currency = event.get('currency', '')
            if event_currency in target_currencies:
                # Add relevance score based on impact
                impact = event.get('impact', '').lower()
                if impact == 'high':
                    event['relevance_score'] = 3
                elif impact == 'medium':
                    event['relevance_score'] = 2
                else:
                    event['relevance_score'] = 1
                    
                relevant_events.append(event)
        
        # Sort by relevance and time
        return sorted(relevant_events, key=lambda x: (x.get('relevance_score', 0), x.get('time', '')), reverse=True)
    
    def _analyze_cot_for_pair(self, cot_data: Dict, pair: str) -> Dict:
        """Analyze COT data for a specific pair"""
        try:
            # Map trading pair to COT instrument
            cot_mapping = {
                'AUDUSD': 'AUD',
                'EURUSD': 'EUR',
                'GBPUSD': 'GBP',
                'USDCAD': 'CAD',
                'USDCHF': 'CHF',
                'XAUUSD': 'Gold'
            }
            
            cot_instrument = cot_mapping.get(pair)
            if not cot_instrument:
                return {'available': False, 'reason': 'No COT mapping'}
            
            # Look for the instrument in both Financial and Commodity data
            cot_analysis = {'available': False}
            
            for data_type in ['Financial', 'Commodity']:
                if data_type in cot_data:
                    type_data = cot_data[data_type]
                    
                    # Find latest data for this instrument
                    instrument_data = [
                        record for record in type_data 
                        if record.get('Name') == cot_instrument
                    ]
                    
                    if instrument_data:
                        # Get latest record
                        latest_record = max(instrument_data, key=lambda x: x.get('Date', ''))
                        
                        cot_analysis = {
                            'available': True,
                            'data_type': data_type,
                            'latest_date': latest_record.get('Date'),
                            'net_position': latest_record.get('Non_Commercial_Net', 0),
                            'net_percentage': latest_record.get('Non_Commercial_Net_Pct', 0),
                            'open_interest': latest_record.get('Open_Interest', 0),
                            'sentiment': 'Bullish' if latest_record.get('Non_Commercial_Net', 0) > 0 else 'Bearish'
                        }
                        break
            
            return cot_analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing COT for {pair}: {e}")
            return {'available': False, 'error': str(e)}
    
    def _calculate_overall_signal(self, pair_intel: Dict) -> tuple:
        """Calculate overall signal and confidence for a pair"""
        try:
            signals = []
            weights = []
            
            # Sentiment signal
            sentiment = pair_intel.get('sentiment_analysis', {})
            if sentiment.get('signal_strength'):
                if sentiment['signal_strength'] == 'Strong Long':
                    signals.append(1)
                elif sentiment['signal_strength'] == 'Strong Short':
                    signals.append(-1)
                else:
                    signals.append(0)
                weights.append(self.processing_rules['sentiment_weight'])
            
            # COT signal
            cot = pair_intel.get('cot_analysis', {})
            if cot.get('available'):
                net_pct = cot.get('net_percentage', 0)
                if net_pct > 10:
                    signals.append(1)
                elif net_pct < -10:
                    signals.append(-1)
                else:
                    signals.append(0)
                weights.append(self.processing_rules['cot_weight'])
            
            # Calendar signal (based on high impact events)
            events = pair_intel.get('fundamental_events', [])
            high_impact_events = [e for e in events if e.get('impact', '').lower() == 'high']
            if high_impact_events:
                # Negative signal if many high impact events (uncertainty)
                signals.append(-0.5)
                weights.append(self.processing_rules['calendar_weight'])
            
            # Correlation risk signal
            corr = pair_intel.get('correlation_analysis', {})
            if corr.get('correlation_risk_level') == 'HIGH':
                signals.append(-0.3)  # Negative for high correlation risk
                weights.append(self.processing_rules['correlation_weight'])
            
            # Calculate weighted average
            if signals and weights:
                weighted_signal = sum(s * w for s, w in zip(signals, weights)) / sum(weights)
                confidence = min(sum(weights), 1.0)  # Max confidence is 1.0
                
                if weighted_signal > 0.3:
                    overall_signal = 'BULLISH'
                elif weighted_signal < -0.3:
                    overall_signal = 'BEARISH'
                else:
                    overall_signal = 'NEUTRAL'
                    
                return overall_signal, round(confidence, 2)
            
            return 'NEUTRAL', 0.0
            
        except Exception as e:
            logger.error(f"❌ Error calculating overall signal: {e}")
            return 'NEUTRAL', 0.0
    
    def _identify_risk_factors(self, pair_intel: Dict) -> List[str]:
        """Identify risk factors for a trading pair"""
        risk_factors = []
        
        # High correlation risk
        corr_analysis = pair_intel.get('correlation_analysis', {})
        if corr_analysis.get('correlation_risk_level') == 'HIGH':
            risk_factors.append('High correlation with other instruments')
        
        # Sentiment blocking
        sentiment = pair_intel.get('sentiment_analysis', {})
        if sentiment.get('blocked_directions'):
            blocked = ', '.join(sentiment['blocked_directions'])
            risk_factors.append(f'Sentiment blocking {blocked} positions')
        
        # High impact events
        events = pair_intel.get('fundamental_events', [])
        high_impact_count = len([e for e in events if e.get('impact', '').lower() == 'high'])
        if high_impact_count >= 2:
            risk_factors.append(f'{high_impact_count} high-impact events scheduled')
        
        # COT extreme positioning
        cot = pair_intel.get('cot_analysis', {})
        if cot.get('available'):
            net_pct = abs(cot.get('net_percentage', 0))
            if net_pct > 30:
                risk_factors.append('Extreme COT positioning detected')
        
        return risk_factors
    
    def _generate_market_overview(self, raw_data: Dict) -> Dict:
        """Generate overall market overview"""
        try:
            overview = {
                'timestamp': datetime.now().isoformat(),
                'market_sentiment': 'NEUTRAL',
                'risk_level': 'MEDIUM',
                'major_events_count': 0,
                'high_correlation_pairs': 0,
                'cot_extremes': 0
            }
            
            # Analyze sentiment data
            if raw_data.get('sentiment'):
                pairs = raw_data['sentiment'].get('pairs', {})
                strong_signals = sum(1 for p in pairs.values() if p.get('signal_strength') in ['Strong Long', 'Strong Short'])
                total_pairs = len(pairs)
                
                if total_pairs > 0:
                    strong_ratio = strong_signals / total_pairs
                    if strong_ratio > 0.6:
                        overview['market_sentiment'] = 'EXTREME'
                        overview['risk_level'] = 'HIGH'
                    elif strong_ratio > 0.3:
                        overview['market_sentiment'] = 'DIRECTIONAL'
            
            # Count major events
            if raw_data.get('calendar'):
                events = raw_data['calendar'].get('events', [])
                overview['major_events_count'] = len([e for e in events if e.get('impact', '').lower() == 'high'])
            
            # Count high correlation warnings
            if raw_data.get('correlation'):
                warnings = raw_data['correlation'].get('warnings', [])
                overview['high_correlation_pairs'] = len(warnings)
                if len(warnings) > 10:
                    overview['risk_level'] = 'HIGH'
            
            return overview
            
        except Exception as e:
            logger.error(f"❌ Error generating market overview: {e}")
            return {'error': str(e)}
    
    def _generate_alerts(self, raw_data: Dict, pair_analysis: Dict) -> List[Dict]:
        """Generate system alerts"""
        alerts = []
        
        try:
            # High correlation alerts
            if raw_data.get('correlation'):
                warnings = raw_data['correlation'].get('warnings', [])
                for warning in warnings[:5]:  # Top 5 correlation warnings
                    alerts.append({
                        'type': 'CORRELATION_WARNING',
                        'severity': 'MEDIUM',
                        'message': warning.get('message', ''),
                        'pair': warning.get('pair', ''),
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Major event alerts
            if raw_data.get('calendar'):
                events = raw_data['calendar'].get('events', [])
                high_impact = [e for e in events if e.get('impact', '').lower() == 'high']
                for event in high_impact[:3]:  # Top 3 high impact events
                    alerts.append({
                        'type': 'HIGH_IMPACT_EVENT',
                        'severity': 'HIGH',
                        'message': f"High impact event: {event.get('event_name', 'Unknown')}",
                        'currency': event.get('currency', ''),
                        'time': event.get('time', ''),
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Sentiment blocking alerts
            for pair, analysis in pair_analysis.items():
                sentiment = analysis.get('sentiment_analysis', {})
                blocked = sentiment.get('blocked_directions', [])
                if blocked:
                    alerts.append({
                        'type': 'SENTIMENT_BLOCKING',
                        'severity': 'MEDIUM',
                        'message': f"{pair} sentiment blocking {', '.join(blocked)} positions",
                        'pair': pair,
                        'timestamp': datetime.now().isoformat()
                    })
            
            return sorted(alerts, key=lambda x: x['severity'], reverse=True)
            
        except Exception as e:
            logger.error(f"❌ Error generating alerts: {e}")
            return []
    
    def _assess_system_health(self, raw_data: Dict) -> Dict:
        """Assess overall system health"""
        try:
            health = {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'HEALTHY',
                'data_sources': {},
                'issues': []
            }
            
            # Check each data source
            for source_name in SCRAPER_MODULES.keys():
                if raw_data.get(source_name):
                    health['data_sources'][source_name] = 'ACTIVE'
                else:
                    health['data_sources'][source_name] = 'MISSING'
                    health['issues'].append(f'{source_name} data not available')
            
            # Determine overall status
            missing_count = sum(1 for status in health['data_sources'].values() if status == 'MISSING')
            if missing_count >= 3:
                health['overall_status'] = 'DEGRADED'
            elif missing_count >= 2:
                health['overall_status'] = 'WARNING'
            
            return health
            
        except Exception as e:
            logger.error(f"❌ Error assessing system health: {e}")
            return {'error': str(e)}
    
    def _create_fallback_intelligence(self) -> Dict:
        """Create fallback intelligence when processing fails"""
        return {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'system_version': SYSTEM_CONFIG['version'],
                'status': 'FALLBACK_MODE',
                'error': 'Intelligence processing failed'
            },
            'pair_analysis': {pair: {
                'pair': pair,
                'overall_signal': 'NEUTRAL',
                'confidence_score': 0.0,
                'risk_factors': ['System in fallback mode']
            } for pair in MONITORED_PAIRS},
            'market_overview': {
                'market_sentiment': 'UNKNOWN',
                'risk_level': 'HIGH',
                'status': 'FALLBACK_MODE'
            },
            'alerts': [{
                'type': 'SYSTEM_ERROR',
                'severity': 'HIGH',
                'message': 'FA Intelligence system in fallback mode',
                'timestamp': datetime.now().isoformat()
            }],
            'system_health': {
                'overall_status': 'ERROR',
                'timestamp': datetime.now().isoformat()
            }
        }

class SchedulerManager:
    """Manages scheduling for all data sources"""
    
    def __init__(self, data_manager: DataSourceManager, processor: IntelligenceProcessor):
        self.data_manager = data_manager
        self.processor = processor
        self.running = False
        self.scheduler_thread = None
        self.intelligence_file = SYSTEM_CONFIG['output_file']
        
    def start_scheduler(self):
        """Start the central scheduler"""
        try:
            logger.info("🕐 Starting Central FA Scheduler...")
            
            # Schedule each data source based on its interval
            intervals = SYSTEM_CONFIG['update_intervals']
            
            # Sentiment - every 30 minutes
            schedule.every(intervals['sentiment']).minutes.do(
                self._scheduled_update, 'sentiment'
            )
            
            # Correlation - every 30 minutes (offset by 15 minutes)
            schedule.every(intervals['correlation']).minutes.do(
                self._scheduled_update, 'correlation'
            ).tag('correlation')
            
            # Calendar - every 4 hours
            schedule.every(intervals['calendar']).minutes.do(
                self._scheduled_update, 'calendar'
            )
            
            # COT - weekly on Fridays at 6 PM
            schedule.every().friday.at("18:00").do(
                self._scheduled_update, 'cot'
            )
            
            # Intelligence processing - every 15 minutes
            schedule.every(15).minutes.do(self._process_intelligence)
            
            # Run initial updates
            logger.info("🚀 Running initial FA data collection...")
            self._run_initial_updates()
            
            # Start scheduler thread
            self.running = True
            self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
            self.scheduler_thread.start()
            
            logger.info("✅ Central FA Scheduler started successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error starting scheduler: {e}")
            return False
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        try:
            self.running = False
            schedule.clear()
            
            if self.scheduler_thread:
                self.scheduler_thread.join(timeout=10)
            
            logger.info("🛑 Central FA Scheduler stopped")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error stopping scheduler: {e}")
            return False
    
    def _run_scheduler(self):
        """Main scheduler loop"""
        logger.info("⚙️ Scheduler thread started")
        
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in scheduler loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
        
        logger.info("⚙️ Scheduler thread stopped")
    
    def _run_initial_updates(self):
        """Run initial updates for all data sources"""
        try:
            # Start with fastest updating sources first
            update_order = ['sentiment', 'correlation', 'calendar', 'cot']
            
            for source in update_order:
                try:
                    logger.info(f"🔄 Initial update: {source}")
                    success = self.data_manager.update_data_source(source)
                    if success:
                        logger.info(f"✅ {source} initial update completed")
                    else:
                        logger.warning(f"⚠️ {source} initial update failed")
                    
                    # Small delay between updates
                    time.sleep(5)
                    
                except Exception as e:
                    logger.error(f"❌ Error in {source} initial update: {e}")
                    continue
            
            # Process initial intelligence
            logger.info("🧠 Processing initial intelligence...")
            self._process_intelligence()
            
        except Exception as e:
            logger.error(f"❌ Error in initial updates: {e}")
    
    def _scheduled_update(self, source_name: str):
        """Handle scheduled update for a data source"""
        try:
            logger.info(f"⏰ Scheduled update: {source_name}")
            success = self.data_manager.update_data_source(source_name)
            
            if success:
                logger.info(f"✅ {source_name} scheduled update completed")
                # Trigger intelligence processing after successful update
                self._process_intelligence()
            else:
                logger.error(f"❌ {source_name} scheduled update failed")
                
        except Exception as e:
            logger.error(f"❌ Error in {source_name} scheduled update: {e}")
    
    def _process_intelligence(self):
        """Process and save intelligence"""
        try:
            logger.info("🧠 Processing FA intelligence...")
            
            # Process all data into intelligence
            intelligence = self.processor.process_all_data()
            
            # Save to file
            self._save_intelligence(intelligence)
            
            logger.info("✅ FA intelligence processed and saved")
            
        except Exception as e:
            logger.error(f"❌ Error processing intelligence: {e}")
    
    def _save_intelligence(self, intelligence: Dict):
        """Save intelligence to JSON file"""
        try:
            # Create backup
            if os.path.exists(self.intelligence_file):
                backup_file = f"{self.intelligence_file}.backup"
                try:
                    import shutil
                    shutil.copy2(self.intelligence_file, backup_file)
                except Exception as e:
                    logger.warning(f"Could not create backup: {e}")
            
            # Save new intelligence
            with open(self.intelligence_file, 'w') as f:
                json.dump(intelligence, f, indent=2, default=str)
            
            logger.info(f"💾 Intelligence saved to {self.intelligence_file}")
            
        except Exception as e:
            logger.error(f"❌ Error saving intelligence: {e}")
    
    def force_update_all(self):
        """Force update of all data sources"""
        try:
            logger.info("🔥 Forcing update of all data sources...")
            
            for source in SCRAPER_MODULES.keys():
                try:
                    logger.info(f"🔄 Force updating {source}...")
                    success = self.data_manager.update_data_source(source)
                    if success:
                        logger.info(f"✅ {source} force update completed")
                    else:
                        logger.warning(f"⚠️ {source} force update failed")
                except Exception as e:
                    logger.error(f"❌ Error force updating {source}: {e}")
            
            # Process intelligence
            self._process_intelligence()
            
            logger.info("✅ Force update completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in force update: {e}")
            return False
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'scheduler_running': self.running,
                'data_sources': {},
                'intelligence_file': {
                    'exists': os.path.exists(self.intelligence_file),
                    'age_minutes': None,
                    'size_kb': None
                },
                'next_scheduled_runs': {},
                'system_health': 'UNKNOWN'
            }
            
            # Check each data source
            for source in SCRAPER_MODULES.keys():
                status['data_sources'][source] = self.data_manager.get_source_status(source)
            
            # Check intelligence file
            if status['intelligence_file']['exists']:
                file_stat = os.stat(self.intelligence_file)
                file_time = datetime.fromtimestamp(file_stat.st_mtime)
                age_minutes = (datetime.now() - file_time).total_seconds() / 60
                status['intelligence_file']['age_minutes'] = round(age_minutes, 1)
                status['intelligence_file']['size_kb'] = round(file_stat.st_size / 1024, 1)
            
            # Get next scheduled runs
            jobs = schedule.jobs
            for job in jobs:
                job_name = str(job.job_func).split('.')[-1] if hasattr(job, 'job_func') else 'unknown'
                status['next_scheduled_runs'][job_name] = job.next_run.isoformat() if job.next_run else None
            
            # Assess overall health
            active_sources = sum(1 for s in status['data_sources'].values() 
                               if s.get('initialized') and s.get('data_fresh'))
            total_sources = len(SCRAPER_MODULES)
            
            if active_sources >= total_sources:
                status['system_health'] = 'HEALTHY'
            elif active_sources >= total_sources * 0.75:
                status['system_health'] = 'WARNING' 
            else:
                status['system_health'] = 'DEGRADED'
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Error getting system status: {e}")
            return {'error': str(e)}

class CentralFASystem:
    """Main Central FA Intelligence System"""
    
    def __init__(self):
        self.data_manager = DataSourceManager()
        self.processor = IntelligenceProcessor()
        self.scheduler = SchedulerManager(self.data_manager, self.processor)
        self.system_started = False
        
    def initialize(self):
        """Initialize the entire FA system"""
        try:
            logger.info("=" * 60)
            logger.info("CENTRAL FA INTELLIGENCE SYSTEM INITIALIZING")
            logger.info("=" * 60)
            logger.info(f"System Version: {SYSTEM_CONFIG['version']}")
            logger.info(f"Monitored Pairs: {len(MONITORED_PAIRS)}")
            logger.info(f"Data Sources: {len(SCRAPER_MODULES)}")
            logger.info("=" * 60)
            
            # Initialize data source managers
            if not self.data_manager.initialize_scrapers():
                logger.error("❌ Failed to initialize data sources")
                return False
            
            logger.info("✅ Central FA System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error initializing system: {e}")
            return False
    
    def start(self):
        """Start the FA system"""
        try:
            if not self.system_started:
                if not self.initialize():
                    return False
            
            # Start scheduler
            if not self.scheduler.start_scheduler():
                logger.error("❌ Failed to start scheduler")
                return False
            
            self.system_started = True
            logger.info("🚀 Central FA System started successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error starting system: {e}")
            return False
    
    def stop(self):
        """Stop the FA system"""
        try:
            logger.info("🛑 Stopping Central FA System...")
            
            # Stop scheduler
            self.scheduler.stop_scheduler()
            
            self.system_started = False
            logger.info("✅ Central FA System stopped")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error stopping system: {e}")
            return False
    
    def get_intelligence(self) -> Dict:
        """Get current FA intelligence"""
        try:
            if os.path.exists(SYSTEM_CONFIG['output_file']):
                with open(SYSTEM_CONFIG['output_file'], 'r') as f:
                    return json.load(f)
            else:
                logger.warning("⚠️ Intelligence file not found")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error loading intelligence: {e}")
            return None
    
    def force_update(self):
        """Force immediate update of all data"""
        try:
            logger.info("🔥 Forcing immediate FA update...")
            return self.scheduler.force_update_all()
            
        except Exception as e:
            logger.error(f"❌ Error in force update: {e}")
            return False
    
    def get_status(self):
        """Get comprehensive system status"""
        try:
            return self.scheduler.get_system_status()
            
        except Exception as e:
            logger.error(f"❌ Error getting status: {e}")
            return {'error': str(e)}
    
    def run_daemon(self):
        """Run system as daemon"""
        try:
            if not self.start():
                logger.error("❌ Failed to start FA system")
                return
            
            logger.info("🎯 Central FA System running. Press Ctrl+C to stop.")
            
            # Keep main thread alive
            while self.system_started:
                try:
                    time.sleep(300)  # Check every 5 minutes
                    
                    # Log status every hour
                    if datetime.now().minute == 0:
                        status = self.get_status()
                        health = status.get('system_health', 'UNKNOWN')
                        logger.info(f"📊 System Health: {health}")
                        
                except KeyboardInterrupt:
                    logger.info("🛑 Stop signal received")
                    break
                    
        except Exception as e:
            logger.error(f"❌ Error in daemon mode: {e}")
            
        finally:
            self.stop()

# ===== UTILITY FUNCTIONS =====

def create_sample_intelligence():
    """Create sample intelligence file for testing"""
    logger.info("📝 Creating sample FA intelligence...")
    
    sample_intelligence = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'system_version': SYSTEM_CONFIG['version'],
            'data_sources_used': ['sentiment', 'correlation', 'calendar', 'cot'],
            'processing_rules': {
                'sentiment_weight': 0.3,
                'correlation_weight': 0.25,
                'calendar_weight': 0.25,
                'cot_weight': 0.2
            }
        },
        'pair_analysis': {},
        'market_overview': {
            'timestamp': datetime.now().isoformat(),
            'market_sentiment': 'DIRECTIONAL',
            'risk_level': 'MEDIUM',
            'major_events_count': 2,
            'high_correlation_pairs': 3,
            'cot_extremes': 1
        },
        'alerts': [
            {
                'type': 'HIGH_IMPACT_EVENT',
                'severity': 'HIGH',
                'message': 'High impact event: US Federal Reserve Interest Rate Decision',
                'currency': 'USD',
                'time': '14:00',
                'timestamp': datetime.now().isoformat()
            },
            {
                'type': 'CORRELATION_WARNING',
                'severity': 'MEDIUM',
                'message': 'High correlation (85.0%) between AUDUSD and AUDCAD',
                'pair': 'AUDUSD-AUDCAD',
                'timestamp': datetime.now().isoformat()
            }
        ],
        'system_health': {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'HEALTHY',
            'data_sources': {
                'sentiment': 'ACTIVE',
                'correlation': 'ACTIVE', 
                'calendar': 'ACTIVE',
                'cot': 'ACTIVE'
            },
            'issues': []
        }
    }
    
    # Add sample pair analysis
    sample_pairs = [
        ('EURUSD', 'BULLISH', 0.75, ['High correlation with GBPUSD']),
        ('XAUUSD', 'BEARISH', 0.65, ['Strong short sentiment', 'High impact USD events']),
        ('BTCUSD', 'NEUTRAL', 0.45, ['Balanced sentiment']),
        ('GBPUSD', 'BULLISH', 0.70, ['Strong long sentiment'])
    ]
    
    for pair, signal, confidence, risks in sample_pairs:
        sample_intelligence['pair_analysis'][pair] = {
            'pair': pair,
            'timestamp': datetime.now().isoformat(),
            'sentiment_analysis': {
                'allowed_directions': ['long'] if signal == 'BULLISH' else ['short'] if signal == 'BEARISH' else ['long', 'short'],
                'blocked_directions': ['short'] if signal == 'BULLISH' else ['long'] if signal == 'BEARISH' else [],
                'sentiment_scores': {'short': 25 if signal == 'BULLISH' else 75 if signal == 'BEARISH' else 50,
                                   'long': 75 if signal == 'BULLISH' else 25 if signal == 'BEARISH' else 50},
                'signal_strength': f'Strong {signal.title()}' if signal != 'NEUTRAL' else 'Balanced'
            },
            'correlation_analysis': {
                'high_correlations': [{'symbol': 'GBPUSD', 'correlation': 72, 'strength': 'Strong Positive'}] if pair == 'EURUSD' else [],
                'correlation_warnings': [],
                'correlation_risk_level': 'HIGH' if (risks and 'correlation' in risks[0].lower()) else 'LOW'
            },
            'fundamental_events': [
                {
                    'event_name': 'GDP Growth Rate',
                    'time': '09:30',
                    'currency': pair[:3],
                    'impact': 'Medium',
                    'relevance_score': 2
                }
            ],
            'cot_analysis': {
                'available': True,
                'data_type': 'Financial',
                'latest_date': '2025-07-22',
                'net_position': 15000 if signal == 'BULLISH' else -12000 if signal == 'BEARISH' else 2000,
                'net_percentage': 15.5 if signal == 'BULLISH' else -12.8 if signal == 'BEARISH' else 2.1,
                'sentiment': signal.title() if signal != 'NEUTRAL' else 'Neutral'
            },
            'overall_signal': signal,
            'risk_factors': risks,
            'confidence_score': confidence
        }
    
    try:
        with open(SYSTEM_CONFIG['output_file'], 'w') as f:
            json.dump(sample_intelligence, f, indent=2, default=str)
        
        logger.info(f"✅ Sample intelligence created: {SYSTEM_CONFIG['output_file']}")
        logger.info(f"   Pairs analyzed: {len(sample_intelligence['pair_analysis'])}")
        logger.info(f"   Alerts generated: {len(sample_intelligence['alerts'])}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating sample intelligence: {e}")
        return False

def test_system():
    """Test the Central FA System"""
    logger.info("🧪 Testing Central FA System...")
    
    system = CentralFASystem()
    
    try:
        # Test initialization
        if not system.initialize():
            logger.error("❌ System initialization failed")
            return False
        
        # Test force update
        logger.info("🔄 Testing force update...")
        if system.force_update():
            logger.info("✅ Force update successful")
        else:
            logger.warning("⚠️ Force update failed")
        
        # Test intelligence generation
        logger.info("🧠 Testing intelligence generation...")
        intelligence = system.get_intelligence()
        
        if intelligence:
            pairs_count = len(intelligence.get('pair_analysis', {}))
            alerts_count = len(intelligence.get('alerts', []))
            logger.info(f"✅ Intelligence generated: {pairs_count} pairs, {alerts_count} alerts")
        else:
            logger.warning("⚠️ No intelligence generated")
        
        # Test status
        status = system.get_status()
        health = status.get('system_health', 'UNKNOWN')
        logger.info(f"📊 System health: {health}")
        
        logger.info("✅ System test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in system test: {e}")
        return False
    
    finally:
        system.stop()

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'test':
            test_system()
        elif command == 'sample':
            create_sample_intelligence()
        elif command == 'daemon':
            system = CentralFASystem()
            system.run_daemon()
        elif command == 'status':
            system = CentralFASystem()
            status = system.get_status()
            print(json.dumps(status, indent=2, default=str))
        elif command == 'force':
            system = CentralFASystem()
            if system.initialize():
                success = system.force_update()
                print(f"Force update: {'SUCCESS' if success else 'FAILED'}")
            system.stop()
        else:
            print("Usage: python central_fa_system.py [test|sample|daemon|status|force]")
            sys.exit(1)
    else:
        # Default: run as daemon
        system = CentralFASystem()
        system.run_daemon()

if __name__ == "__main__":
    main()