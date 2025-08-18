# ===== UNIFIED MARKET INTELLIGENCE SYSTEM =====
# Central orchestrator for all market data scrapers and intelligence fusion

import json
import logging
import threading
import time
import schedule
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import os

# ===== CONFIGURATION MANAGEMENT =====

class ComponentStatus(Enum):
    ENABLED = "enabled"
    DISABLED = "disabled"
    ERROR = "error"
    UPDATING = "updating"

@dataclass
class ScraperConfig:
    enabled: bool
    interval_minutes: Optional[int] = None
    custom_params: Dict[str, Any] = None

class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_file: str = "market_intelligence_config.json"):
        self.config_file = config_file
        self.config = self._load_config()
        self._setup_logging()
    
    def _load_config(self) -> Dict:
        """Load configuration from file or create default"""
        default_config = {
            "system": {
                "master_enabled": True,
                "log_level": "INFO",
                "log_file": "market_intelligence.log",
                "data_retention_days": 30
            },
            "scrapers": {
                "sentiment": {
                    "enabled": True,
                    "interval_minutes": 30,
                    "threshold": 60,
                    "balanced_range": [40, 60],
                    "output_file": "sentiment_signals.json",
                    "monitored_pairs": ["AUDUSD", "USDCAD", "XAUUSD", "EURUSD", "GBPUSD"]
                },
                "correlation": {
                    "enabled": True,
                    "interval_minutes": 30,
                    "high_threshold": 70,
                    "negative_threshold": -70,
                    "output_file": "correlation_data.json"
                },
                "cot": {
                    "enabled": True,
                    "update_day": "friday",
                    "update_time": "18:00",
                    "historical_weeks": 6,
                    "output_csv": "cot_consolidated_data.csv",
                    "output_json": "cot_consolidated_data.json"
                },
                "calendar": {
                    "enabled": True,
                    "update_hours": [6, 12, 18],  # Multiple updates per day
                    "days_ahead": 3,
                    "output_file": "economic_calendar.json",
                    "impact_filter": ["High", "Medium"]
                }
            },
            "intelligence": {
                "enabled": True,
                "fusion_interval_minutes": 5,
                "output_file": "market_intelligence.json",
                "risk_weights": {
                    "sentiment": 0.25,
                    "correlation": 0.30,
                    "cot": 0.25,
                    "calendar": 0.20
                },
                "signal_thresholds": {
                    "strong_signal": 0.75,
                    "moderate_signal": 0.50,
                    "weak_signal": 0.25
                }
            },
            "trading_pairs": {
                "monitored": ["AUDUSD", "USDCAD", "XAUUSD", "EURUSD", "GBPUSD", 
                             "AUDCAD", "USDCHF", "GBPCAD", "AUDNZD", "NZDCAD", "US500", "BTCUSD"],
                "priority": ["XAUUSD", "EURUSD", "GBPUSD", "AUDUSD"]
            }
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults (defaults take precedence for missing keys)
                    return self._merge_configs(default_config, loaded_config)
            else:
                self._save_config(default_config)
                return default_config
        except Exception as e:
            print(f"Error loading config: {e}, using defaults")
            return default_config
    
    def _merge_configs(self, default: Dict, loaded: Dict) -> Dict:
        """Recursively merge configurations"""
        result = default.copy()
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def _save_config(self, config: Dict):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def _setup_logging(self):
        """Setup logging based on configuration"""
        log_level = getattr(logging, self.config['system']['log_level'])
        log_file = self.config['system']['log_file']
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - MARKET_INTELLIGENCE - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def get_scraper_config(self, scraper_name: str) -> ScraperConfig:
        """Get configuration for specific scraper"""
        scraper_config = self.config['scrapers'].get(scraper_name, {})
        return ScraperConfig(
            enabled=scraper_config.get('enabled', False),
            interval_minutes=scraper_config.get('interval_minutes'),
            custom_params=scraper_config
        )
    
    def is_enabled(self, component: str) -> bool:
        """Check if component is enabled"""
        if not self.config['system']['master_enabled']:
            return False
        return self.config.get(component, {}).get('enabled', False)
    
    def update_config(self, updates: Dict):
        """Update configuration and save"""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.config, updates)
        self._save_config(self.config)

# ===== COMPONENT MANAGER =====

class ComponentManager:
    """Manages individual scraper components"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        self.components = {}
        self.component_status = {}
        self.threads = {}
        
    def register_component(self, name: str, component_class, *args, **kwargs):
        """Register a scraper component"""
        try:
            config = self.config_manager.get_scraper_config(name)
            if config.enabled:
                self.components[name] = component_class(*args, **kwargs)
                self.component_status[name] = ComponentStatus.ENABLED
                self.logger.info(f"✅ Registered component: {name}")
            else:
                self.component_status[name] = ComponentStatus.DISABLED
                self.logger.info(f"⏸️ Component disabled: {name}")
        except Exception as e:
            self.component_status[name] = ComponentStatus.ERROR
            self.logger.error(f"❌ Failed to register {name}: {e}")
    
    def start_component(self, name: str):
        """Start a component's scheduled updates"""
        if name not in self.components:
            self.logger.warning(f"⚠️ Component {name} not registered")
            return False
        
        config = self.config_manager.get_scraper_config(name)
        if not config.enabled:
            self.logger.info(f"⏸️ Component {name} is disabled")
            return False
        
        try:
            # Schedule based on component type
            if name == 'cot':
                # COT updates on specific day/time
                update_day = config.custom_params.get('update_day', 'friday')
                update_time = config.custom_params.get('update_time', '18:00')
                getattr(schedule.every(), update_day).at(update_time).do(
                    self._run_component_update, name
                )
            elif name == 'calendar':
                # Calendar updates at specific hours
                update_hours = config.custom_params.get('update_hours', [6])
                for hour in update_hours:
                    schedule.every().day.at(f"{hour:02d}:00").do(
                        self._run_component_update, name
                    )
            else:
                # Regular interval updates
                if config.interval_minutes:
                    schedule.every(config.interval_minutes).minutes.do(
                        self._run_component_update, name
                    )
            
            self.logger.info(f"🚀 Started component: {name}")
            return True
            
        except Exception as e:
            self.component_status[name] = ComponentStatus.ERROR
            self.logger.error(f"❌ Failed to start {name}: {e}")
            return False
    
    def _run_component_update(self, name: str):
        """Run update for a specific component"""
        if name not in self.components:
            return
        
        self.component_status[name] = ComponentStatus.UPDATING
        self.logger.info(f"🔄 Updating {name}...")
        
        try:
            component = self.components[name]
            
            # Call appropriate update method based on component
            if hasattr(component, 'update_sentiment_signals'):
                success = component.update_sentiment_signals()
            elif hasattr(component, 'update_correlation_data'):
                success = component.update_correlation_data()
            elif hasattr(component, 'update_cot_data'):
                success = component.update_cot_data()
            elif hasattr(component, 'update_calendar_data'):
                success = component.update_calendar_data()
            else:
                success = False
                self.logger.error(f"❌ No update method found for {name}")
            
            if success:
                self.component_status[name] = ComponentStatus.ENABLED
                self.logger.info(f"✅ {name} updated successfully")
            else:
                self.component_status[name] = ComponentStatus.ERROR
                self.logger.error(f"❌ {name} update failed")
                
        except Exception as e:
            self.component_status[name] = ComponentStatus.ERROR
            self.logger.error(f"❌ Error updating {name}: {e}")
    
    def force_update(self, name: str) -> bool:
        """Force immediate update of component"""
        self.logger.info(f"🔥 Forcing update for {name}")
        self._run_component_update(name)
        return self.component_status[name] == ComponentStatus.ENABLED
    
    def get_status(self) -> Dict[str, str]:
        """Get status of all components"""
        return {name: status.value for name, status in self.component_status.items()}

# ===== INTELLIGENCE FUSION ENGINE =====

class IntelligenceFusion:
    """Combines data from all scrapers into unified intelligence"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        self.weights = config_manager.config['intelligence']['risk_weights']
        
    def fuse_market_data(self) -> Optional[Dict]:
        """Combine all market data sources into unified intelligence"""
        try:
            self.logger.info("🧠 Starting intelligence fusion...")
            
            # Load data from all sources
            sentiment_data = self._load_sentiment_data()
            correlation_data = self._load_correlation_data()
            cot_data = self._load_cot_data()
            calendar_data = self._load_calendar_data()
            
            # Create unified intelligence
            intelligence = {
                'timestamp': datetime.now().isoformat(),
                'data_sources': {
                    'sentiment': sentiment_data is not None,
                    'correlation': correlation_data is not None,
                    'cot': cot_data is not None,
                    'calendar': calendar_data is not None
                },
                'pair_analysis': {},
                'market_overview': {},
                'risk_alerts': []
            }
            
            # Analyze each monitored pair
            monitored_pairs = self.config_manager.config['trading_pairs']['monitored']
            
            for pair in monitored_pairs:
                pair_intel = self._analyze_pair(
                    pair, sentiment_data, correlation_data, cot_data, calendar_data
                )
                if pair_intel:
                    intelligence['pair_analysis'][pair] = pair_intel
            
            # Generate market overview
            intelligence['market_overview'] = self._generate_market_overview(
                intelligence['pair_analysis']
            )
            
            # Generate risk alerts
            intelligence['risk_alerts'] = self._generate_risk_alerts(
                intelligence['pair_analysis']
            )
            
            self.logger.info(f"✅ Intelligence fusion completed for {len(intelligence['pair_analysis'])} pairs")
            return intelligence
            
        except Exception as e:
            self.logger.error(f"❌ Error in intelligence fusion: {e}")
            return None
    
    def _load_sentiment_data(self) -> Optional[Dict]:
        """Load sentiment data"""
        try:
            config = self.config_manager.config['scrapers']['sentiment']
            file_path = config.get('output_file', 'sentiment_signals.json')
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"⚠️ Could not load sentiment data: {e}")
        return None
    
    def _load_correlation_data(self) -> Optional[Dict]:
        """Load correlation data"""
        try:
            config = self.config_manager.config['scrapers']['correlation']
            file_path = config.get('output_file', 'correlation_data.json')
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"⚠️ Could not load correlation data: {e}")
        return None
    
    def _load_cot_data(self) -> Optional[Dict]:
        """Load COT data"""
        try:
            config = self.config_manager.config['scrapers']['cot']
            file_path = config.get('output_json', 'cot_consolidated_data.json')
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"⚠️ Could not load COT data: {e}")
        return None
    
    def _load_calendar_data(self) -> Optional[Dict]:
        """Load economic calendar data"""
        try:
            config = self.config_manager.config['scrapers']['calendar']
            file_path = config.get('output_file', 'economic_calendar.json')
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"⚠️ Could not load calendar data: {e}")
        return None
    
    def _analyze_pair(self, pair: str, sentiment_data: Dict, correlation_data: Dict, 
                     cot_data: Dict, calendar_data: Dict) -> Dict:
        """Analyze intelligence for a specific trading pair"""
        analysis = {
            'pair': pair,
            'overall_signal': 'NEUTRAL',
            'confidence': 0.0,
            'components': {
                'sentiment': self._analyze_sentiment_component(pair, sentiment_data),
                'correlation': self._analyze_correlation_component(pair, correlation_data),
                'cot': self._analyze_cot_component(pair, cot_data),
                'calendar': self._analyze_calendar_component(pair, calendar_data)
            },
            'recommendations': [],
            'risk_factors': []
        }
        
        # Calculate weighted overall signal
        total_weight = 0
        weighted_score = 0
        
        for component, data in analysis['components'].items():
            if data and data.get('available'):
                weight = self.weights.get(component, 0)
                score = data.get('signal_strength', 0)  # -1 to 1 scale
                weighted_score += weight * score
                total_weight += weight
        
        if total_weight > 0:
            final_score = weighted_score / total_weight
            analysis['confidence'] = abs(final_score)
            
            if final_score > 0.25:
                analysis['overall_signal'] = 'BULLISH'
            elif final_score < -0.25:
                analysis['overall_signal'] = 'BEARISH'
            else:
                analysis['overall_signal'] = 'NEUTRAL'
        
        return analysis
    
    def _analyze_sentiment_component(self, pair: str, sentiment_data: Dict) -> Dict:
        """Analyze sentiment component for pair"""
        if not sentiment_data or 'pairs' not in sentiment_data:
            return {'available': False}
        
        pair_sentiment = sentiment_data['pairs'].get(pair)
        if not pair_sentiment:
            return {'available': False}
        
        sentiment = pair_sentiment.get('sentiment', {})
        long_pct = sentiment.get('long', 50)
        short_pct = sentiment.get('short', 50)
        
        # Convert to signal strength (-1 to 1)
        signal_strength = (long_pct - short_pct) / 100
        
        return {
            'available': True,
            'signal_strength': signal_strength,
            'long_percentage': long_pct,
            'short_percentage': short_pct,
            'blocked_directions': pair_sentiment.get('blocked_directions', [])
        }
    
    def _analyze_correlation_component(self, pair: str, correlation_data: Dict) -> Dict:
        """Analyze correlation component for pair"""
        if not correlation_data or 'correlation_matrix' not in correlation_data:
            return {'available': False}
        
        warnings = correlation_data.get('warnings', [])
        pair_warnings = [w for w in warnings if pair in w.get('pair', '')]
        
        # Simple risk score based on warnings
        risk_score = len(pair_warnings) * 0.1  # Each warning adds 10% risk
        
        return {
            'available': True,
            'risk_score': min(risk_score, 1.0),
            'warnings_count': len(pair_warnings),
            'high_correlations': pair_warnings
        }
    
    def _analyze_cot_component(self, pair: str, cot_data: Dict) -> Dict:
        """Analyze COT component for pair"""
        if not cot_data or 'data' not in cot_data:
            return {'available': False}
        
        # Look for COT data related to this pair
        # This would need pair mapping logic
        return {'available': False}  # Placeholder
    
    def _analyze_calendar_component(self, pair: str, calendar_data: Dict) -> Dict:
        """Analyze economic calendar component for pair"""
        if not calendar_data or 'events' not in calendar_data:
            return {'available': False}
        
        # Analyze upcoming events that might affect this pair
        # This would need currency mapping logic
        return {'available': False}  # Placeholder
    
    def _generate_market_overview(self, pair_analyses: Dict) -> Dict:
        """Generate overall market overview"""
        if not pair_analyses:
            return {}
        
        signals = [analysis['overall_signal'] for analysis in pair_analyses.values()]
        
        overview = {
            'total_pairs': len(pair_analyses),
            'bullish_count': signals.count('BULLISH'),
            'bearish_count': signals.count('BEARISH'),
            'neutral_count': signals.count('NEUTRAL'),
            'market_sentiment': 'NEUTRAL'
        }
        
        if overview['bullish_count'] > overview['bearish_count']:
            overview['market_sentiment'] = 'BULLISH'
        elif overview['bearish_count'] > overview['bullish_count']:
            overview['market_sentiment'] = 'BEARISH'
        
        return overview
    
    def _generate_risk_alerts(self, pair_analyses: Dict) -> List[Dict]:
        """Generate risk alerts based on analysis"""
        alerts = []
        
        for pair, analysis in pair_analyses.items():
            # High correlation warnings
            correlation = analysis['components'].get('correlation', {})
            if correlation.get('warnings_count', 0) > 2:
                alerts.append({
                    'type': 'HIGH_CORRELATION_RISK',
                    'pair': pair,
                    'severity': 'MEDIUM',
                    'message': f"{pair} has {correlation['warnings_count']} correlation warnings"
                })
        
        return alerts

# ===== MAIN ORCHESTRATOR =====

class MarketIntelligenceOrchestrator:
    """Main orchestrator for the unified market intelligence system"""
    
    def __init__(self, config_file: str = "market_intelligence_config.json"):
        self.config_manager = ConfigManager(config_file)
        self.component_manager = ComponentManager(self.config_manager)
        self.intelligence_fusion = IntelligenceFusion(self.config_manager)
        self.logger = logging.getLogger(__name__)
        self.running = False
        
    def initialize(self):
        """Initialize all components"""
        self.logger.info("🚀 Initializing Market Intelligence System...")
        
        # Import scraper classes dynamically to avoid import errors
        try:
            # These would import your existing scraper classes
            from sentiment_scraper import SentimentSignalManager
            from correlation_scraper import CorrelationSignalManager
            from cot_scraper import COTDataManager
            from calendar_scraper import FixedEconomicCalendarScraper
            
            # Register components
            self.component_manager.register_component('sentiment', SentimentSignalManager)
            self.component_manager.register_component('correlation', CorrelationSignalManager)
            self.component_manager.register_component('cot', COTDataManager)
            self.component_manager.register_component('calendar', FixedEconomicCalendarScraper)
            
        except ImportError as e:
            self.logger.error(f"❌ Could not import scraper classes: {e}")
            return False
        
        # Schedule intelligence fusion
        if self.config_manager.is_enabled('intelligence'):
            fusion_interval = self.config_manager.config['intelligence']['fusion_interval_minutes']
            schedule.every(fusion_interval).minutes.do(self._run_intelligence_fusion)
        
        self.logger.info("✅ Market Intelligence System initialized")
        return True
    
    def start(self):
        """Start the orchestrator"""
        if not self.initialize():
            self.logger.error("❌ Failed to initialize, cannot start")
            return False
        
        self.logger.info("🚀 Starting Market Intelligence System...")
        
        # Start all enabled components
        for component_name in self.component_manager.components.keys():
            self.component_manager.start_component(component_name)
        
        self.running = True
        
        # Start scheduler thread
        scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        scheduler_thread.start()
        
        # Run initial intelligence fusion
        self._run_intelligence_fusion()
        
        self.logger.info("✅ Market Intelligence System started")
        return True
    
    def stop(self):
        """Stop the orchestrator"""
        self.logger.info("🛑 Stopping Market Intelligence System...")
        self.running = False
        schedule.clear()
        self.logger.info("✅ Market Intelligence System stopped")
    
    def _run_scheduler(self):
        """Run the scheduler in background thread"""
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"❌ Scheduler error: {e}")
                time.sleep(5)
    
    def _run_intelligence_fusion(self):
        """Run intelligence fusion"""
        try:
            self.logger.info("🧠 Running intelligence fusion...")
            intelligence = self.intelligence_fusion.fuse_market_data()
            
            if intelligence:
                # Save intelligence output
                output_file = self.config_manager.config['intelligence']['output_file']
                with open(output_file, 'w') as f:
                    json.dump(intelligence, f, indent=2, default=str)
                
                self.logger.info(f"✅ Intelligence saved to {output_file}")
            else:
                self.logger.warning("⚠️ Intelligence fusion produced no output")
                
        except Exception as e:
            self.logger.error(f"❌ Intelligence fusion error: {e}")
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        return {
            'system_running': self.running,
            'master_enabled': self.config_manager.config['system']['master_enabled'],
            'components': self.component_manager.get_status(),
            'intelligence_enabled': self.config_manager.is_enabled('intelligence'),
            'timestamp': datetime.now().isoformat()
        }
    
    def force_update_all(self):
        """Force update of all components"""
        self.logger.info("🔥 Forcing update of all components...")
        for component_name in self.component_manager.components.keys():
            self.component_manager.force_update(component_name)
        
        # Run intelligence fusion after updates
        time.sleep(2)  # Give components time to complete
        self._run_intelligence_fusion()

# ===== MAIN EXECUTION =====

def main():
    """Main execution function"""
    orchestrator = MarketIntelligenceOrchestrator()
    
    try:
        if orchestrator.start():
            print("🎯 Market Intelligence System is running. Press Ctrl+C to stop.")
            
            while orchestrator.running:
                try:
                    time.sleep(60)  # Check every minute
                    
                    # Log status every 10 minutes
                    if datetime.now().minute % 10 == 0:
                        status = orchestrator.get_system_status()
                        print(f"📊 System Status: {status['components']}")
                        
                except KeyboardInterrupt:
                    print("\n🛑 Stop signal received")
                    break
        else:
            print("❌ Failed to start Market Intelligence System")
            
    except Exception as e:
        print(f"❌ System error: {e}")
        
    finally:
        orchestrator.stop()
        print("✅ Market Intelligence System stopped")

if __name__ == "__main__":
    main()