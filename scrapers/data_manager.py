#!/usr/bin/env python3
# ===== UNIFIED DATA MANAGER - PHASE 2 =====
# Integrates all scrapers into a coordinated data collection system
# Manages economic calendar, sentiment, correlation, and COT data

import pandas as pd  # FIX: Add this import at the top
import os
import sys
import json
import time
import threading
import schedule
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import importlib.util

# Add scrapers to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scrapers'))

class DataManager:
    """Unified data collection and management system"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        
        # Data management
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Unified data file
        self.market_data_file = self.data_dir / "market_data.json"
        
        # Component status
        self.component_status = {}
        self.data_lock = threading.Lock()
        self.shutdown_event = threading.Event()
        
        # Scraper instances
        self.scrapers = {}
        self.schedulers = {}
        
        # Data freshness tracking
        self.last_updates = {}
        
        # Initialize components
        self._initialize_scrapers()
        self._initialize_schedules()
        
        self.logger.info("🎯 Data Manager initialized")
    
    def _initialize_scrapers(self):
        """Initialize all scraper components"""
        try:
            # Economic Calendar Scraper
            self._load_scraper('calendar', 'calendar_scraper', 'FixedEconomicCalendarScraper')
            
            # Sentiment Scraper
            self._load_scraper('sentiment', 'sentiment_scraper', 'SentimentSignalManager')
            
            # Correlation Scraper
            self._load_scraper('correlation', 'correlation_scraper', 'CorrelationSignalManager')
            
            # COT Scraper
            self._load_scraper('cot', 'cot_scraper', 'COTDataManager')
            
            self.logger.info(f"✅ Initialized {len(self.scrapers)} scrapers")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing scrapers: {e}")
    
    def _load_scraper(self, name: str, module_name: str, class_name: str):
        """Load a scraper module and class"""
        try:
            # Check if module file exists
            module_path = Path("scrapers") / f"{module_name}.py"
            if not module_path.exists():
                self.logger.warning(f"⚠️ Scraper not found: {module_path}")
                return False
            
            # Import module dynamically
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the class
            if hasattr(module, class_name):
                scraper_class = getattr(module, class_name)
                
                # Initialize scraper (different init patterns)
                if name == 'calendar':
                    scraper_instance = scraper_class()
                else:
                    scraper_instance = scraper_class()
                
                self.scrapers[name] = scraper_instance
                self.component_status[name] = {'status': 'loaded', 'last_update': None, 'error': None}
                
                self.logger.info(f"✅ Loaded {name} scraper: {class_name}")
                return True
            else:
                self.logger.error(f"❌ Class {class_name} not found in {module_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error loading {name} scraper: {e}")
            self.component_status[name] = {'status': 'error', 'last_update': None, 'error': str(e)}
            return False
    
    def _initialize_schedules(self):
        """Initialize update schedules for all data sources"""
        try:
            schedules_config = self.config.schedules.get('data_collection', {})
            
            # Economic Calendar - every hour
            calendar_config = schedules_config.get('economic_calendar', {})
            if calendar_config.get('enabled', True):
                interval = calendar_config.get('interval_minutes', 60)
                schedule.every(interval).minutes.do(self._update_economic_calendar)
                self.logger.info(f"📅 Calendar updates: every {interval} minutes")
            
            # Sentiment - every 30 minutes
            sentiment_config = schedules_config.get('sentiment', {})
            if sentiment_config.get('enabled', True):
                interval = sentiment_config.get('interval_minutes', 30)
                schedule.every(interval).minutes.do(self._update_sentiment)
                self.logger.info(f"😊 Sentiment updates: every {interval} minutes")
            
            # Correlation - every 30 minutes
            correlation_config = schedules_config.get('correlation', {})
            if correlation_config.get('enabled', True):
                interval = correlation_config.get('interval_minutes', 30)
                schedule.every(interval).minutes.do(self._update_correlation)
                self.logger.info(f"🔗 Correlation updates: every {interval} minutes")
            
            # COT - weekly on Friday
            cot_config = schedules_config.get('cot', {})
            if cot_config.get('enabled', True):
                update_day = cot_config.get('update_day', 'friday')
                update_time = cot_config.get('update_time', '18:00')
                getattr(schedule.every(), update_day).at(update_time).do(self._update_cot)
                self.logger.info(f"📊 COT updates: {update_day} at {update_time}")
            
            self.logger.info("✅ Data collection schedules initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing schedules: {e}")
    
    def run(self):
        """Main data manager loop"""
        try:
            self.logger.info("🚀 Data Manager starting...")
            
            # Initialize market data file
            self._initialize_market_data_file()
            
            # Run initial data collection
            self._run_initial_collection()
            
            # Start scheduler thread
            scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
            scheduler_thread.start()
            
            # Main monitoring loop
            while not self.shutdown_event.is_set():
                try:
                    # Update system status
                    self._update_system_status()
                    
                    # Check data freshness
                    self._check_data_freshness()
                    
                    # Log status periodically
                    if datetime.now().minute % 10 == 0:  # Every 10 minutes
                        self._log_status_summary()
                    
                    # Sleep before next check
                    time.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    self.logger.error(f"❌ Error in data manager loop: {e}")
                    time.sleep(10)
            
        except Exception as e:
            self.logger.error(f"❌ Fatal error in data manager: {e}")
        
        finally:
            self.logger.info("🛑 Data Manager stopped")
    
    def _initialize_market_data_file(self):
        """Initialize the unified market data file"""
        try:
            initial_data = {
                "metadata": {
                    "system_name": "Unified Trading System",
                    "version": self.config.get('system.version', '1.0'),
                    "created": datetime.now().isoformat(),
                    "account_number": self.config.get('mt5.account_number'),
                    "magic_number": self.config.get('mt5.magic_number')
                },
                "last_updated": datetime.now().isoformat(),
                "system_status": "initializing",
                "data_freshness": {
                    "economic_calendar": {"fresh": False, "last_update": None, "age_minutes": None},
                    "sentiment": {"fresh": False, "last_update": None, "age_minutes": None},
                    "correlation": {"fresh": False, "last_update": None, "age_minutes": None},
                    "cot": {"fresh": False, "last_update": None, "age_minutes": None}
                },
                "data_sources": {
                    "economic_calendar": {
                        "status": "waiting",
                        "next_update": None,
                        "events": [],
                        "error": None,
                        "events_count": 0
                    },
                    "sentiment": {
                        "status": "waiting", 
                        "next_update": None,
                        "pairs": {},
                        "threshold": 60,
                        "error": None,
                        "pairs_count": 0
                    },
                    "correlation": {
                        "status": "waiting",
                        "next_update": None,
                        "matrix": {},
                        "warnings": [],
                        "error": None,
                        "currencies_count": 0
                    },
                    "cot": {
                        "status": "waiting",
                        "next_update": None,
                        "financial": {},
                        "commodity": {},
                        "error": None,
                        "records_count": 0
                    }
                },
                "trading_signals": {
                    "generated": [],
                    "executed": [],
                    "last_analysis": None
                },
                "risk_status": {
                    "emergency_stop": False,
                    "warnings": [],
                    "last_check": None
                }
            }
            
            # Load existing data or create new
            if self.market_data_file.exists():
                try:
                    with open(self.market_data_file, 'r') as f:
                        existing_data = json.load(f)
                    
                    # Merge with initial structure (preserve existing data)
                    self._merge_data_structure(existing_data, initial_data)
                    self.logger.info("📁 Loaded existing market data file")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Error loading existing data, creating new: {e}")
                    self._save_market_data(initial_data)
            else:
                self._save_market_data(initial_data)
                self.logger.info("📄 Created new market data file")
                
        except Exception as e:
            self.logger.error(f"❌ Error initializing market data file: {e}")
    
    def _merge_data_structure(self, existing: Dict, template: Dict) -> Dict:
        """Merge existing data with template structure"""
        try:
            # Update metadata while preserving created timestamp
            if 'metadata' in existing:
                template['metadata'].update(existing['metadata'])
                # Don't overwrite created timestamp
                if 'created' in existing['metadata']:
                    template['metadata']['created'] = existing['metadata']['created']
            
            # Preserve data sources if they exist
            if 'data_sources' in existing:
                for source_name, source_data in existing['data_sources'].items():
                    if source_name in template['data_sources']:
                        template['data_sources'][source_name].update(source_data)
            
            # Preserve freshness data
            if 'data_freshness' in existing:
                for source_name, freshness_data in existing['data_freshness'].items():
                    if source_name in template['data_freshness']:
                        template['data_freshness'][source_name].update(freshness_data)
            
            # Save merged data
            self._save_market_data(template)
            
        except Exception as e:
            self.logger.error(f"❌ Error merging data structure: {e}")
    
    def _save_market_data(self, data: Dict):
        """Thread-safe save of market data"""
        try:
            with self.data_lock:
                data['last_updated'] = datetime.now().isoformat()
                
                # Create backup
                if self.market_data_file.exists():
                    backup_file = self.market_data_file.with_suffix('.json.backup')
                    try:
                        import shutil
                        shutil.copy2(self.market_data_file, backup_file)
                    except Exception:
                        pass  # Backup not critical
                
                # Write new data
                with open(self.market_data_file, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                    
        except Exception as e:
            self.logger.error(f"❌ Error saving market data: {e}")
    
    def _load_market_data(self) -> Dict:
        """Thread-safe load of market data"""
        try:
            with self.data_lock:
                if self.market_data_file.exists():
                    with open(self.market_data_file, 'r') as f:
                        return json.load(f)
                else:
                    return {}
        except Exception as e:
            self.logger.error(f"❌ Error loading market data: {e}")
            return {}
    
    def _run_initial_collection(self):
        """Run initial data collection for all sources"""
        try:
            self.logger.info("🔄 Running initial data collection...")
            
            # Run each collector in separate thread to avoid blocking
            collectors = [
                ('economic_calendar', self._update_economic_calendar),
                ('sentiment', self._update_sentiment),
                ('correlation', self._update_correlation),
            ]
            
            threads = []
            for name, collector in collectors:
                thread = threading.Thread(
                    target=self._safe_collector_run,
                    args=(name, collector),
                    daemon=True
                )
                thread.start()
                threads.append(thread)
            
            # Wait for initial collection with timeout
            for thread in threads:
                thread.join(timeout=60)  # 1 minute timeout per collector
            
            self.logger.info("✅ Initial data collection completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error in initial collection: {e}")
    
    def _safe_collector_run(self, name: str, collector_func):
        """Safe wrapper for collector functions"""
        try:
            self.logger.info(f"🔄 Running initial {name} collection...")
            collector_func()
            self.logger.info(f"✅ Initial {name} collection completed")
        except Exception as e:
            self.logger.error(f"❌ Error in initial {name} collection: {e}")
    
    def _run_scheduler(self):
        """Run the scheduling loop"""
        self.logger.info("⏰ Data collection scheduler started")
        
        while not self.shutdown_event.is_set():
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"❌ Error in scheduler: {e}")
                time.sleep(10)
        
        self.logger.info("⏰ Data collection scheduler stopped")
    
    # ===== DATA COLLECTION METHODS =====
    
    def _update_economic_calendar(self):
        """Update economic calendar data"""
        try:
            self.logger.info("📅 Updating economic calendar data...")
            
            if 'calendar' not in self.scrapers:
                self.logger.warning("⚠️ Calendar scraper not available")
                return
            
            # Update component status
            self._update_component_status('economic_calendar', 'updating')
            
            # Generate target dates (next 3 trading days)
            target_dates = self._generate_calendar_dates()
            
            # Scrape data
            scraper = self.scrapers['calendar']
            events = scraper.scrape_calendar_data(target_dates)
            
            if events:
                # Update market data
                market_data = self._load_market_data()
                market_data['data_sources']['economic_calendar'] = {
                    'status': 'fresh',
                    'last_update': datetime.now().isoformat(),
                    'events': events,
                    'events_count': len(events),
                    'target_dates': target_dates,
                    'error': None
                }
                
                # Update freshness
                market_data['data_freshness']['economic_calendar'] = {
                    'fresh': True,
                    'last_update': datetime.now().isoformat(),
                    'age_minutes': 0
                }
                
                self._save_market_data(market_data)
                self._update_component_status('economic_calendar', 'fresh')
                
                self.logger.info(f"✅ Calendar updated: {len(events)} events for {len(target_dates)} dates")
                
            else:
                self._update_component_status('economic_calendar', 'error', 'No events scraped')
                self.logger.warning("⚠️ No calendar events scraped")
                
        except Exception as e:
            self.logger.error(f"❌ Error updating calendar: {e}")
            self._update_component_status('economic_calendar', 'error', str(e))
    
    def _update_sentiment(self):
        """Update sentiment data"""
        try:
            self.logger.info("😊 Updating sentiment data...")
            
            if 'sentiment' not in self.scrapers:
                self.logger.warning("⚠️ Sentiment scraper not available")
                return
            
            self._update_component_status('sentiment', 'updating')
            
            # Update sentiment signals
            scraper = self.scrapers['sentiment']
            success = scraper.update_sentiment_signals()
            
            if success:
                # Load the generated signals file
                signals_file = Path("sentiment_signals.json")
                if signals_file.exists():
                    with open(signals_file, 'r') as f:
                        signals_data = json.load(f)
                    
                    # Update market data
                    market_data = self._load_market_data()
                    market_data['data_sources']['sentiment'] = {
                        'status': 'fresh',
                        'last_update': datetime.now().isoformat(),
                        'pairs': signals_data.get('pairs', {}),
                        'pairs_count': len(signals_data.get('pairs', {})),
                        'threshold': signals_data.get('threshold_used', 60),
                        'data_source': signals_data.get('data_source', 'MyFXBook'),
                        'error': None
                    }
                    
                    # Update freshness
                    market_data['data_freshness']['sentiment'] = {
                        'fresh': True,
                        'last_update': datetime.now().isoformat(),
                        'age_minutes': 0
                    }
                    
                    self._save_market_data(market_data)
                    self._update_component_status('sentiment', 'fresh')
                    
                    pairs_count = len(signals_data.get('pairs', {}))
                    self.logger.info(f"✅ Sentiment updated: {pairs_count} pairs processed")
                else:
                    self._update_component_status('sentiment', 'error', 'Signals file not found')
            else:
                self._update_component_status('sentiment', 'error', 'Update failed')
                
        except Exception as e:
            self.logger.error(f"❌ Error updating sentiment: {e}")
            self._update_component_status('sentiment', 'error', str(e))
    
    def _update_correlation(self):
        """Update correlation data"""
        try:
            self.logger.info("🔗 Updating correlation data...")
            
            if 'correlation' not in self.scrapers:
                self.logger.warning("⚠️ Correlation scraper not available")
                return
            
            self._update_component_status('correlation', 'updating')
            
            # Update correlation data
            scraper = self.scrapers['correlation']
            success = scraper.update_correlation_data()
            
            if success:
                # Load the generated correlation file
                correlation_file = Path("correlation_data.json")
                if correlation_file.exists():
                    with open(correlation_file, 'r') as f:
                        correlation_data = json.load(f)
                    
                    # Update market data
                    market_data = self._load_market_data()
                    matrix = correlation_data.get('correlation_matrix', {})
                    warnings = correlation_data.get('warnings', [])
                    
                    market_data['data_sources']['correlation'] = {
                        'status': 'fresh',
                        'last_update': datetime.now().isoformat(),
                        'matrix': matrix,
                        'warnings': warnings,
                        'currencies_count': len(matrix),
                        'warnings_count': len(warnings),
                        'data_source': correlation_data.get('data_source', 'MyFXBook'),
                        'error': None
                    }
                    
                    # Update freshness
                    market_data['data_freshness']['correlation'] = {
                        'fresh': True,
                        'last_update': datetime.now().isoformat(),
                        'age_minutes': 0
                    }
                    
                    self._save_market_data(market_data)
                    self._update_component_status('correlation', 'fresh')
                    
                    self.logger.info(f"✅ Correlation updated: {len(matrix)} currencies, {len(warnings)} warnings")
                else:
                    self._update_component_status('correlation', 'error', 'Correlation file not found')
            else:
                self._update_component_status('correlation', 'error', 'Update failed')
                
        except Exception as e:
            self.logger.error(f"❌ Error updating correlation: {e}")
            self._update_component_status('correlation', 'error', str(e))
    
    def _update_cot(self):
        """Update COT data"""
        try:
            self.logger.info("📊 Updating COT data...")
            
            if 'cot' not in self.scrapers:
                self.logger.warning("⚠️ COT scraper not available")
                return
            
            self._update_component_status('cot', 'updating')
            
            # Update COT data
            scraper = self.scrapers['cot']
            success = scraper.update_cot_data()
            
            if success:
                # Load the generated COT data
                cot_data = scraper.load_data()
                if cot_data:
                    # Update market data
                    market_data = self._load_market_data()
                    
                    # FIX: Handle pandas DataFrame safely
                    financial_count = 0
                    commodity_count = 0
                    
                    if 'Financial' in cot_data:
                        financial_df = cot_data['Financial']
                        if hasattr(financial_df, '__len__'):  # Check if it has length
                            financial_count = len(financial_df)
                    
                    if 'Commodity' in cot_data:
                        commodity_df = cot_data['Commodity']
                        if hasattr(commodity_df, '__len__'):  # Check if it has length
                            commodity_count = len(commodity_df)
                    
                    market_data['data_sources']['cot'] = {
                        'status': 'fresh',
                        'last_update': datetime.now().isoformat(),
                        'financial': {'record_count': financial_count},
                        'commodity': {'record_count': commodity_count},
                        'records_count': financial_count + commodity_count,
                        'error': None
                    }
                    
                    # Update freshness
                    market_data['data_freshness']['cot'] = {
                        'fresh': True,
                        'last_update': datetime.now().isoformat(),
                        'age_minutes': 0
                    }
                    
                    self._save_market_data(market_data)
                    self._update_component_status('cot', 'fresh')
                    
                    self.logger.info(f"✅ COT updated: {financial_count} financial + {commodity_count} commodity records")
                else:
                    self._update_component_status('cot', 'error', 'COT data not found')
            else:
                self._update_component_status('cot', 'error', 'Update failed')
                
        except Exception as e:
            self.logger.error(f"❌ Error updating COT: {e}")
            self._update_component_status('cot', 'error', str(e))
    
    # ===== UTILITY METHODS =====
    
    def _generate_calendar_dates(self, days_ahead: int = 3) -> List[str]:
        """Generate calendar dates for the next N trading days"""
        try:
            dates = []
            current_date = datetime.now()
            
            days_added = 0
            while days_added < days_ahead:
                # Skip weekends
                if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                    date_str = current_date.strftime("%A, %b %d, %Y")
                    dates.append(date_str)
                    days_added += 1
                
                current_date += timedelta(days=1)
            
            return dates
            
        except Exception as e:
            self.logger.error(f"❌ Error generating calendar dates: {e}")
            return []
    
    def _update_component_status(self, component: str, status: str, error: str = None):
        """Update component status"""
        self.component_status[component] = {
            'status': status,
            'last_update': datetime.now().isoformat(),
            'error': error
        }
    
    def _check_data_freshness(self):
        """Check and update data freshness for all sources"""
        try:
            market_data = self._load_market_data()
            freshness_data = market_data.get('data_freshness', {})
            
            # Freshness limits (in minutes)
            limits = {
                'economic_calendar': self.config.schedules.get('data_collection', {}).get('economic_calendar', {}).get('interval_minutes', 60),
                'sentiment': self.config.schedules.get('data_collection', {}).get('sentiment', {}).get('interval_minutes', 30),
                'correlation': self.config.schedules.get('data_collection', {}).get('correlation', {}).get('interval_minutes', 30),
                'cot': 7 * 24 * 60  # 1 week for COT
            }
            
            current_time = datetime.now()
            updated = False
            
            for source, limit_minutes in limits.items():
                if source in freshness_data:
                    last_update_str = freshness_data[source].get('last_update')
                    
                    if last_update_str:
                        try:
                            last_update = datetime.fromisoformat(last_update_str)
                            age_minutes = (current_time - last_update).total_seconds() / 60
                            
                            # Update freshness status
                            is_fresh = age_minutes < limit_minutes
                            
                            if (freshness_data[source].get('fresh') != is_fresh or 
                                abs(freshness_data[source].get('age_minutes', 0) - age_minutes) > 1):
                                
                                freshness_data[source]['fresh'] = is_fresh
                                freshness_data[source]['age_minutes'] = round(age_minutes, 1)
                                updated = True
                        except ValueError:
                            # Invalid timestamp
                            freshness_data[source]['fresh'] = False
                            freshness_data[source]['age_minutes'] = None
                            updated = True
                    else:
                        # No last update
                        freshness_data[source]['fresh'] = False
                        freshness_data[source]['age_minutes'] = None
            
            # Save if updated
            if updated:
                market_data['data_freshness'] = freshness_data
                self._save_market_data(market_data)
                
        except Exception as e:
            self.logger.error(f"❌ Error checking data freshness: {e}")
    
    def _update_system_status(self):
        """Update overall system status"""
        try:
            market_data = self._load_market_data()
            freshness_data = market_data.get('data_freshness', {})
            
            # Count fresh sources
            fresh_count = sum(1 for source_data in freshness_data.values() 
                            if source_data.get('fresh', False))
            total_count = len(freshness_data)
            
            # Determine system status
            if fresh_count == total_count and total_count > 0:
                system_status = "optimal"
            elif fresh_count > total_count * 0.5:
                system_status = "good"
            elif fresh_count > 0:
                system_status = "partial"
            else:
                system_status = "degraded"
            
            # Update if changed
            if market_data.get('system_status') != system_status:
                market_data['system_status'] = system_status
                self._save_market_data(market_data)
                
        except Exception as e:
            self.logger.error(f"❌ Error updating system status: {e}")
    
    def _log_status_summary(self):
        """Log periodic status summary"""
        try:
            market_data = self._load_market_data()
            
            # System status
            system_status = market_data.get('system_status', 'unknown')
            
            # Data freshness summary
            freshness_data = market_data.get('data_freshness', {})
            fresh_sources = [name for name, data in freshness_data.items() 
                           if data.get('fresh', False)]
            stale_sources = [name for name, data in freshness_data.items() 
                           if not data.get('fresh', False)]
            
            self.logger.info(f"📊 System Status: {system_status.upper()}")
            if fresh_sources:
                self.logger.info(f"   ✅ Fresh: {', '.join(fresh_sources)}")
            if stale_sources:
                self.logger.info(f"   ⚠️ Stale: {', '.join(stale_sources)}")
            
            # Data source summary
            data_sources = market_data.get('data_sources', {})
            for source_name, source_data in data_sources.items():
                status = source_data.get('status', 'unknown')
                last_update = source_data.get('last_update')
                
                if last_update:
                    try:
                        update_time = datetime.fromisoformat(last_update)
                        age = datetime.now() - update_time
                        age_str = f"{age.seconds // 60}m ago" if age.days == 0 else f"{age.days}d ago"
                    except:
                        age_str = "unknown"
                else:
                    age_str = "never"
                
                # Get count information
                count_info = ""
                if source_name == 'economic_calendar':
                    count = source_data.get('events_count', 0)
                    count_info = f" ({count} events)"
                elif source_name == 'sentiment':
                    count = source_data.get('pairs_count', 0)
                    count_info = f" ({count} pairs)"
                elif source_name == 'correlation':
                    count = source_data.get('currencies_count', 0)
                    warnings_count = source_data.get('warnings_count', 0)
                    count_info = f" ({count} currencies, {warnings_count} warnings)"
                elif source_name == 'cot':
                    count = source_data.get('records_count', 0)
                    count_info = f" ({count} records)"
                
                self.logger.info(f"   📊 {source_name}: {status} - {age_str}{count_info}")
                
        except Exception as e:
            self.logger.error(f"❌ Error logging status summary: {e}")
    
    # ===== PUBLIC API METHODS =====
    
    def get_market_data(self) -> Dict:
        """Get current market data (thread-safe)"""
        return self._load_market_data()
    
    def get_economic_calendar(self) -> List[Dict]:
        """Get current economic calendar events"""
        try:
            market_data = self._load_market_data()
            return market_data.get('data_sources', {}).get('economic_calendar', {}).get('events', [])
        except Exception as e:
            self.logger.error(f"❌ Error getting calendar: {e}")
            return []
    
    def get_sentiment_signals(self) -> Dict:
        """Get current sentiment signals"""
        try:
            market_data = self._load_market_data()
            return market_data.get('data_sources', {}).get('sentiment', {}).get('pairs', {})
        except Exception as e:
            self.logger.error(f"❌ Error getting sentiment: {e}")
            return {}
    
    def get_correlation_data(self) -> Dict:
        """Get current correlation data"""
        try:
            market_data = self._load_market_data()
            correlation_source = market_data.get('data_sources', {}).get('correlation', {})
            return {
                'matrix': correlation_source.get('matrix', {}),
                'warnings': correlation_source.get('warnings', [])
            }
        except Exception as e:
            self.logger.error(f"❌ Error getting correlation: {e}")
            return {'matrix': {}, 'warnings': []}
    
    def get_cot_data(self) -> Dict:
        """Get current COT data"""
        try:
            market_data = self._load_market_data()
            cot_source = market_data.get('data_sources', {}).get('cot', {})
            return {
                'financial': cot_source.get('financial', {}),
                'commodity': cot_source.get('commodity', {})
            }
        except Exception as e:
            self.logger.error(f"❌ Error getting COT data: {e}")
            return {'financial': {}, 'commodity': {}}
    
    def is_data_fresh(self, source: str, max_age_minutes: int = None) -> bool:
        """Check if data source is fresh"""
        try:
            market_data = self._load_market_data()
            freshness_data = market_data.get('data_freshness', {})
            
            if source not in freshness_data:
                return False
            
            source_data = freshness_data[source]
            
            if max_age_minutes is None:
                return source_data.get('fresh', False)
            else:
                age_minutes = source_data.get('age_minutes', float('inf'))
                return age_minutes is not None and age_minutes <= max_age_minutes
                
        except Exception as e:
            self.logger.error(f"❌ Error checking freshness for {source}: {e}")
            return False
    
    def force_update(self, source: str = None) -> bool:
        """Force update of specific source or all sources"""
        try:
            if source is None:
                # Update all sources
                self.logger.info("🔥 Forcing update of all data sources...")
                
                success_count = 0
                total_count = 4
                
                if self._safe_update_wrapper('economic_calendar', self._update_economic_calendar):
                    success_count += 1
                
                if self._safe_update_wrapper('sentiment', self._update_sentiment):
                    success_count += 1
                
                if self._safe_update_wrapper('correlation', self._update_correlation):
                    success_count += 1
                
                if self._safe_update_wrapper('cot', self._update_cot):
                    success_count += 1
                
                self.logger.info(f"✅ Force update completed: {success_count}/{total_count} sources updated")
                return success_count > 0
                
            else:
                # Update specific source - FIX: Handle different source name formats
                self.logger.info(f"🔥 Forcing update of {source}...")
                
                # Normalize source names
                source_mapping = {
                    'calendar': 'economic_calendar',
                    'economic_calendar': 'economic_calendar',
                    'sentiment': 'sentiment',
                    'correlation': 'correlation',
                    'cot': 'cot'
                }
                
                normalized_source = source_mapping.get(source.lower())
                if not normalized_source:
                    self.logger.error(f"❌ Unknown data source: {source}. Available: {list(source_mapping.keys())}")
                    return False
                
                update_methods = {
                    'economic_calendar': self._update_economic_calendar,
                    'sentiment': self._update_sentiment,
                    'correlation': self._update_correlation,
                    'cot': self._update_cot
                }
                
                return self._safe_update_wrapper(normalized_source, update_methods[normalized_source])
                    
        except Exception as e:
            self.logger.error(f"❌ Error in force update: {e}")
            return False
    
    def _safe_update_wrapper(self, source_name: str, update_method) -> bool:
        """Safe wrapper for update methods"""
        try:
            update_method()
            return True
        except Exception as e:
            self.logger.error(f"❌ Error updating {source_name}: {e}")
            return False
    
    def get_system_health(self) -> Dict:
        """Get comprehensive system health information"""
        try:
            market_data = self._load_market_data()
            
            # Calculate health metrics
            freshness_data = market_data.get('data_freshness', {})
            fresh_count = sum(1 for data in freshness_data.values() if data.get('fresh', False))
            total_count = len(freshness_data)
            
            health_score = (fresh_count / total_count * 100) if total_count > 0 else 0
            
            # Get component statuses
            component_health = {}
            for component, status_data in self.component_status.items():
                component_health[component] = {
                    'status': status_data.get('status', 'unknown'),
                    'last_update': status_data.get('last_update'),
                    'error': status_data.get('error'),
                    'healthy': status_data.get('status') in ['loaded', 'fresh']
                }
            
            # Get next scheduled updates
            next_updates = {}
            jobs = schedule.jobs
            for job in jobs:
                job_name = str(job.job_func).split('.')[-1].replace('_update_', '')
                next_updates[job_name] = job.next_run.isoformat() if job.next_run else None
            
            return {
                'overall_status': market_data.get('system_status', 'unknown'),
                'health_score': round(health_score, 1),
                'fresh_sources': fresh_count,
                'total_sources': total_count,
                'component_health': component_health,
                'data_freshness': freshness_data,
                'next_updates': next_updates,
                'last_updated': market_data.get('last_updated'),
                'uptime': self._calculate_uptime()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting system health: {e}")
            return {'error': str(e)}
    
    def _calculate_uptime(self) -> str:
        """Calculate system uptime"""
        try:
            market_data = self._load_market_data()
            created_str = market_data.get('metadata', {}).get('created')
            
            if created_str:
                created_time = datetime.fromisoformat(created_str)
                uptime = datetime.now() - created_time
                
                days = uptime.days
                hours, remainder = divmod(uptime.seconds, 3600)
                minutes, _ = divmod(remainder, 60)
                
                if days > 0:
                    return f"{days}d {hours}h {minutes}m"
                elif hours > 0:
                    return f"{hours}h {minutes}m"
                else:
                    return f"{minutes}m"
            else:
                return "unknown"
                
        except Exception:
            return "unknown"
    
    def create_status_report(self) -> Dict:
        """Create detailed status report for monitoring"""
        try:
            market_data = self._load_market_data()
            health_data = self.get_system_health()
            
            # Collect data source details
            data_source_details = {}
            data_sources = market_data.get('data_sources', {})
            
            for source_name, source_data in data_sources.items():
                data_source_details[source_name] = {
                    'status': source_data.get('status', 'unknown'),
                    'last_update': source_data.get('last_update'),
                    'error': source_data.get('error'),
                    'data_count': self._get_data_count(source_name, source_data)
                }
            
            return {
                'report_timestamp': datetime.now().isoformat(),
                'system_metadata': market_data.get('metadata', {}),
                'overall_health': health_data,
                'data_source_details': data_source_details,
                'configuration': {
                    'data_collection_enabled': self.config.get('data_collection.enabled', True),
                    'update_intervals': self._get_update_intervals(),
                    'monitored_pairs': self.config.pairs.get('monitored_pairs', [])
                },
                'file_status': self._get_file_status()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error creating status report: {e}")
            return {'error': str(e)}
    
    def _get_data_count(self, source_name: str, source_data: Dict) -> int:
        """Get data count for a source"""
        count_fields = {
            'economic_calendar': 'events_count',
            'sentiment': 'pairs_count',
            'correlation': 'currencies_count',
            'cot': 'records_count'
        }
        
        field = count_fields.get(source_name, 'count')
        return source_data.get(field, 0)
    
    def _get_update_intervals(self) -> Dict:
        """Get configured update intervals"""
        try:
            schedules_config = self.config.schedules.get('data_collection', {})
            return {
                'economic_calendar': schedules_config.get('economic_calendar', {}).get('interval_minutes', 60),
                'sentiment': schedules_config.get('sentiment', {}).get('interval_minutes', 30),
                'correlation': schedules_config.get('correlation', {}).get('interval_minutes', 30),
                'cot': f"{schedules_config.get('cot', {}).get('update_day', 'friday')} at {schedules_config.get('cot', {}).get('update_time', '18:00')}"
            }
        except Exception:
            return {}
    
    def _get_file_status(self) -> Dict:
        """Get status of important files"""
        try:
            files_to_check = {
                'market_data.json': self.market_data_file,
                'sentiment_signals.json': Path('sentiment_signals.json'),
                'correlation_data.json': Path('correlation_data.json'),
                'cot_consolidated_data.json': Path('cot_consolidated_data.json')
            }
            
            file_status = {}
            for file_name, file_path in files_to_check.items():
                if file_path.exists():
                    stat = file_path.stat()
                    file_status[file_name] = {
                        'exists': True,
                        'size_bytes': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    }
                else:
                    file_status[file_name] = {
                        'exists': False,
                        'size_bytes': 0,
                        'modified': None
                    }
            
            return file_status
            
        except Exception as e:
            self.logger.error(f"❌ Error getting file status: {e}")
            return {}
    
    # ===== CLEANUP AND SHUTDOWN =====
    
    def request_shutdown(self):
        """Request graceful shutdown"""
        self.logger.info("🛑 Shutdown requested for Data Manager")
        self.shutdown_event.set()
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.logger.info("🔄 Cleaning up Data Manager...")
            
            # Clear scheduled jobs
            schedule.clear()
            
            # Update final status
            try:
                market_data = self._load_market_data()
                market_data['system_status'] = 'stopped'
                market_data['last_updated'] = datetime.now().isoformat()
                self._save_market_data(market_data)
            except Exception as e:
                self.logger.warning(f"⚠️ Could not update final status: {e}")
            
            self.logger.info("✅ Data Manager cleanup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error during cleanup: {e}")

# ===== TESTING AND VALIDATION FUNCTIONS =====

def test_data_manager(config_manager=None):
    """Test function for data manager"""
    print("🧪 Testing Data Manager...")
    
    # Mock config if none provided
    if config_manager is None:
        class MockConfig:
            def __init__(self):
                self.schedules = {
                    'data_collection': {
                        'economic_calendar': {'enabled': True, 'interval_minutes': 60},
                        'sentiment': {'enabled': True, 'interval_minutes': 30},
                        'correlation': {'enabled': True, 'interval_minutes': 30},
                        'cot': {'enabled': True, 'update_day': 'friday', 'update_time': '18:00'}
                    }
                }
                self.pairs = {'monitored_pairs': ['EURUSD', 'GBPUSD', 'AUDUSD']}
            
            def get(self, key, default=None):
                keys = key.split('.')
                value = self.__dict__
                try:
                    for k in keys:
                        value = value[k]
                    return value
                except (KeyError, TypeError):
                    return default
        
        config_manager = MockConfig()
    
    # Mock logger
    import logging
    logger = logging.getLogger('test_data_manager')
    logger.setLevel(logging.INFO)
    
    # Create data manager
    data_manager = DataManager(config_manager, logger)
    
    # Test initialization
    print(f"✅ Scrapers loaded: {list(data_manager.scrapers.keys())}")
    print(f"✅ Components status: {data_manager.component_status}")
    
    # Test data file creation
    if data_manager.market_data_file.exists():
        print("✅ Market data file created")
        
        # Load and display sample data
        market_data = data_manager.get_market_data()
        if market_data:
            print(f"✅ Market data loaded: {market_data.get('system_status', 'unknown')} status")
    
    # Test health check
    health = data_manager.get_system_health()
    print(f"✅ System health: {health.get('overall_status', 'unknown')} ({health.get('health_score', 0)}% healthy)")
    
    # Test force update (if scrapers available)
    available_scrapers = list(data_manager.scrapers.keys())
    if available_scrapers:
        test_scraper = available_scrapers[0]
        print(f"🔄 Testing force update for {test_scraper}...")
        success = data_manager.force_update(test_scraper)
        print(f"✅ Force update result: {success}")
    
    print("🧪 Data Manager test completed")
    return data_manager

if __name__ == "__main__":
    test_data_manager()