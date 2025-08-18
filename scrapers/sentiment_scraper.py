# ===== SENTIMENT DATA MANAGER =====
# Combined scraper and processor for MyFXBook sentiment data
# Runs every 30 minutes and generates sentiment_signals.json for main bot

# PART 1: IMPORTS AND CONFIGURATION

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import json
import logging
import os
from datetime import datetime, timedelta
from urllib.parse import urljoin
import schedule
import threading
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# ===== CONFIGURATION =====
# Scraping settings
SCRAPE_INTERVAL_MINUTES = 30
SENTIMENT_OUTPUT_FILE = "sentiment_signals.json"
SENTIMENT_LOG_FILE = "sentiment_manager.log"

# Sentiment analysis settings
SENTIMENT_THRESHOLD = 60  # % threshold for direction blocking
BALANCED_RANGE = (40, 60)  # Range where both directions allowed
DATA_FRESHNESS_LIMIT_MINUTES = 60  # When to trigger fallback mode

# MyFXBook settings
MYFXBOOK_URL = "https://www.myfxbook.com/community/outlook"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

# Trading pairs to monitor (should match main bot)
MONITORED_PAIRS = ['AUDUSD', 'USDCAD', 'XAUUSD', 'EURUSD', 'GBPUSD', 
                  'AUDCAD', 'USDCHF', 'GBPCAD', 'AUDNZD', 'NZDCAD', 'US500', 'BTCUSD']

# ===== LOGGING SETUP =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - SENTIMENT_MANAGER - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(SENTIMENT_LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# PART 2: MYFXBOOK SCRAPER CLASS

class MyFXBookScraper:
    """Enhanced scraper for MyFXBook community outlook data"""
    
    def __init__(self):
        self.url = MYFXBOOK_URL
        self.headers = HEADERS
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
    def scrape_sentiment_data(self):
        """
        Scrape MyFXBook community outlook data and return structured data
        Returns: List of dictionaries with sentiment data or None if failed
        """
        try:
            logger.info("🔄 Fetching sentiment data from MyFXBook...")
            
            # Make request with timeout
            response = self.session.get(self.url, timeout=30)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the outlook table
            outlook_table = soup.find('table', {'id': 'outlookSymbolsTable'})
            
            if not outlook_table:
                logger.error("❌ Could not find outlook table on MyFXBook page")
                return None
            
            # Extract data from each row
            data_rows = []
            symbol_rows = outlook_table.find('tbody').find_all('tr', class_='outlook-symbol-row')
            
            logger.info(f"📊 Found {len(symbol_rows)} symbols to process")
            
            for row in symbol_rows:
                try:
                    symbol_data = self._extract_symbol_data(row, soup)
                    if symbol_data:
                        data_rows.extend(symbol_data)
                        
                except Exception as e:
                    logger.warning(f"⚠️ Error processing symbol row: {e}")
                    continue
            
            logger.info(f"✅ Successfully scraped {len(data_rows)} sentiment records")
            return data_rows
            
        except requests.RequestException as e:
            logger.error(f"❌ Network error scraping MyFXBook: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error scraping MyFXBook: {e}")
            return None
    
    def _extract_symbol_data(self, row, soup):
        """Extract sentiment data for a single symbol"""
        try:
            # Extract symbol name
            symbol_cell = row.find('td').find('a')
            if not symbol_cell:
                return None
                
            symbol = symbol_cell.text.strip()
            
            # Get symbol ID from the row
            symbol_id = row.get('symbolid', '')
            
            # Initialize data containers
            sentiment_data = {
                'symbol': symbol,
                'short_percentage': "N/A",
                'long_percentage': "N/A", 
                'short_volume': "N/A",
                'long_volume': "N/A",
                'short_positions': "N/A",
                'long_positions': "N/A",
                'popularity': "N/A",
                'current_price': "N/A"
            }
            
            # Extract current price
            rate_elem = soup.find('span', {'id': f'rateCell{symbol}'})
            if rate_elem:
                current_price_text = rate_elem.get_text()
                sentiment_data['current_price'] = re.sub(r'[↓↑]', '', current_price_text).strip()
            
            # Extract popularity from main table row
            cells = row.find_all('td')
            if len(cells) > 2:
                popularity_cell = cells[2]
                popularity_bar = popularity_cell.find('div', class_='progress-bar')
                if popularity_bar and popularity_bar.get('style'):
                    style = popularity_bar.get('style', '')
                    width_match = re.search(r'width:\s*(\d+)%', style)
                    if width_match:
                        sentiment_data['popularity'] = width_match.group(1) + '%'
            
            # Extract community trend percentages from progress bars
            if len(cells) > 1:
                trend_cell = cells[1]
                progress_bars = trend_cell.find_all('div', class_='progress-bar')
                if len(progress_bars) >= 2:
                    # First bar is short (red/danger), second is long (green/success)
                    short_bar = progress_bars[0]
                    long_bar = progress_bars[1]
                    
                    short_style = short_bar.get('style', '')
                    long_style = long_bar.get('style', '')
                    
                    short_match = re.search(r'width:\s*(\d+)%', short_style)
                    long_match = re.search(r'width:\s*(\d+)%', long_style)
                    
                    if short_match:
                        sentiment_data['short_percentage'] = short_match.group(1) + '%'
                    if long_match:
                        sentiment_data['long_percentage'] = long_match.group(1) + '%'
            
            # Extract detailed volume and position data from popover
            self._extract_popover_data(soup, symbol, symbol_id, sentiment_data)
            
            # Create individual records for short and long
            records = []
            
            # Short record
            short_record = {
                'Pair': symbol,
                'Direction': 'Short',
                'Percentage': sentiment_data['short_percentage'],
                'Volume': sentiment_data['short_volume'],
                'Positions': sentiment_data['short_positions'],
                'Popularity': sentiment_data['popularity'],
                'Current_Price': sentiment_data['current_price']
            }
            records.append(short_record)
            
            # Long record  
            long_record = {
                'Pair': symbol,
                'Direction': 'Long',
                'Percentage': sentiment_data['long_percentage'],
                'Volume': sentiment_data['long_volume'],
                'Positions': sentiment_data['long_positions'],
                'Popularity': sentiment_data['popularity'],
                'Current_Price': sentiment_data['current_price']
            }
            records.append(long_record)
            
            return records
            
        except Exception as e:
            logger.error(f"❌ Error extracting data for symbol: {e}")
            return None
    
    def _extract_popover_data(self, soup, symbol, symbol_id, sentiment_data):
        """Extract volume and position data from popover elements"""
        try:
            # Find the hidden popover div
            popover_div = soup.find('div', {'id': f'outlookSymbolPopover{symbol_id}'})
            if not popover_div:
                # Try different possible popover IDs or find in hidden cells
                hidden_cells = soup.find_all('td', style=lambda x: x and 'display: none' in x)
                for hidden_cell in hidden_cells:
                    popover_div = hidden_cell.find('div')
                    if popover_div:
                        break
            
            if popover_div:
                popover_table = popover_div.find('table')
                if popover_table:
                    tbody = popover_table.find('tbody')
                    if tbody:
                        popover_rows = tbody.find_all('tr')
                        for i, pop_row in enumerate(popover_rows):
                            pop_cells = pop_row.find_all('td')
                            
                            # First row has 5 cells (symbol, action, percentage, volume, positions)
                            # Second row has 4 cells (action, percentage, volume, positions) due to rowspan
                            if i == 0 and len(pop_cells) >= 5:
                                # First row - Short data
                                action = pop_cells[1].text.strip()
                                percentage_cell = pop_cells[2].text.strip()
                                volume_cell = pop_cells[3].text.strip()
                                positions_cell = pop_cells[4].text.strip()
                                
                                if action.lower() == "short":
                                    sentiment_data['short_volume'] = volume_cell
                                    sentiment_data['short_positions'] = positions_cell
                                    if percentage_cell != "N/A":
                                        sentiment_data['short_percentage'] = percentage_cell
                                        
                            elif i == 1 and len(pop_cells) >= 4:
                                # Second row - Long data (no symbol cell due to rowspan)
                                action = pop_cells[0].text.strip()
                                percentage_cell = pop_cells[1].text.strip()
                                volume_cell = pop_cells[2].text.strip()
                                positions_cell = pop_cells[3].text.strip()
                                
                                if action.lower() == "long":
                                    sentiment_data['long_volume'] = volume_cell
                                    sentiment_data['long_positions'] = positions_cell
                                    if percentage_cell != "N/A":
                                        sentiment_data['long_percentage'] = percentage_cell
                                        
        except Exception as e:
            logger.debug(f"Could not extract popover data for {symbol}: {e}")
            # This is not critical, continue without popover data
            
            # PART 3: SENTIMENT ANALYZER CLASS

class SentimentAnalyzer:
    """Processes raw sentiment data into trading decisions"""
    
    def __init__(self):
        self.threshold = SENTIMENT_THRESHOLD
        self.balanced_range = BALANCED_RANGE
        
    def process_sentiment_data(self, raw_data):
        """
        Process raw sentiment data into trading signals
        Args:
            raw_data: List of sentiment records from scraper
        Returns:
            Dict with processed sentiment signals
        """
        if not raw_data:
            logger.error("❌ No raw sentiment data to process")
            return None
        
        try:
            logger.info("🔄 Processing sentiment data into trading signals...")
            
            # Group data by pair
            pairs_data = {}
            for record in raw_data:
                pair = record['Pair']
                direction = record['Direction'].lower()
                
                if pair not in pairs_data:
                    pairs_data[pair] = {'short': {}, 'long': {}}
                
                pairs_data[pair][direction] = record
            
            # Process each pair
            processed_signals = {
                'timestamp': datetime.now().isoformat(),
                'data_source': 'MyFXBook',
                'threshold_used': self.threshold,
                'pairs': {}
            }
            
            for pair, data in pairs_data.items():
                try:
                    pair_signals = self._analyze_pair_sentiment(pair, data)
                    if pair_signals:
                        processed_signals['pairs'][pair] = pair_signals
                        
                except Exception as e:
                    logger.warning(f"⚠️ Error processing {pair}: {e}")
                    continue
            
            logger.info(f"✅ Processed sentiment for {len(processed_signals['pairs'])} pairs")
            return processed_signals
            
        except Exception as e:
            logger.error(f"❌ Error processing sentiment data: {e}")
            return None
    
    def _analyze_pair_sentiment(self, pair, data):
        """Analyze sentiment for a single pair"""
        try:
            # Extract percentages
            short_data = data.get('short', {})
            long_data = data.get('long', {})
            
            short_pct_str = short_data.get('Percentage', 'N/A')
            long_pct_str = long_data.get('Percentage', 'N/A')
            
            # Parse percentages (remove % symbol and convert to int)
            short_pct = self._parse_percentage(short_pct_str)
            long_pct = self._parse_percentage(long_pct_str)
            
            if short_pct is None or long_pct is None:
                logger.warning(f"⚠️ Invalid sentiment data for {pair}: Short={short_pct_str}, Long={long_pct_str}")
                return None
            
            # Validate percentages add up reasonably (allow some tolerance)
            total_pct = short_pct + long_pct
            if abs(total_pct - 100) > 5:  # 5% tolerance
                logger.warning(f"⚠️ {pair} percentages don't add to 100%: {short_pct}% + {long_pct}% = {total_pct}%")
            
            # Determine allowed directions based on sentiment threshold
            allowed_directions = []
            blocked_directions = []
            
            if short_pct >= self.threshold:
                allowed_directions.append('short')
                blocked_directions.append('long')
                signal_strength = "Strong Short"
            elif long_pct >= self.threshold:
                allowed_directions.append('long') 
                blocked_directions.append('short')
                signal_strength = "Strong Long"
            else:
                # Balanced sentiment - allow both directions
                allowed_directions = ['short', 'long']
                blocked_directions = []
                signal_strength = "Balanced"
            
            # Extract additional data
            current_price = short_data.get('Current_Price', 'N/A')
            popularity = short_data.get('Popularity', 'N/A')
            
            # Extract volume and position data
            short_volume = short_data.get('Volume', 'N/A')
            long_volume = long_data.get('Volume', 'N/A')
            short_positions = short_data.get('Positions', 'N/A')
            long_positions = long_data.get('Positions', 'N/A')
            
            pair_analysis = {
                'allowed_directions': allowed_directions,
                'blocked_directions': blocked_directions,
                'sentiment': {
                    'short': short_pct,
                    'long': long_pct
                },
                'signal_strength': signal_strength,
                'current_price': current_price,
                'popularity': popularity,
                'volume_data': {
                    'short': short_volume,
                    'long': long_volume
                },
                'position_data': {
                    'short': short_positions,
                    'long': long_positions
                },
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Log the analysis
            direction_status = f"Allowed: {allowed_directions}" if allowed_directions else "None allowed"
            if blocked_directions:
                direction_status += f", Blocked: {blocked_directions}"
            
            logger.info(f"📊 {pair}: {short_pct}%↓ {long_pct}%↑ → {signal_strength} ({direction_status})")
            
            return pair_analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {pair} sentiment: {e}")
            return None
    
    def _parse_percentage(self, pct_str):
        """Parse percentage string to integer"""
        try:
            if pct_str == 'N/A' or not pct_str:
                return None
            
            # Remove % symbol and any whitespace
            clean_str = str(pct_str).replace('%', '').strip()
            
            # Convert to integer
            return int(float(clean_str))
            
        except (ValueError, TypeError):
            logger.debug(f"Could not parse percentage: {pct_str}")
            return None
    
    def validate_signals(self, signals):
        """Validate processed signals before saving"""
        if not signals or 'pairs' not in signals:
            return False
        
        if not signals['pairs']:
            logger.warning("⚠️ No pairs in processed signals")
            return False
        
        # Check for monitored pairs
        found_pairs = set(signals['pairs'].keys())
        monitored_pairs = set(MONITORED_PAIRS)
        
        # Normalize pair names for comparison (remove common variations)
        normalized_found = {self._normalize_pair_name(p) for p in found_pairs}
        normalized_monitored = {self._normalize_pair_name(p) for p in monitored_pairs}
        
        overlap = normalized_found.intersection(normalized_monitored)
        overlap_ratio = len(overlap) / len(normalized_monitored) if normalized_monitored else 0
        
        logger.info(f"📈 Coverage: {len(overlap)}/{len(normalized_monitored)} monitored pairs ({overlap_ratio:.1%})")
        
        if overlap_ratio < 0.3:  # Less than 30% coverage
            logger.warning(f"⚠️ Low coverage of monitored pairs: {overlap_ratio:.1%}")
        
        return True
    
    def _normalize_pair_name(self, pair):
        """Normalize pair name for comparison"""
        # Handle common variations
        normalized = pair.upper()
        
        # Map common variations
        mapping = {
            'GOLD': 'XAUUSD',
            'XAUUSD': 'XAUUSD',
            'SPXUSD': 'US500',
            'SPX500': 'US500',
            'US500': 'US500',
            'BITCOIN': 'BTCUSD',
            'BTCUSD': 'BTCUSD'
        }
        
        return mapping.get(normalized, normalized)
    
    # PART 4: SIGNAL MANAGER AND FILE OPERATIONS

class SentimentSignalManager:
    """Manages sentiment signals and file operations"""
    
    def __init__(self):
        self.output_file = SENTIMENT_OUTPUT_FILE
        self.scraper = MyFXBookScraper()
        self.analyzer = SentimentAnalyzer()
        
    def update_sentiment_signals(self):
        """Main function to update sentiment signals"""
        try:
            logger.info("🚀 Starting sentiment signal update...")
            
            # Step 1: Scrape raw data
            raw_data = self.scraper.scrape_sentiment_data()
            if not raw_data:
                logger.error("❌ Failed to scrape sentiment data")
                self._handle_scraping_failure()
                return False
            
            # Step 2: Process into signals
            processed_signals = self.analyzer.process_sentiment_data(raw_data)
            if not processed_signals:
                logger.error("❌ Failed to process sentiment data")
                return False
            
            # Step 3: Validate signals
            if not self.analyzer.validate_signals(processed_signals):
                logger.error("❌ Signal validation failed")
                return False
            
            # Step 4: Save to file
            if self.save_signals(processed_signals):
                logger.info("✅ Sentiment signals updated successfully")
                self._log_summary(processed_signals)
                return True
            else:
                logger.error("❌ Failed to save sentiment signals")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error updating sentiment signals: {e}")
            return False
    
    def save_signals(self, signals):
        """Save processed signals to JSON file"""
        try:
            # Add metadata
            signals['file_info'] = {
                'version': '1.0',
                'created_by': 'sentiment_manager.py',
                'data_freshness_limit_minutes': DATA_FRESHNESS_LIMIT_MINUTES
            }
            
            # Create backup of existing file
            if os.path.exists(self.output_file):
                backup_file = f"{self.output_file}.backup"
                try:
                    import shutil
                    shutil.copy2(self.output_file, backup_file)
                    logger.debug(f"Created backup: {backup_file}")
                except Exception as e:
                    logger.warning(f"Could not create backup: {e}")
            
            # Write new signals
            with open(self.output_file, 'w') as f:
                json.dump(signals, f, indent=2, default=str)
            
            logger.info(f"💾 Signals saved to {self.output_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving signals: {e}")
            return False
    
    def load_signals(self):
        """Load signals from file (for testing/verification)"""
        try:
            if not os.path.exists(self.output_file):
                logger.warning(f"⚠️ Signal file not found: {self.output_file}")
                return None
            
            with open(self.output_file, 'r') as f:
                signals = json.load(f)
            
            # Check data freshness
            timestamp = datetime.fromisoformat(signals['timestamp'])
            age_minutes = (datetime.now() - timestamp).total_seconds() / 60
            
            logger.info(f"📖 Loaded signals from {timestamp} ({age_minutes:.1f} minutes old)")
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error loading signals: {e}")
            return None
    
    def _handle_scraping_failure(self):
        """Handle scraping failure - create fallback signal file"""
        try:
            fallback_signals = {
                'timestamp': datetime.now().isoformat(),
                'data_source': 'Fallback',
                'error': 'Scraping failed - all pairs set to normal trading',
                'threshold_used': self.analyzer.threshold,
                'pairs': {}
            }
            
            # Set all monitored pairs to allow both directions (fallback mode)
            for pair in MONITORED_PAIRS:
                fallback_signals['pairs'][pair] = {
                    'allowed_directions': ['short', 'long'],
                    'blocked_directions': [],
                    'sentiment': {'short': 50, 'long': 50},  # Balanced
                    'signal_strength': 'Fallback - No Data',
                    'current_price': 'N/A',
                    'popularity': 'N/A',
                    'volume_data': {'short': 'N/A', 'long': 'N/A'},
                    'position_data': {'short': 'N/A', 'long': 'N/A'},
                    'analysis_timestamp': datetime.now().isoformat()
                }
            
            self.save_signals(fallback_signals)
            logger.info("🔄 Created fallback signal file")
            
        except Exception as e:
            logger.error(f"❌ Error creating fallback signals: {e}")
    
    def _log_summary(self, signals):
        """Log summary of processed signals"""
        try:
            pairs = signals.get('pairs', {})
            
            # Count directions
            strong_short = sum(1 for p in pairs.values() if p.get('signal_strength') == 'Strong Short')
            strong_long = sum(1 for p in pairs.values() if p.get('signal_strength') == 'Strong Long')
            balanced = sum(1 for p in pairs.values() if p.get('signal_strength') == 'Balanced')
            
            logger.info("📊 SENTIMENT SUMMARY:")
            logger.info(f"   Total pairs processed: {len(pairs)}")
            logger.info(f"   Strong Short signals: {strong_short}")
            logger.info(f"   Strong Long signals: {strong_long}")
            logger.info(f"   Balanced signals: {balanced}")
            
            # Log specific strong signals
            for pair, data in pairs.items():
                if data.get('signal_strength') in ['Strong Short', 'Strong Long']:
                    sentiment = data.get('sentiment', {})
                    short_pct = sentiment.get('short', 0)
                    long_pct = sentiment.get('long', 0)
                    blocked = data.get('blocked_directions', [])
                    logger.info(f"   🎯 {pair}: {short_pct}%↓ {long_pct}%↑ (Blocked: {blocked})")
            
        except Exception as e:
            logger.error(f"❌ Error logging summary: {e}")
    
    def get_signal_status(self):
        """Get current signal status for monitoring"""
        try:
            signals = self.load_signals()
            if not signals:
                return "No signal file found"
            
            timestamp = datetime.fromisoformat(signals['timestamp'])
            age_minutes = (datetime.now() - timestamp).total_seconds() / 60
            
            pairs_count = len(signals.get('pairs', {}))
            data_source = signals.get('data_source', 'Unknown')
            
            if age_minutes > DATA_FRESHNESS_LIMIT_MINUTES:
                freshness = f"STALE ({age_minutes:.1f}m old)"
            else:
                freshness = f"FRESH ({age_minutes:.1f}m old)"
            
            return f"{data_source}: {pairs_count} pairs, {freshness}"
            
        except Exception as e:
            return f"Error: {e}"
        
        # PART 5: SCHEDULER AND MAIN FUNCTIONS

class SentimentScheduler:
    """Handles scheduling of sentiment updates"""
    
    def __init__(self):
        self.signal_manager = SentimentSignalManager()
        self.running = False
        self.thread = None
        
    def start_scheduler(self):
        """Start the sentiment update scheduler"""
        try:
            logger.info("🕐 Starting sentiment scheduler...")
            
            # Schedule updates every 30 minutes
            schedule.every(SCRAPE_INTERVAL_MINUTES).minutes.do(self._scheduled_update)
            
            # Run initial update immediately
            logger.info("🚀 Running initial sentiment update...")
            self._scheduled_update()
            
            self.running = True
            
            # Start scheduler in separate thread
            self.thread = threading.Thread(target=self._run_scheduler, daemon=True)
            self.thread.start()
            
            logger.info(f"✅ Scheduler started - updates every {SCRAPE_INTERVAL_MINUTES} minutes")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error starting scheduler: {e}")
            return False
    
    def stop_scheduler(self):
        """Stop the sentiment scheduler"""
        try:
            self.running = False
            schedule.clear()
            
            if self.thread:
                self.thread.join(timeout=5)
            
            logger.info("🛑 Sentiment scheduler stopped")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error stopping scheduler: {e}")
            return False
    
    def _run_scheduler(self):
        """Run the scheduler loop"""
        logger.info("🔄 Scheduler thread started")
        
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"❌ Error in scheduler loop: {e}")
                time.sleep(10)  # Wait before retrying
        
        logger.info("🔄 Scheduler thread stopped")
    
    def _scheduled_update(self):
        """Perform scheduled sentiment update"""
        try:
            logger.info("⏰ Running scheduled sentiment update...")
            
            success = self.signal_manager.update_sentiment_signals()
            
            if success:
                status = self.signal_manager.get_signal_status()
                logger.info(f"✅ Scheduled update completed: {status}")
            else:
                logger.error("❌ Scheduled update failed")
                
        except Exception as e:
            logger.error(f"❌ Error in scheduled update: {e}")
    
    def force_update(self):
        """Force an immediate sentiment update"""
        try:
            logger.info("🔥 Forcing immediate sentiment update...")
            return self.signal_manager.update_sentiment_signals()
            
        except Exception as e:
            logger.error(f"❌ Error in forced update: {e}")
            return False
    
    def get_status(self):
        """Get current scheduler status"""
        try:
            status_info = {
                'running': self.running,
                'interval_minutes': SCRAPE_INTERVAL_MINUTES,
                'next_run': None,
                'signal_status': self.signal_manager.get_signal_status()
            }
            
            # Get next scheduled run time
            jobs = schedule.jobs
            if jobs:
                next_run = min(jobs, key=lambda x: x.next_run).next_run
                status_info['next_run'] = next_run.isoformat()
            
            return status_info
            
        except Exception as e:
            logger.error(f"❌ Error getting status: {e}")
            return {'error': str(e)}

# ===== MAIN FUNCTIONS =====

def run_sentiment_manager():
    """Main function to run sentiment manager"""
    logger.info("="*60)
    logger.info("SENTIMENT DATA MANAGER STARTED")
    logger.info("="*60)
    logger.info(f"Update interval: {SCRAPE_INTERVAL_MINUTES} minutes")
    logger.info(f"Sentiment threshold: {SENTIMENT_THRESHOLD}%")
    logger.info(f"Monitored pairs: {len(MONITORED_PAIRS)}")
    logger.info(f"Output file: {SENTIMENT_OUTPUT_FILE}")
    logger.info("="*60)
    
    scheduler = SentimentScheduler()
    
    try:
        # Start scheduler
        if not scheduler.start_scheduler():
            logger.error("❌ Failed to start scheduler")
            return
        
        logger.info("🎯 Sentiment manager running. Press Ctrl+C to stop.")
        
        # Keep main thread alive
        while scheduler.running:
            try:
                time.sleep(60)  # Check every minute
                
                # Log status every 10 minutes
                if datetime.now().minute % 10 == 0:
                    status = scheduler.get_status()
                    logger.info(f"📊 Status: {status['signal_status']}")
                    
            except KeyboardInterrupt:
                logger.info("🛑 Stop signal received")
                break
                
    except Exception as e:
        logger.error(f"❌ Error in main loop: {e}")
        
    finally:
        # Cleanup
        logger.info("🔄 Shutting down sentiment manager...")
        scheduler.stop_scheduler()
        logger.info("✅ Sentiment manager stopped")

def test_sentiment_scraping():
    """Test function for sentiment scraping"""
    logger.info("🧪 Testing sentiment scraping...")
    
    signal_manager = SentimentSignalManager()
    
    # Test scraping
    success = signal_manager.update_sentiment_signals()
    
    if success:
        # Load and display results
        signals = signal_manager.load_signals()
        if signals:
            pairs = signals.get('pairs', {})
            logger.info(f"✅ Test successful: {len(pairs)} pairs processed")
            
            # Show sample results
            for pair, data in list(pairs.items())[:3]:  # First 3 pairs
                sentiment = data.get('sentiment', {})
                allowed = data.get('allowed_directions', [])
                blocked = data.get('blocked_directions', [])
                logger.info(f"   {pair}: {sentiment.get('short', 0)}%↓ {sentiment.get('long', 0)}%↑")
                logger.info(f"         Allowed: {allowed}, Blocked: {blocked}")
        else:
            logger.error("❌ Could not load saved signals")
    else:
        logger.error("❌ Test failed")

def create_sample_signals():
    """Create sample signals file for testing main bot integration"""
    logger.info("📝 Creating sample signals file...")
    
    sample_signals = {
        'timestamp': datetime.now().isoformat(),
        'data_source': 'Sample/Test',
        'threshold_used': SENTIMENT_THRESHOLD,
        'pairs': {}
    }
    
    # Create sample data for monitored pairs
    sample_data = [
        ('XAUUSD', 70, 30, 'Strong Short'),  # Block long
        ('EURUSD', 45, 55, 'Balanced'),      # Allow both
        ('GBPUSD', 25, 75, 'Strong Long'),   # Block short
        ('BTCUSD', 50, 50, 'Balanced'),      # Allow both
    ]
    
    for pair, short_pct, long_pct, strength in sample_data:
        if strength == 'Strong Short':
            allowed = ['short']
            blocked = ['long']
        elif strength == 'Strong Long':
            allowed = ['long']
            blocked = ['short']
        else:
            allowed = ['short', 'long']
            blocked = []
        
        sample_signals['pairs'][pair] = {
            'allowed_directions': allowed,
            'blocked_directions': blocked,
            'sentiment': {'short': short_pct, 'long': long_pct},
            'signal_strength': strength,
            'current_price': 'Sample',
            'popularity': 'Sample',
            'volume_data': {'short': 'Sample', 'long': 'Sample'},
            'position_data': {'short': 'Sample', 'long': 'Sample'},
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    # Add remaining pairs as balanced
    for pair in MONITORED_PAIRS:
        if pair not in sample_signals['pairs']:
            sample_signals['pairs'][pair] = {
                'allowed_directions': ['short', 'long'],
                'blocked_directions': [],
                'sentiment': {'short': 50, 'long': 50},
                'signal_strength': 'Balanced',
                'current_price': 'Sample',
                'popularity': 'Sample',
                'volume_data': {'short': 'Sample', 'long': 'Sample'},
                'position_data': {'short': 'Sample', 'long': 'Sample'},
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    try:
        with open(SENTIMENT_OUTPUT_FILE, 'w') as f:
            json.dump(sample_signals, f, indent=2)
        
        logger.info(f"✅ Sample signals created: {SENTIMENT_OUTPUT_FILE}")
        logger.info(f"   Pairs included: {len(sample_signals['pairs'])}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating sample signals: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'test':
            test_sentiment_scraping()
        elif command == 'sample':
            create_sample_signals()
        elif command == 'once':
            # Run update once and exit
            signal_manager = SentimentSignalManager()
            success = signal_manager.update_sentiment_signals()
            sys.exit(0 if success else 1)
        else:
            print("Usage: python sentiment_manager.py [test|sample|once]")
            sys.exit(1)
    else:
        # Run continuous scheduler
        run_sentiment_manager()