# ===== CORRELATION DATA MANAGER =====
# Combined scraper and processor for MyFXBook correlation data
# Runs every 30 minutes and generates correlation_data.json for main bot

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
CORRELATION_OUTPUT_FILE = "correlation_data.json"
CORRELATION_LOG_FILE = "correlation_manager.log"

# Correlation analysis settings
HIGH_CORRELATION_THRESHOLD = 70  # % threshold for high correlation warning
NEGATIVE_CORRELATION_THRESHOLD = -70  # % threshold for negative correlation warning
DATA_FRESHNESS_LIMIT_MINUTES = 60  # When to trigger fallback mode

# MyFXBook settings
MYFXBOOK_CORRELATION_URL = "https://www.myfxbook.com/forex-market/correlation"
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
    format='%(asctime)s - CORRELATION_MANAGER - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CORRELATION_LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# PART 2: MYFXBOOK CORRELATION SCRAPER CLASS

class MyFXBookCorrelationScraper:
    """Enhanced scraper for MyFXBook correlation data"""
    
    def __init__(self):
        self.url = MYFXBOOK_CORRELATION_URL
        self.headers = HEADERS
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
    def scrape_correlation_data(self):
        """
        Scrape MyFXBook correlation data and return structured data
        Returns: Dictionary with correlation matrix or None if failed
        """
        try:
            logger.info("🔄 Fetching correlation data from MyFXBook...")
            
            # Make request with timeout
            response = self.session.get(self.url, timeout=30)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the correlation table
            correlation_table = soup.find('table', {'class': 'correlation-table'})
            
            if not correlation_table:
                # Try alternative selectors
                correlation_table = soup.find('table', {'id': 'correlationTable'})
                
            if not correlation_table:
                # Try finding any table with correlation data
                tables = soup.find_all('table')
                for table in tables:
                    if self._has_correlation_data(table):
                        correlation_table = table
                        break
            
            if not correlation_table:
                logger.error("❌ Could not find correlation table on MyFXBook page")
                return None
            
            # Extract correlation matrix
            correlation_matrix = self._extract_correlation_matrix(correlation_table)
            
            if not correlation_matrix:
                logger.error("❌ Failed to extract correlation matrix")
                return None
            
            logger.info(f"✅ Successfully scraped correlation data for {len(correlation_matrix)} currencies")
            return correlation_matrix
            
        except requests.RequestException as e:
            logger.error(f"❌ Network error scraping MyFXBook correlation: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error scraping MyFXBook correlation: {e}")
            return None
    
    def _has_correlation_data(self, table):
        """Check if table contains correlation data"""
        try:
            # Look for correlation-specific elements
            spans = table.find_all('span', {'name': 'correlationSymbolByRowAndCol'})
            if spans:
                return True
            
            # Look for percentage values that could be correlations
            cells = table.find_all('td')
            correlation_count = 0
            for cell in cells[:20]:  # Check first 20 cells
                text = cell.get_text(strip=True)
                if re.match(r'^-?\d+\.\d+%$', text):
                    correlation_count += 1
            
            return correlation_count >= 5
            
        except Exception:
            return False
    
    def _extract_correlation_matrix(self, table):
        """Extract correlation matrix from table"""
        try:
            correlation_data = {}
            
            # Find header row to get column symbols
            header_symbols = []
            thead = table.find('thead')
            if thead:
                header_row = thead.find('tr')
                if header_row:
                    headers = header_row.find_all(['th', 'td'])
                    for header in headers[1:]:  # Skip first column (usually currency names)
                        symbol_text = header.get_text(strip=True)
                        if symbol_text and len(symbol_text) <= 10:  # Reasonable symbol length
                            header_symbols.append(symbol_text)
            
            # If no header found, try to extract from first data row
            if not header_symbols:
                logger.warning("⚠️ No header found, attempting to extract symbols from data")
                header_symbols = self._extract_symbols_from_data(table)
            
            logger.info(f"📊 Found column symbols: {header_symbols[:10]}...")  # Show first 10
            
            # Find tbody or use the whole table
            tbody = table.find('tbody')
            if not tbody:
                tbody = table
            
            # Extract data rows
            rows = tbody.find_all('tr')
            logger.info(f"📊 Processing {len(rows)} correlation rows")
            
            for row_idx, row in enumerate(rows):
                try:
                    row_data = self._extract_row_correlation_data(row, header_symbols)
                    if row_data:
                        correlation_data.update(row_data)
                        
                except Exception as e:
                    logger.warning(f"⚠️ Error processing row {row_idx}: {e}")
                    continue
            
            if not correlation_data:
                logger.error("❌ No correlation data extracted")
                return None
            
            # Validate and clean the matrix
            cleaned_matrix = self._validate_correlation_matrix(correlation_data)
            
            return cleaned_matrix
            
        except Exception as e:
            logger.error(f"❌ Error extracting correlation matrix: {e}")
            return None
    
    def _extract_symbols_from_data(self, table):
        """Extract symbols from correlation spans in the data"""
        try:
            symbols = set()
            spans = table.find_all('span', {'name': 'correlationSymbolByRowAndCol'})
            
            for span in spans[:50]:  # Limit to first 50 to avoid too much processing
                row_symbol = span.get('rowsymbol', '')
                col_symbol = span.get('colsymbol', '')
                
                if row_symbol:
                    symbols.add(row_symbol)
                if col_symbol:
                    symbols.add(col_symbol)
            
            return sorted(list(symbols))
            
        except Exception as e:
            logger.error(f"❌ Error extracting symbols from data: {e}")
            return []
    
    def _extract_row_correlation_data(self, row, header_symbols):
        """Extract correlation data from a single row"""
        try:
            cells = row.find_all(['td', 'th'])
            if len(cells) < 2:
                return None
            
            # First cell should contain the row symbol/currency
            first_cell = cells[0]
            row_symbol = self._extract_symbol_from_cell(first_cell)
            
            if not row_symbol:
                return None
            
            row_correlations = {}
            
            # Process correlation cells
            for col_idx, cell in enumerate(cells[1:]):  # Skip first cell (symbol name)
                try:
                    # Look for correlation span with specific attributes
                    correlation_span = cell.find('span', {'name': 'correlationSymbolByRowAndCol'})
                    
                    if correlation_span:
                        # Extract from span attributes
                        col_symbol = correlation_span.get('colsymbol', '')
                        correlation_value = correlation_span.get('value', '')
                        
                        if col_symbol and correlation_value:
                            try:
                                # Parse correlation value
                                correlation_float = float(correlation_value)
                                
                                # Store correlation
                                if row_symbol not in row_correlations:
                                    row_correlations[row_symbol] = {}
                                
                                row_correlations[row_symbol][col_symbol] = {
                                    'value': correlation_float,
                                    'percentage': f"{correlation_float}%"
                                }
                                
                            except ValueError:
                                logger.debug(f"Could not parse correlation value: {correlation_value}")
                                continue
                    else:
                        # Fallback: extract from cell text
                        cell_text = cell.get_text(strip=True)
                        correlation_match = re.match(r'^(-?\d+\.\d+)%?$', cell_text)
                        
                        if correlation_match and col_idx < len(header_symbols):
                            correlation_value = float(correlation_match.group(1))
                            col_symbol = header_symbols[col_idx]
                            
                            if row_symbol not in row_correlations:
                                row_correlations[row_symbol] = {}
                            
                            row_correlations[row_symbol][col_symbol] = {
                                'value': correlation_value,
                                'percentage': f"{correlation_value}%"
                            }
                            
                except Exception as e:
                    logger.debug(f"Error processing cell {col_idx} in row {row_symbol}: {e}")
                    continue
            
            return row_correlations if row_correlations else None
            
        except Exception as e:
            logger.error(f"❌ Error extracting row correlation data: {e}")
            return None
    
    def _extract_symbol_from_cell(self, cell):
        """Extract currency symbol from cell"""
        try:
            # Look for link with symbol
            link = cell.find('a')
            if link:
                # Check if link has name attribute
                symbol = link.get('name', '')
                if symbol:
                    return symbol.upper()
                
                # Extract from link text
                symbol_text = link.get_text(strip=True)
                if symbol_text and len(symbol_text) <= 10:
                    return symbol_text.upper()
            
            # Fallback: extract from cell text
            cell_text = cell.get_text(strip=True)
            if cell_text and len(cell_text) <= 10:
                return cell_text.upper()
            
            return None
            
        except Exception:
            return None
    
    def _validate_correlation_matrix(self, raw_matrix):
        """Validate and clean correlation matrix"""
        try:
            cleaned_matrix = {}
            
            for row_symbol, correlations in raw_matrix.items():
                if not correlations:
                    continue
                
                cleaned_correlations = {}
                
                for col_symbol, data in correlations.items():
                    correlation_value = data.get('value')
                    
                    # Validate correlation value
                    if correlation_value is None:
                        continue
                    
                    # Correlation should be between -100 and 100
                    if not (-100 <= correlation_value <= 100):
                        logger.warning(f"⚠️ Invalid correlation value {correlation_value} for {row_symbol}-{col_symbol}")
                        continue
                    
                    cleaned_correlations[col_symbol] = data
                
                if cleaned_correlations:
                    cleaned_matrix[row_symbol] = cleaned_correlations
            
            logger.info(f"📊 Cleaned matrix: {len(cleaned_matrix)} currencies with correlations")
            return cleaned_matrix
            
        except Exception as e:
            logger.error(f"❌ Error validating correlation matrix: {e}")
            return raw_matrix

# PART 3: CORRELATION ANALYZER CLASS

class CorrelationAnalyzer:
    """Processes raw correlation data into trading insights"""
    
    def __init__(self):
        self.high_threshold = HIGH_CORRELATION_THRESHOLD
        self.negative_threshold = NEGATIVE_CORRELATION_THRESHOLD
        
    def process_correlation_data(self, raw_matrix):
        """
        Process raw correlation matrix into trading insights
        Args:
            raw_matrix: Dictionary with correlation data from scraper
        Returns:
            Dict with processed correlation analysis
        """
        if not raw_matrix:
            logger.error("❌ No raw correlation data to process")
            return None
        
        try:
            logger.info("🔄 Processing correlation data into trading insights...")
            
            # Process correlation matrix
            processed_analysis = {
                'timestamp': datetime.now().isoformat(),
                'data_source': 'MyFXBook',
                'thresholds': {
                    'high_correlation': self.high_threshold,
                    'negative_correlation': self.negative_threshold
                },
                'correlation_matrix': raw_matrix,
                'insights': {},
                'warnings': [],
                'monitored_pairs_analysis': {}
            }
            
            # Analyze correlations for insights
            insights = self._analyze_correlations(raw_matrix)
            processed_analysis['insights'] = insights
            
            # Generate warnings for high correlations
            warnings = self._generate_correlation_warnings(raw_matrix)
            processed_analysis['warnings'] = warnings
            
            # Analyze monitored pairs specifically
            monitored_analysis = self._analyze_monitored_pairs(raw_matrix)
            processed_analysis['monitored_pairs_analysis'] = monitored_analysis
            
            logger.info(f"✅ Processed correlation analysis with {len(insights)} insights and {len(warnings)} warnings")
            return processed_analysis
            
        except Exception as e:
            logger.error(f"❌ Error processing correlation data: {e}")
            return None
    
    def _analyze_correlations(self, matrix):
        """Generate insights from correlation matrix"""
        try:
            insights = {
                'highest_correlations': [],
                'strongest_negative_correlations': [],
                'currency_summary': {},
                'correlation_statistics': {}
            }
            
            all_correlations = []
            
            # Collect all correlations and analyze
            for row_symbol, correlations in matrix.items():
                currency_correlations = []
                
                for col_symbol, data in correlations.items():
                    if row_symbol != col_symbol:  # Skip self-correlation
                        correlation_value = data.get('value', 0)
                        
                        all_correlations.append({
                            'pair': f"{row_symbol}-{col_symbol}",
                            'value': correlation_value
                        })
                        
                        currency_correlations.append(correlation_value)
                
                # Calculate statistics for this currency
                if currency_correlations:
                    insights['currency_summary'][row_symbol] = {
                        'avg_correlation': sum(currency_correlations) / len(currency_correlations),
                        'max_correlation': max(currency_correlations),
                        'min_correlation': min(currency_correlations),
                        'correlation_count': len(currency_correlations)
                    }
            
            # Find highest positive correlations
            positive_correlations = [c for c in all_correlations if c['value'] > 0]
            positive_correlations.sort(key=lambda x: x['value'], reverse=True)
            insights['highest_correlations'] = positive_correlations[:10]
            
            # Find strongest negative correlations
            negative_correlations = [c for c in all_correlations if c['value'] < 0]
            negative_correlations.sort(key=lambda x: x['value'])
            insights['strongest_negative_correlations'] = negative_correlations[:10]
            
            # Overall statistics
            if all_correlations:
                all_values = [c['value'] for c in all_correlations]
                insights['correlation_statistics'] = {
                    'total_pairs': len(all_correlations),
                    'avg_correlation': sum(all_values) / len(all_values),
                    'max_correlation': max(all_values),
                    'min_correlation': min(all_values),
                    'high_correlation_count': len([v for v in all_values if v >= self.high_threshold]),
                    'negative_correlation_count': len([v for v in all_values if v <= self.negative_threshold])
                }
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Error analyzing correlations: {e}")
            return {}
    
    def _generate_correlation_warnings(self, matrix):
        """Generate warnings for problematic correlations"""
        try:
            warnings = []
            
            for row_symbol, correlations in matrix.items():
                for col_symbol, data in correlations.items():
                    if row_symbol == col_symbol:
                        continue
                    
                    correlation_value = data.get('value', 0)
                    
                    # High positive correlation warning
                    if correlation_value >= self.high_threshold:
                        warnings.append({
                            'type': 'HIGH_CORRELATION',
                            'pair': f"{row_symbol}-{col_symbol}",
                            'value': correlation_value,
                            'message': f"High correlation ({correlation_value}%) between {row_symbol} and {col_symbol} - risk of similar movements"
                        })
                    
                    # Strong negative correlation warning
                    elif correlation_value <= self.negative_threshold:
                        warnings.append({
                            'type': 'NEGATIVE_CORRELATION',
                            'pair': f"{row_symbol}-{col_symbol}",
                            'value': correlation_value,
                            'message': f"Strong negative correlation ({correlation_value}%) between {row_symbol} and {col_symbol} - risk of opposite movements"
                        })
            
            # Sort warnings by severity (absolute value)
            warnings.sort(key=lambda x: abs(x['value']), reverse=True)
            
            return warnings
            
        except Exception as e:
            logger.error(f"❌ Error generating warnings: {e}")
            return []
    
    def _analyze_monitored_pairs(self, matrix):
        """Analyze correlations specifically for monitored pairs"""
        try:
            monitored_analysis = {}
            
            # Normalize monitored pairs
            normalized_pairs = [self._normalize_pair_name(pair) for pair in MONITORED_PAIRS]
            
            for pair in MONITORED_PAIRS:
                normalized_pair = self._normalize_pair_name(pair)
                
                # Find correlations involving this pair
                pair_correlations = {}
                
                for row_symbol, correlations in matrix.items():
                    if row_symbol == normalized_pair:
                        pair_correlations = correlations
                        break
                
                if pair_correlations:
                    # Analyze correlations for this monitored pair
                    high_correlations = []
                    negative_correlations = []
                    
                    for col_symbol, data in pair_correlations.items():
                        correlation_value = data.get('value', 0)
                        
                        if abs(correlation_value) >= self.high_threshold:
                            if correlation_value > 0:
                                high_correlations.append({
                                    'symbol': col_symbol,
                                    'correlation': correlation_value
                                })
                            else:
                                negative_correlations.append({
                                    'symbol': col_symbol,
                                    'correlation': correlation_value
                                })
                    
                    monitored_analysis[pair] = {
                        'high_positive_correlations': high_correlations,
                        'high_negative_correlations': negative_correlations,
                        'total_correlations': len(pair_correlations),
                        'risk_level': self._assess_risk_level(high_correlations, negative_correlations)
                    }
            
            return monitored_analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing monitored pairs: {e}")
            return {}
    
    def _normalize_pair_name(self, pair):
        """Normalize pair name for correlation matching"""
        # Handle common variations
        normalized = pair.upper()
        
        # Map common variations
        mapping = {
            'XAUUSD': 'GOLD',
            'US500': 'SPX500',
            'BTCUSD': 'BITCOIN'
        }
        
        return mapping.get(normalized, normalized)
    
    def _assess_risk_level(self, high_positive, high_negative):
        """Assess risk level based on correlation count"""
        total_high_correlations = len(high_positive) + len(high_negative)
        
        if total_high_correlations >= 5:
            return "HIGH"
        elif total_high_correlations >= 3:
            return "MEDIUM"
        elif total_high_correlations >= 1:
            return "LOW"
        else:
            return "MINIMAL"

# PART 4: CORRELATION SIGNAL MANAGER

class CorrelationSignalManager:
    """Manages correlation signals and file operations"""
    
    def __init__(self):
        self.output_file = CORRELATION_OUTPUT_FILE
        self.scraper = MyFXBookCorrelationScraper()
        self.analyzer = CorrelationAnalyzer()
        
    def update_correlation_data(self):
        """Main function to update correlation data"""
        try:
            logger.info("🚀 Starting correlation data update...")
            
            # Step 1: Scrape raw data
            raw_matrix = self.scraper.scrape_correlation_data()
            if not raw_matrix:
                logger.error("❌ Failed to scrape correlation data")
                self._handle_scraping_failure()
                return False
            
            # Step 2: Process into analysis
            processed_analysis = self.analyzer.process_correlation_data(raw_matrix)
            if not processed_analysis:
                logger.error("❌ Failed to process correlation data")
                return False
            
            # Step 3: Save to file
            if self.save_correlation_data(processed_analysis):
                logger.info("✅ Correlation data updated successfully")
                self._log_summary(processed_analysis)
                return True
            else:
                logger.error("❌ Failed to save correlation data")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error updating correlation data: {e}")
            return False
    
    def save_correlation_data(self, analysis):
        """Save processed correlation analysis to JSON file"""
        try:
            # Add metadata
            analysis['file_info'] = {
                'version': '1.0',
                'created_by': 'correlation_manager.py',
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
            
            # Write new analysis
            with open(self.output_file, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            logger.info(f"💾 Correlation data saved to {self.output_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving correlation data: {e}")
            return False
    
    def load_correlation_data(self):
        """Load correlation data from file (for testing/verification)"""
        try:
            if not os.path.exists(self.output_file):
                logger.warning(f"⚠️ Correlation file not found: {self.output_file}")
                return None
            
            with open(self.output_file, 'r') as f:
                analysis = json.load(f)
            
            # Check data freshness
            timestamp = datetime.fromisoformat(analysis['timestamp'])
            age_minutes = (datetime.now() - timestamp).total_seconds() / 60
            
            logger.info(f"📖 Loaded correlation data from {timestamp} ({age_minutes:.1f} minutes old)")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error loading correlation data: {e}")
            return None
    
    def _handle_scraping_failure(self):
        """Handle scraping failure - create fallback correlation file"""
        try:
            fallback_analysis = {
                'timestamp': datetime.now().isoformat(),
                'data_source': 'Fallback',
                'error': 'Scraping failed - no correlation data available',
                'thresholds': {
                    'high_correlation': self.analyzer.high_threshold,
                    'negative_correlation': self.analyzer.negative_threshold
                },
                'correlation_matrix': {},
                'insights': {},
                'warnings': [],
                'monitored_pairs_analysis': {}
            }
            
            self.save_correlation_data(fallback_analysis)
            logger.info("🔄 Created fallback correlation file")
            
        except Exception as e:
            logger.error(f"❌ Error creating fallback correlation data: {e}")
    
    def _log_summary(self, analysis):
        """Log summary of processed correlation analysis"""
        try:
            matrix = analysis.get('correlation_matrix', {})
            insights = analysis.get('insights', {})
            warnings = analysis.get('warnings', [])
            
            logger.info("📊 CORRELATION SUMMARY:")
            logger.info(f"   Currencies processed: {len(matrix)}")
            logger.info(f"   Total warnings: {len(warnings)}")
            
            # Log statistics
            stats = insights.get('correlation_statistics', {})
            if stats:
                logger.info(f"   Average correlation: {stats.get('avg_correlation', 0):.1f}%")
                logger.info(f"   High correlations: {stats.get('high_correlation_count', 0)}")
                logger.info(f"   Negative correlations: {stats.get('negative_correlation_count', 0)}")
            
            # Log top warnings
            for warning in warnings[:3]:  # Top 3 warnings
                logger.info(f"   ⚠️ {warning['type']}: {warning['pair']} ({warning['value']:.1f}%)")
            
        except Exception as e:
            logger.error(f"❌ Error logging summary: {e}")
    
    def get_correlation_status(self):
        """Get current correlation status for monitoring"""
        try:
            analysis = self.load_correlation_data()
            if not analysis:
                return "No correlation file found"
            
            timestamp = datetime.fromisoformat(analysis['timestamp'])
            age_minutes = (datetime.now() - timestamp).total_seconds() / 60
            
            matrix_size = len(analysis.get('correlation_matrix', {}))
            warnings_count = len(analysis.get('warnings', []))
            data_source = analysis.get('data_source', 'Unknown')
            
            if age_minutes > DATA_FRESHNESS_LIMIT_MINUTES:
                freshness = f"STALE ({age_minutes:.1f}m old)"
            else:
                freshness = f"FRESH ({age_minutes:.1f}m old)"
            
            return f"{data_source}: {matrix_size} currencies, {warnings_count} warnings, {freshness}"
            
        except Exception as e:
            return f"Error: {e}"

# PART 5: SCHEDULER AND MAIN FUNCTIONS

class CorrelationScheduler:
    """Handles scheduling of correlation updates"""
    
    def __init__(self):
        self.signal_manager = CorrelationSignalManager()
        self.running = False
        self.thread = None
        
    def start_scheduler(self):
        """Start the correlation update scheduler"""
        try:
            logger.info("🕐 Starting correlation scheduler...")
            
            # Schedule updates every 30 minutes
            schedule.every(SCRAPE_INTERVAL_MINUTES).minutes.do(self._scheduled_update)
            
            # Run initial update immediately
            logger.info("🚀 Running initial correlation update...")
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
        """Stop the correlation scheduler"""
        try:
            self.running = False
            schedule.clear()
            
            if self.thread:
                self.thread.join(timeout=5)
            
            logger.info("🛑 Correlation scheduler stopped")
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
        """Perform scheduled correlation update"""
        try:
            logger.info("⏰ Running scheduled correlation update...")
            
            success = self.signal_manager.update_correlation_data()
            
            if success:
                status = self.signal_manager.get_correlation_status()
                logger.info(f"✅ Scheduled update completed: {status}")
            else:
                logger.error("❌ Scheduled update failed")
                
        except Exception as e:
            logger.error(f"❌ Error in scheduled update: {e}")
    
    def force_update(self):
        """Force an immediate correlation update"""
        try:
            logger.info("🔥 Forcing immediate correlation update...")
            return self.signal_manager.update_correlation_data()
            
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
                'correlation_status': self.signal_manager.get_correlation_status()
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

def run_correlation_manager():
    """Main function to run correlation manager"""
    logger.info("="*60)
    logger.info("CORRELATION DATA MANAGER STARTED")
    logger.info("="*60)
    logger.info(f"Update interval: {SCRAPE_INTERVAL_MINUTES} minutes")
    logger.info(f"High correlation threshold: {HIGH_CORRELATION_THRESHOLD}%")
    logger.info(f"Negative correlation threshold: {NEGATIVE_CORRELATION_THRESHOLD}%")
    logger.info(f"Monitored pairs: {len(MONITORED_PAIRS)}")
    logger.info(f"Output file: {CORRELATION_OUTPUT_FILE}")
    logger.info("="*60)
    
    scheduler = CorrelationScheduler()
    
    try:
        # Start scheduler
        if not scheduler.start_scheduler():
            logger.error("❌ Failed to start scheduler")
            return
        
        logger.info("🎯 Correlation manager running. Press Ctrl+C to stop.")
        
        # Keep main thread alive
        while scheduler.running:
            try:
                time.sleep(60)  # Check every minute
                
                # Log status every 10 minutes
                if datetime.now().minute % 10 == 0:
                    status = scheduler.get_status()
                    logger.info(f"📊 Status: {status['correlation_status']}")
                    
            except KeyboardInterrupt:
                logger.info("🛑 Stop signal received")
                break
                
    except Exception as e:
        logger.error(f"❌ Error in main loop: {e}")
        
    finally:
        # Cleanup
        logger.info("🔄 Shutting down correlation manager...")
        scheduler.stop_scheduler()
        logger.info("✅ Correlation manager stopped")

def test_correlation_scraping():
    """Test function for correlation scraping"""
    logger.info("🧪 Testing correlation scraping...")
    
    signal_manager = CorrelationSignalManager()
    
    # Test scraping
    success = signal_manager.update_correlation_data()
    
    if success:
        # Load and display results
        analysis = signal_manager.load_correlation_data()
        if analysis:
            matrix = analysis.get('correlation_matrix', {})
            warnings = analysis.get('warnings', [])
            insights = analysis.get('insights', {})
            
            logger.info(f"✅ Test successful: {len(matrix)} currencies processed")
            logger.info(f"   Warnings generated: {len(warnings)}")
            
            # Show sample correlations
            sample_count = 0
            for row_symbol, correlations in matrix.items():
                if sample_count >= 3:  # Show first 3 currencies
                    break
                
                logger.info(f"   {row_symbol} correlations:")
                for col_symbol, data in list(correlations.items())[:5]:  # First 5 correlations
                    value = data.get('value', 0)
                    logger.info(f"     -> {col_symbol}: {value:.1f}%")
                
                sample_count += 1
            
            # Show top warnings
            for warning in warnings[:3]:  # Top 3 warnings
                logger.info(f"   ⚠️ {warning['type']}: {warning['pair']} ({warning['value']:.1f}%)")
                
        else:
            logger.error("❌ Could not load saved correlation data")
    else:
        logger.error("❌ Test failed")

def create_sample_correlation_data():
    """Create sample correlation data for testing main bot integration"""
    logger.info("📝 Creating sample correlation data...")
    
    # Create sample correlation matrix
    sample_currencies = ['AUDUSD', 'EURUSD', 'GBPUSD', 'USDCAD', 'USDCHF', 'AUDCAD', 'EURJPY']
    
    sample_matrix = {}
    for i, row_currency in enumerate(sample_currencies):
        sample_matrix[row_currency] = {}
        
        for j, col_currency in enumerate(sample_currencies):
            if row_currency == col_currency:
                # Self-correlation is always 100%
                correlation_value = 100.0
            else:
                # Generate realistic sample correlations
                if (row_currency.startswith('AUD') and col_currency.startswith('AUD')) or \
                   (row_currency.startswith('EUR') and col_currency.startswith('EUR')):
                    # High positive correlation for same base currency
                    correlation_value = 75.0 + (i + j) % 20
                elif (row_currency.startswith('USD') and col_currency.endswith('USD')) or \
                     (row_currency.endswith('USD') and col_currency.startswith('USD')):
                    # Negative correlation with USD pairs
                    correlation_value = -60.0 - (i + j) % 15
                else:
                    # Moderate correlations
                    correlation_value = 30.0 + (i * j) % 40 - 20
            
            sample_matrix[row_currency][col_currency] = {
                'value': correlation_value,
                'percentage': f"{correlation_value}%"
            }
    
    # Create sample analysis
    sample_analysis = {
        'timestamp': datetime.now().isoformat(),
        'data_source': 'Sample/Test',
        'thresholds': {
            'high_correlation': HIGH_CORRELATION_THRESHOLD,
            'negative_correlation': NEGATIVE_CORRELATION_THRESHOLD
        },
        'correlation_matrix': sample_matrix,
        'insights': {
            'highest_correlations': [
                {'pair': 'AUDUSD-AUDCAD', 'value': 85.0},
                {'pair': 'EURUSD-EURJPY', 'value': 82.0}
            ],
            'strongest_negative_correlations': [
                {'pair': 'AUDUSD-USDCAD', 'value': -72.0},
                {'pair': 'EURUSD-USDCHF', 'value': -68.0}
            ],
            'currency_summary': {},
            'correlation_statistics': {
                'total_pairs': 42,
                'avg_correlation': 15.5,
                'max_correlation': 100.0,
                'min_correlation': -75.0,
                'high_correlation_count': 8,
                'negative_correlation_count': 6
            }
        },
        'warnings': [
            {
                'type': 'HIGH_CORRELATION',
                'pair': 'AUDUSD-AUDCAD',
                'value': 85.0,
                'message': 'High correlation (85.0%) between AUDUSD and AUDCAD - risk of similar movements'
            },
            {
                'type': 'NEGATIVE_CORRELATION',
                'pair': 'AUDUSD-USDCAD',
                'value': -72.0,
                'message': 'Strong negative correlation (-72.0%) between AUDUSD and USDCAD - risk of opposite movements'
            }
        ],
        'monitored_pairs_analysis': {
            'AUDUSD': {
                'high_positive_correlations': [{'symbol': 'AUDCAD', 'correlation': 85.0}],
                'high_negative_correlations': [{'symbol': 'USDCAD', 'correlation': -72.0}],
                'total_correlations': 7,
                'risk_level': 'MEDIUM'
            }
        }
    }
    
    try:
        with open(CORRELATION_OUTPUT_FILE, 'w') as f:
            json.dump(sample_analysis, f, indent=2)
        
        logger.info(f"✅ Sample correlation data created: {CORRELATION_OUTPUT_FILE}")
        logger.info(f"   Currencies included: {len(sample_matrix)}")
        logger.info(f"   Warnings generated: {len(sample_analysis['warnings'])}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating sample correlation data: {e}")
        return False

def export_correlation_csv():
    """Export correlation matrix to CSV format"""
    try:
        logger.info("📊 Exporting correlation data to CSV...")
        
        signal_manager = CorrelationSignalManager()
        analysis = signal_manager.load_correlation_data()
        
        if not analysis:
            logger.error("❌ No correlation data found")
            return False
        
        matrix = analysis.get('correlation_matrix', {})
        if not matrix:
            logger.error("❌ No correlation matrix found")
            return False
        
        # Get all unique currencies
        all_currencies = set()
        for row_currency, correlations in matrix.items():
            all_currencies.add(row_currency)
            for col_currency in correlations.keys():
                all_currencies.add(col_currency)
        
        all_currencies = sorted(list(all_currencies))
        
        # Create CSV data
        csv_data = []
        
        # Header row
        header = ['Currency'] + all_currencies
        csv_data.append(header)
        
        # Data rows
        for row_currency in all_currencies:
            row = [row_currency]
            
            for col_currency in all_currencies:
                if row_currency in matrix and col_currency in matrix[row_currency]:
                    correlation_value = matrix[row_currency][col_currency].get('value', 0)
                    row.append(f"{correlation_value:.1f}%")
                else:
                    row.append("N/A")
            
            csv_data.append(row)
        
        # Save to CSV file
        csv_filename = CORRELATION_OUTPUT_FILE.replace('.json', '.csv')
        
        import csv
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(csv_data)
        
        logger.info(f"✅ Correlation matrix exported to: {csv_filename}")
        logger.info(f"   Matrix size: {len(all_currencies)}x{len(all_currencies)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error exporting correlation CSV: {e}")
        return False

def analyze_pair_correlation(pair1, pair2=None):
    """Analyze correlation for specific pair(s)"""
    try:
        logger.info(f"🔍 Analyzing correlation for {pair1}" + (f" vs {pair2}" if pair2 else ""))
        
        signal_manager = CorrelationSignalManager()
        analysis = signal_manager.load_correlation_data()
        
        if not analysis:
            logger.error("❌ No correlation data found")
            return
        
        matrix = analysis.get('correlation_matrix', {})
        
        # Normalize pair names
        analyzer = CorrelationAnalyzer()
        norm_pair1 = analyzer._normalize_pair_name(pair1)
        
        if norm_pair1 not in matrix:
            logger.error(f"❌ {pair1} not found in correlation matrix")
            return
        
        correlations = matrix[norm_pair1]
        
        if pair2:
            # Analyze specific pair correlation
            norm_pair2 = analyzer._normalize_pair_name(pair2)
            
            if norm_pair2 in correlations:
                correlation_data = correlations[norm_pair2]
                correlation_value = correlation_data.get('value', 0)
                
                logger.info(f"📊 {pair1} vs {pair2}: {correlation_value:.1f}%")
                
                if correlation_value >= HIGH_CORRELATION_THRESHOLD:
                    logger.info("   ⚠️ HIGH POSITIVE CORRELATION - Risk of similar movements")
                elif correlation_value <= NEGATIVE_CORRELATION_THRESHOLD:
                    logger.info("   ⚠️ HIGH NEGATIVE CORRELATION - Risk of opposite movements")
                else:
                    logger.info("   ✅ Normal correlation level")
            else:
                logger.error(f"❌ {pair2} not found in correlations for {pair1}")
        else:
            # Show all correlations for this pair
            logger.info(f"📊 All correlations for {pair1}:")
            
            # Sort by absolute correlation value
            sorted_correlations = sorted(
                correlations.items(),
                key=lambda x: abs(x[1].get('value', 0)),
                reverse=True
            )
            
            for col_currency, data in sorted_correlations[:10]:  # Top 10
                correlation_value = data.get('value', 0)
                logger.info(f"   {col_currency}: {correlation_value:.1f}%")
        
    except Exception as e:
        logger.error(f"❌ Error analyzing pair correlation: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'test':
            test_correlation_scraping()
        elif command == 'sample':
            create_sample_correlation_data()
        elif command == 'once':
            # Run update once and exit
            signal_manager = CorrelationSignalManager()
            success = signal_manager.update_correlation_data()
            sys.exit(0 if success else 1)
        elif command == 'csv':
            export_correlation_csv()
        elif command == 'analyze':
            if len(sys.argv) >= 3:
                pair1 = sys.argv[2]
                pair2 = sys.argv[3] if len(sys.argv) >= 4 else None
                analyze_pair_correlation(pair1, pair2)
            else:
                print("Usage: python correlation_manager.py analyze PAIR1 [PAIR2]")
                sys.exit(1)
        else:
            print("Usage: python correlation_manager.py [test|sample|once|csv|analyze]")
            sys.exit(1)
    else:
        # Run continuous scheduler
        run_correlation_manager()