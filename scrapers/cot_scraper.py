# ===== COT DATA MANAGER =====
# Scrapes and processes Commitments of Traders data from MyFXBook
# Collects historical data from multiple Tuesdays and consolidates into analysis-ready format

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
import schedule
import threading
import warnings
from typing import List, Dict, Optional, Tuple

# Suppress warnings
warnings.filterwarnings("ignore")

# ===== CONFIGURATION =====
# COT data settings
HISTORICAL_WEEKS = 6  # Number of historical weeks to collect
COT_OUTPUT_CSV = "cot_consolidated_data.csv"
COT_OUTPUT_JSON = "cot_consolidated_data.json"
COT_LOG_FILE = "cot_manager.log"

# Update settings
UPDATE_DAY = "friday"  # When to check for new COT data
UPDATE_TIME = "18:00"  # 6 PM (COT data usually published Friday afternoon)

# MyFXBook COT settings
COT_BASE_URL = "https://www.myfxbook.com/commitments-of-traders/historical-view"
COT_CSV_URL = "https://www.myfxbook.com/cot_statement.csv"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

# ===== LOGGING SETUP =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - COT_MANAGER - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(COT_LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# PART 2: COT DATA SCRAPER CLASS

class COTDataScraper:
    """Scraper for MyFXBook COT data"""
    
    def __init__(self):
        self.base_url = COT_BASE_URL
        self.csv_url = COT_CSV_URL
        self.headers = HEADERS
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
    def get_tuesday_dates(self, weeks_back: int = HISTORICAL_WEEKS) -> List[str]:
        """Generate list of Tuesday dates for the past N weeks"""
        tuesday_dates = []
        
        # Start from current date and work backwards
        current_date = datetime.now()
        
        # Find the most recent Tuesday
        days_since_tuesday = (current_date.weekday() - 1) % 7
        if days_since_tuesday == 0 and current_date.hour < 12:
            # If it's Tuesday morning, start from previous Tuesday
            days_since_tuesday = 7
            
        most_recent_tuesday = current_date - timedelta(days=days_since_tuesday)
        
        # Generate list of Tuesdays
        for i in range(weeks_back):
            tuesday = most_recent_tuesday - timedelta(weeks=i)
            
            # Format date as required by MyFXBook (YYYY-M-D)
            date_str = f"{tuesday.year}-{tuesday.month}-{tuesday.day}"
            tuesday_dates.append(date_str)
            
        logger.info(f"📅 Generated {len(tuesday_dates)} Tuesday dates: {tuesday_dates}")
        return tuesday_dates
    
    def scrape_cot_page(self, date_str: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Scrape COT page for given date and extract data from HTML tables
        Returns: (financial_dataframe, commodity_dataframe)
        """
        try:
            page_url = f"{self.base_url}/{date_str}"
            logger.info(f"🔄 Scraping COT page: {page_url}")
            
            response = self.session.get(page_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract Financial Futures data
            financial_df = self._extract_table_data(soup, "Financial Futures", date_str)
            
            # Extract Commodity Futures data
            commodity_df = self._extract_table_data(soup, "Commodity Futures", date_str)
            
            if financial_df is not None or commodity_df is not None:
                logger.info(f"✅ Extracted data for {date_str}")
                logger.info(f"   Financial: {len(financial_df) if financial_df is not None else 0} rows")
                logger.info(f"   Commodity: {len(commodity_df) if commodity_df is not None else 0} rows")
            else:
                logger.warning(f"⚠️ No data found for {date_str}")
                
            return financial_df, commodity_df
            
        except Exception as e:
            logger.error(f"❌ Error scraping COT page for {date_str}: {e}")
            return None, None
    
    def _extract_table_data(self, soup: BeautifulSoup, section_name: str, date_str: str) -> Optional[pd.DataFrame]:
        """Extract data from a specific table section - FIXED HTML PARSING"""
        try:
            # Look for h1 tags containing the section name
            section_headers = soup.find_all('h1')
            target_h1 = None
            
            for h1 in section_headers:
                h1_text = h1.get_text(strip=True)
                logger.debug(f"Found h1 text: '{h1_text}'")
                if section_name in h1_text:
                    target_h1 = h1
                    break
            
            if not target_h1:
                logger.warning(f"⚠️ Section '{section_name}' not found for {date_str}")
                logger.debug(f"Available h1 texts: {[h1.get_text(strip=True) for h1 in section_headers]}")
                return None
            
            # Navigate up to find the portlet containing this h1
            # Structure: h1 -> div.caption -> div.portlet-title -> div.portlet
            portlet = target_h1.find_parent('div', class_='portlet')
            if not portlet:
                logger.warning(f"⚠️ Portlet not found for '{section_name}' in {date_str}")
                return None
            
            # Find the table within this portlet's portlet-body
            portlet_body = portlet.find('div', class_='portlet-body')
            if not portlet_body:
                logger.warning(f"⚠️ Portlet body not found for '{section_name}' in {date_str}")
                return None
            
            # Find the table within the portlet body
            table = portlet_body.find('table', {'id': 'reportTable'})
            if not table:
                # Try finding any table in the portlet body
                table = portlet_body.find('table')
                if not table:
                    logger.warning(f"⚠️ Table not found for '{section_name}' in {date_str}")
                    return None
            
            # Extract table headers
            headers = self._extract_table_headers(table)
            if not headers:
                logger.warning(f"⚠️ No headers found for '{section_name}' in {date_str}")
                return None
            
            # Extract table rows
            rows_data = self._extract_table_rows(table)
            if not rows_data:
                logger.warning(f"⚠️ No data rows found for '{section_name}' in {date_str}")
                return None
            
            # Create DataFrame
            df = pd.DataFrame(rows_data, columns=headers)
            
            # Add metadata
            df['Date'] = date_str
            df['Data_Type'] = section_name
            
            logger.info(f"✅ Extracted {len(df)} rows from '{section_name}' table for {date_str}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error extracting '{section_name}' table for {date_str}: {e}")
            return None
    
    def _extract_table_headers(self, table) -> List[str]:
        """Extract column headers from table - ENHANCED FOR COT STRUCTURE"""
        try:
            headers = []
            thead = table.find('thead')
            if not thead:
                logger.debug("No thead found, looking for first tr")
                # Try to find headers in first row
                first_row = table.find('tr')
                if first_row:
                    th_elements = first_row.find_all(['th', 'td'])
                    for th in th_elements:
                        header_text = th.get_text(strip=True)
                        if header_text:
                            headers.append(header_text)
                return headers
            
            # Get all header rows
            header_rows = thead.find_all('tr')
            logger.debug(f"Found {len(header_rows)} header rows")
            
            if len(header_rows) >= 2:
                # COT tables have 2-row headers: grouped categories + Long/Short
                headers = self._parse_cot_headers(header_rows)
            else:
                # Fallback to simple header parsing
                if header_rows:
                    th_elements = header_rows[0].find_all('th')
                    for th in th_elements:
                        header_text = th.get_text(strip=True)
                        if header_text:
                            headers.append(header_text)
            
            logger.debug(f"Final headers: {headers}")
            return headers
            
        except Exception as e:
            logger.error(f"❌ Error extracting table headers: {e}")
            return []
    
    def _parse_cot_headers(self, header_rows) -> List[str]:
        """Parse COT-specific 2-row header structure"""
        try:
            # First row: grouped categories with colspan
            # Second row: Name, open interest, Long, Short, Long, Short, ...
            
            first_row = header_rows[0]
            second_row = header_rows[1]
            
            # Get the grouped categories from first row
            categories = []
            category_elements = first_row.find_all('th')
            
            for th in category_elements:
                category_text = th.get_text(strip=True)
                colspan = int(th.get('colspan', 1))
                
                if category_text and category_text not in ['', ' ']:
                    categories.append((category_text, colspan))
                else:
                    # Empty cells in first row
                    categories.append(('', colspan))
            
            # Get the column names from second row
            column_names = []
            column_elements = second_row.find_all('th')
            
            for th in column_elements:
                col_text = th.get_text(strip=True)
                column_names.append(col_text)
            
            logger.debug(f"Categories: {categories}")
            logger.debug(f"Column names: {column_names}")
            
            # Build final headers by combining categories with column names
            final_headers = []
            col_idx = 0
            
            for category, colspan in categories:
                if colspan == 1:
                    # Single column (Name, open interest)
                    if col_idx < len(column_names):
                        final_headers.append(column_names[col_idx])
                        col_idx += 1
                else:
                    # Multiple columns (Long/Short pairs)
                    for i in range(colspan):
                        if col_idx < len(column_names):
                            if category:
                                # Create descriptive header like "Non_Commercial_Long"
                                category_clean = category.replace(' Positions', '').replace('/', '_').replace(' ', '_')
                                col_name = column_names[col_idx]
                                combined_header = f"{category_clean}_{col_name}"
                                final_headers.append(combined_header)
                            else:
                                final_headers.append(column_names[col_idx])
                            col_idx += 1
            
            # Ensure we have basic columns
            if not final_headers:
                final_headers = column_names
            
            return final_headers
            
        except Exception as e:
            logger.error(f"❌ Error parsing COT headers: {e}")
            # Fallback to simple column names
            second_row = header_rows[1] if len(header_rows) > 1 else header_rows[0]
            headers = []
            for th in second_row.find_all('th'):
                header_text = th.get_text(strip=True)
                if header_text:
                    headers.append(header_text)
            return headers
    
    def _extract_table_rows(self, table) -> List[List[str]]:
        """Extract data rows from table - IMPROVED PARSING"""
        try:
            rows_data = []
            tbody = table.find('tbody')
            
            # If no tbody, look for tr elements directly in table
            if not tbody:
                tbody = table
            
            rows = tbody.find_all('tr')
            
            for row in rows:
                row_data = []
                cells = row.find_all(['td', 'th'])
                
                for cell in cells:
                    # Get text from cell, handling links
                    cell_text = cell.get_text(strip=True)
                    # Clean up the text (remove extra whitespace)
                    cell_text = ' '.join(cell_text.split())
                    row_data.append(cell_text)
                
                # Only add non-empty rows and skip header rows
                if row_data and len(row_data) > 1:
                    # Skip if this looks like a header row
                    first_cell = row_data[0].lower()
                    if first_cell not in ['name', 'long', 'short', '']:
                        rows_data.append(row_data)
            
            logger.debug(f"Extracted {len(rows_data)} data rows")
            
            # Log sample of extracted data for debugging
            if rows_data:
                logger.debug(f"Sample row: {rows_data[0]}")
            
            return rows_data
            
        except Exception as e:
            logger.error(f"❌ Error extracting table rows: {e}")
            return []
    
    def collect_historical_data(self, weeks_back: int = HISTORICAL_WEEKS) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        """
        Collect COT data for multiple historical dates
        Returns: (financial_dataframes, commodity_dataframes)
        """
        tuesday_dates = self.get_tuesday_dates(weeks_back)
        
        financial_dfs = []
        commodity_dfs = []
        
        for date_str in tuesday_dates:
            try:
                logger.info(f"🔄 Processing date: {date_str}")
                
                # Scrape page to get table data
                financial_df, commodity_df = self.scrape_cot_page(date_str)
                
                # Collect Financial data
                if financial_df is not None and not financial_df.empty:
                    financial_dfs.append(financial_df)
                
                # Collect Commodity data
                if commodity_df is not None and not commodity_df.empty:
                    commodity_dfs.append(commodity_df)
                
                # Be respectful to the server
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Error processing date {date_str}: {e}")
                continue
        
        logger.info(f"📊 Collected data: {len(financial_dfs)} Financial, {len(commodity_dfs)} Commodity datasets")
        return financial_dfs, commodity_dfs

# PART 3: COT DATA PROCESSOR CLASS

class COTDataProcessor:
    """Processes and consolidates COT data"""
    
    def __init__(self):
        # Define pair mappings for trading
        self.pair_mappings = {
            # Financial Futures to FX pairs
            'AUD': 'AUDUSD',
            'CAD': 'USDCAD', 
            'CHF': 'USDCHF',
            'EUR': 'EURUSD',
            'GBP': 'GBPUSD',
            'JPY': 'USDJPY',
            'NZD': 'NZDUSD',
            'MXN': 'USDMXN',
            'BRL': 'USDBRL',
            
            # Commodity Futures to trading symbols
            'Gold': 'XAUUSD',
            'Silver': 'XAGUSD',
            'Crude Oil': 'USOIL',
            'Copper': 'COPPER',
            'Platinum': 'XPTUSD',
            'Palladium': 'XPDUSD',
            'Aluminium': 'ALUMINUM'
        }
    
    def standardize_dataframe(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Standardize column names and structure based on data type"""
        try:
            # Create a copy to avoid modifying original
            df_clean = df.copy()
            
            # Log the original columns for debugging
            logger.debug(f"Original columns for {data_type}: {list(df_clean.columns)}")
            
            # Find the name and open interest columns (case insensitive)
            name_col = None
            oi_col = None
            
            for col in df_clean.columns:
                col_lower = col.lower()
                if 'name' in col_lower:
                    name_col = col
                elif 'open interest' in col_lower or 'open_interest' in col_lower:
                    oi_col = col
            
            if not name_col:
                logger.warning(f"⚠️ No 'Name' column found in {data_type} data")
                logger.debug(f"Available columns: {list(df_clean.columns)}")
                return pd.DataFrame()
            
            if not oi_col:
                logger.warning(f"⚠️ No 'Open Interest' column found in {data_type} data")
                logger.debug(f"Available columns: {list(df_clean.columns)}")
                # Don't return empty, continue without open interest
            
            # Standardize column names
            rename_dict = {name_col: 'Name'}
            if oi_col:
                rename_dict[oi_col] = 'Open_Interest'
            
            df_clean = df_clean.rename(columns=rename_dict)
            
            # Add trading pair mapping
            df_clean['Trading_Pair'] = df_clean['Name'].map(self.pair_mappings)
            
            # Convert numeric columns (skip Name, Date, Data_Type, Trading_Pair)
            skip_cols = ['Name', 'Date', 'Data_Type', 'Trading_Pair']
            numeric_cols = [col for col in df_clean.columns if col not in skip_cols]
            
            for col in numeric_cols:
                df_clean[col] = pd.to_numeric(df_clean[col].astype(str).str.replace(',', ''), errors='coerce')
            
            # Convert date to standard format
            if 'Date' in df_clean.columns:
                df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
            
            logger.info(f"✅ Standardized {data_type} dataframe: {len(df_clean)} rows, {len(df_clean.columns)} columns")
            return df_clean
            
        except Exception as e:
            logger.error(f"❌ Error standardizing {data_type} dataframe: {e}")
            return df
    
    def consolidate_data(self, financial_dfs: List[pd.DataFrame], 
                        commodity_dfs: List[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Consolidate COT data into separate Financial and Commodity DataFrames"""
        try:
            result = {'Financial': pd.DataFrame(), 'Commodity': pd.DataFrame()}
            
            # Process Financial data
            if financial_dfs:
                financial_clean_dfs = []
                for df in financial_dfs:
                    if not df.empty:
                        df_clean = self.standardize_dataframe(df, "Financial")
                        if not df_clean.empty:
                            financial_clean_dfs.append(df_clean)
                
                if financial_clean_dfs:
                    result['Financial'] = pd.concat(financial_clean_dfs, ignore_index=True)
                    result['Financial'] = result['Financial'].sort_values(['Date', 'Name'])
                    # Remove duplicates
                    result['Financial'] = result['Financial'].drop_duplicates(
                        subset=['Date', 'Name'], keep='last'
                    )
            
            # Process Commodity data
            if commodity_dfs:
                commodity_clean_dfs = []
                for df in commodity_dfs:
                    if not df.empty:
                        df_clean = self.standardize_dataframe(df, "Commodity")
                        if not df_clean.empty:
                            commodity_clean_dfs.append(df_clean)
                
                if commodity_clean_dfs:
                    result['Commodity'] = pd.concat(commodity_clean_dfs, ignore_index=True)
                    result['Commodity'] = result['Commodity'].sort_values(['Date', 'Name'])
                    # Remove duplicates
                    result['Commodity'] = result['Commodity'].drop_duplicates(
                        subset=['Date', 'Name'], keep='last'
                    )
            
            logger.info(f"✅ Consolidated data:")
            logger.info(f"   Financial: {len(result['Financial'])} records")
            logger.info(f"   Commodity: {len(result['Commodity'])} records")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error consolidating data: {e}")
            return {'Financial': pd.DataFrame(), 'Commodity': pd.DataFrame()}
    
    def add_calculated_fields(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Add calculated fields for analysis based on data type"""
        try:
            df_calc = df.copy()
            
            # Find Long and Short columns
            long_cols = [col for col in df_calc.columns if 'Long' in col and 'Trading_Pair' not in col]
            short_cols = [col for col in df_calc.columns if 'Short' in col]
            
            logger.debug(f"Found Long columns: {long_cols}")
            logger.debug(f"Found Short columns: {short_cols}")
            
            # For both Financial and Commodity data, calculate Non-Commercial net positions
            # Non-Commercial represents speculative positions (hedge funds, etc.)
            if long_cols and short_cols:
                # Assume first Long and Short columns are Non-Commercial
                if len(long_cols) >= 1 and len(short_cols) >= 1:
                    df_calc['Non_Commercial_Long'] = df_calc[long_cols[0]]
                    df_calc['Non_Commercial_Short'] = df_calc[short_cols[0]]
                    df_calc['Non_Commercial_Net'] = (
                        df_calc['Non_Commercial_Long'] - df_calc['Non_Commercial_Short']
                    )
                    
                    # Calculate percentage of open interest
                    if 'Open_Interest' in df_calc.columns:
                        df_calc['Non_Commercial_Net_Pct'] = (
                            df_calc['Non_Commercial_Net'] / df_calc['Open_Interest'] * 100
                        ).round(2)
                
                # For Commodity data, also look for Managed Money if available
                if data_type == "Commodity" and len(long_cols) >= 6 and len(short_cols) >= 6:
                    # Managed Money is typically the last pair in commodity data
                    df_calc['Managed_Money_Long'] = df_calc[long_cols[-1]]
                    df_calc['Managed_Money_Short'] = df_calc[short_cols[-1]]
                    df_calc['Managed_Money_Net'] = (
                        df_calc['Managed_Money_Long'] - df_calc['Managed_Money_Short']
                    )
                    
                    if 'Open_Interest' in df_calc.columns:
                        df_calc['Managed_Money_Net_Pct'] = (
                            df_calc['Managed_Money_Net'] / df_calc['Open_Interest'] * 100
                        ).round(2)
            
            logger.info(f"✅ Added calculated fields for {data_type} data")
            return df_calc
            
        except Exception as e:
            logger.error(f"❌ Error adding calculated fields for {data_type}: {e}")
            return df

# PART 4: COT DATA MANAGER

class COTDataManager:
    """Main COT data management class"""
    
    def __init__(self):
        self.scraper = COTDataScraper()
        self.processor = COTDataProcessor()
        self.output_csv = COT_OUTPUT_CSV
        self.output_json = COT_OUTPUT_JSON
        
    def update_cot_data(self, weeks_back: int = HISTORICAL_WEEKS) -> bool:
        """Main function to update COT data"""
        try:
            logger.info("🚀 Starting COT data update...")
            
            # Step 1: Collect historical data
            financial_dfs, commodity_dfs = self.scraper.collect_historical_data(weeks_back)
            
            if not financial_dfs and not commodity_dfs:
                logger.error("❌ No data collected")
                return False
            
            # Step 2: Consolidate data (keep Financial and Commodity separate)
            consolidated_data = self.processor.consolidate_data(financial_dfs, commodity_dfs)
            
            if consolidated_data['Financial'].empty and consolidated_data['Commodity'].empty:
                logger.error("❌ Failed to consolidate data")
                return False
            
            # Step 3: Add calculated fields
            final_data = {}
            for data_type, df in consolidated_data.items():
                if not df.empty:
                    final_data[data_type] = self.processor.add_calculated_fields(df, data_type)
                else:
                    final_data[data_type] = df
            
            # Step 4: Save outputs
            self.save_data(final_data)
            
            logger.info("✅ COT data update completed successfully")
            self._log_summary(final_data)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating COT data: {e}")
            return False
    
    def save_data(self, data_dict: Dict[str, pd.DataFrame]) -> bool:
        """Save data to CSV and JSON formats with separate worksheets/sections"""
        try:
            # Save separate CSV files
            for data_type, df in data_dict.items():
                if not df.empty:
                    csv_filename = f"cot_{data_type.lower()}_data.csv"
                    df.to_csv(csv_filename, index=False)
                    logger.info(f"💾 Saved {data_type} CSV: {csv_filename}")
            
            # Save combined Excel file with separate worksheets
            excel_filename = self.output_csv.replace('.csv', '.xlsx')
            try:
                with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
                    for data_type, df in data_dict.items():
                        if not df.empty:
                            df.to_excel(writer, sheet_name=data_type, index=False)
                logger.info(f"💾 Saved Excel with worksheets: {excel_filename}")
            except ImportError:
                logger.warning("⚠️ openpyxl not available, skipping Excel export")
            
            # Save JSON with separate sections
            json_data = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'data_types': list(data_dict.keys()),
                    'total_records': sum(len(df) for df in data_dict.values()),
                    'date_ranges': {},
                    'instruments': {}
                },
                'data': {}
            }
            
            # Add metadata and data for each type
            for data_type, df in data_dict.items():
                if not df.empty:
                    json_data['metadata']['date_ranges'][data_type] = {
                        'earliest': df['Date'].min().isoformat() if 'Date' in df.columns else None,
                        'latest': df['Date'].max().isoformat() if 'Date' in df.columns else None
                    }
                    json_data['metadata']['instruments'][data_type] = df['Name'].unique().tolist() if 'Name' in df.columns else []
                    json_data['data'][data_type] = df.to_dict('records')
            
            with open(self.output_json, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            logger.info(f"💾 Saved JSON: {self.output_json}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving data: {e}")
            return False
    
    def load_data(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Load existing COT data"""
        try:
            result = {}
            
            # Try loading separate CSV files first
            for data_type in ['Financial', 'Commodity']:
                csv_filename = f"cot_{data_type.lower()}_data.csv"
                if os.path.exists(csv_filename):
                    df = pd.read_csv(csv_filename)
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                    result[data_type] = df
                    logger.info(f"📖 Loaded {len(df)} {data_type} COT records")
            
            # If no separate files, try JSON
            if not result and os.path.exists(self.output_json):
                with open(self.output_json, 'r') as f:
                    json_data = json.load(f)
                
                for data_type, records in json_data.get('data', {}).items():
                    df = pd.DataFrame(records)
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                    result[data_type] = df
                    logger.info(f"📖 Loaded {len(df)} {data_type} COT records from JSON")
            
            if not result:
                logger.warning("⚠️ No existing COT data found")
                return None
                
            return result
                
        except Exception as e:
            logger.error(f"❌ Error loading COT data: {e}")
            return None
    
    def _log_summary(self, data_dict: Dict[str, pd.DataFrame]):
        """Log summary of COT data"""
        try:
            logger.info("📊 COT DATA SUMMARY:")
            
            for data_type, df in data_dict.items():
                if df.empty:
                    continue
                
                logger.info(f"   === {data_type.upper()} DATA ===")
                logger.info(f"   Records: {len(df)}")
                
                if 'Date' in df.columns:
                    logger.info(f"   Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
                    latest_date = df['Date'].max()
                    latest_data = df[df['Date'] == latest_date]
                    
                    logger.info(f"   Latest data ({latest_date.date()}):")
                    for _, row in latest_data.head(5).iterrows():
                        net_position = row.get('Non_Commercial_Net', 0)
                        trading_pair = row.get('Trading_Pair', row.get('Name', 'Unknown'))
                        direction = "📈" if net_position > 0 else "📉" if net_position < 0 else "➡️"
                        logger.info(f"     {direction} {trading_pair}: {net_position:,.0f} net")
                
                if 'Name' in df.columns:
                    instruments = df['Name'].unique()
                    logger.info(f"   Instruments ({len(instruments)}): {', '.join(instruments[:10])}{'...' if len(instruments) > 10 else ''}")
            
        except Exception as e:
            logger.error(f"❌ Error logging summary: {e}")
    
    def get_cot_status(self) -> str:
        """Get current COT data status"""
        try:
            data_dict = self.load_data()
            if not data_dict:
                return "No COT data available"
            
            status_parts = []
            total_records = 0
            latest_date = None
            
            for data_type, df in data_dict.items():
                if not df.empty:
                    total_records += len(df)
                    if 'Date' in df.columns:
                        df_latest = df['Date'].max()
                        if latest_date is None or df_latest > latest_date:
                            latest_date = df_latest
                    
                    instruments = len(df['Name'].unique()) if 'Name' in df.columns else 0
                    status_parts.append(f"{data_type}: {len(df)} records, {instruments} instruments")
            
            if latest_date:
                age_days = (datetime.now() - latest_date).days
                age_str = f"Latest: {latest_date.date()} ({age_days} days old)"
            else:
                age_str = "No date info"
            
            return f"COT Data - {', '.join(status_parts)}, {age_str}"
            
        except Exception as e:
            return f"Error: {e}"

# PART 5: MAIN FUNCTIONS

def run_cot_update():
    """Run COT data update"""
    logger.info("="*60)
    logger.info("COT DATA MANAGER STARTED")
    logger.info("="*60)
    logger.info(f"Historical weeks: {HISTORICAL_WEEKS}")
    logger.info(f"Output files: {COT_OUTPUT_CSV}, {COT_OUTPUT_JSON}")
    logger.info("="*60)
    
    manager = COTDataManager()
    success = manager.update_cot_data()
    
    if success:
        status = manager.get_cot_status()
        logger.info(f"✅ Update completed: {status}")
    else:
        logger.error("❌ Update failed")

def test_cot_scraping():
    """Test COT data scraping"""
    logger.info("🧪 Testing COT data scraping...")
    
    manager = COTDataManager()
    
    # Test with just 2 weeks for faster testing
    success = manager.update_cot_data(weeks_back=2)
    
    if success:
        data_dict = manager.load_data()
        if data_dict:
            logger.info("✅ Test successful:")
            for data_type, df in data_dict.items():
                logger.info(f"   {data_type}: {len(df)} records collected")
                
                # Show sample data
                if not df.empty:
                    logger.info(f"   Sample {data_type} data:")
                    for _, row in df.head(3).iterrows():
                        date_str = row['Date'].date() if 'Date' in row and pd.notna(row['Date']) else 'No date'
                        name = row.get('Name', 'Unknown')
                        net = row.get('Non_Commercial_Net', 'N/A')
                        trading_pair = row.get('Trading_Pair', 'No mapping')
                        logger.info(f"     {date_str} {name} ({trading_pair}): Net={net}")
        else:
            logger.error("❌ Could not load saved data")
    else:
        logger.error("❌ Test failed")

def test_single_date():
    """Test scraping for a single known date"""
    logger.info("🧪 Testing single date scraping...")
    
    scraper = COTDataScraper()
    test_date = "2025-7-22"  # Use the date from your example
    
    logger.info(f"Testing date: {test_date}")
    financial_df, commodity_df = scraper.scrape_cot_page(test_date)
    
    if financial_df is not None:
        logger.info(f"✅ Financial data extracted: {len(financial_df)} rows")
        logger.info(f"   Columns: {list(financial_df.columns)}")
        if not financial_df.empty:
            logger.info(f"   Sample row: {financial_df.iloc[0].to_dict()}")
    else:
        logger.error("❌ No financial data extracted")
    
    if commodity_df is not None:
        logger.info(f"✅ Commodity data extracted: {len(commodity_df)} rows")
        logger.info(f"   Columns: {list(commodity_df.columns)}")
        if not commodity_df.empty:
            logger.info(f"   Sample row: {commodity_df.iloc[0].to_dict()}")
    else:
        logger.error("❌ No commodity data extracted")

def create_sample_cot_data():
    """Create sample COT data for testing"""
    logger.info("📝 Creating sample COT data...")
    
    # Sample financial data
    sample_financial = pd.DataFrame({
        'Date': ['2025-07-22', '2025-07-22', '2025-07-15', '2025-07-15'],
        'Data_Type': ['Financial', 'Financial', 'Financial', 'Financial'],
        'Name': ['AUD', 'EUR', 'AUD', 'EUR'],
        'Open_Interest': [151338, 779031, 148000, 765000],
        'Long': [25539, 224979, 23000, 220000],
        'Short': [95629, 117442, 98000, 125000],
        'Trading_Pair': ['AUDUSD', 'EURUSD', 'AUDUSD', 'EURUSD']
    })
    
    # Sample commodity data
    sample_commodity = pd.DataFrame({
        'Date': ['2025-07-22', '2025-07-22', '2025-07-15', '2025-07-15'],
        'Data_Type': ['Commodity', 'Commodity', 'Commodity', 'Commodity'],
        'Name': ['Gold', 'Silver', 'Gold', 'Silver'],
        'Open_Interest': [437662, 163567, 435000, 160000],
        'Long': [258631, 82747, 255000, 80000],
        'Short': [56651, 19347, 60000, 22000],
        'Trading_Pair': ['XAUUSD', 'XAGUSD', 'XAUUSD', 'XAGUSD']
    })
    
    # Add calculated fields
    for df in [sample_financial, sample_commodity]:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Non_Commercial_Net'] = df['Long'] - df['Short']
        df['Non_Commercial_Net_Pct'] = (df['Non_Commercial_Net'] / df['Open_Interest'] * 100).round(2)
    
    sample_data = {
        'Financial': sample_financial,
        'Commodity': sample_commodity
    }
    
    manager = COTDataManager()
    success = manager.save_data(sample_data)
    
    if success:
        logger.info("✅ Sample COT data created successfully")
        status = manager.get_cot_status()
        logger.info(f"   Status: {status}")
    else:
        logger.error("❌ Failed to create sample data")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'test':
            test_cot_scraping()
        elif command == 'single':
            test_single_date()
        elif command == 'update':
            run_cot_update()
        elif command == 'sample':
            create_sample_cot_data()
        else:
            print("Usage: python cot_data_manager.py [test|single|update|sample]")
            sys.exit(1)
    else:
        # Default: run update
        run_cot_update()