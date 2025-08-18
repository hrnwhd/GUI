# ===== FIXED ECONOMIC CALENDAR SCRAPER =====
# Updated scraper based on actual HTML structure

import requests
from bs4 import BeautifulSoup
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedEconomicCalendarScraper:
    """Fixed scraper for MyFXBook economic calendar data"""
    
    def __init__(self):
        self.url = "https://www.myfxbook.com/forex-economic-calendar"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
    def scrape_calendar_data(self, target_dates: List[str]) -> List[Dict]:
        """
        Scrape economic calendar data for specified dates
        Args:
            target_dates: List of dates in format "Wednesday, Jul 30, 2025"
        Returns:
            List of economic events
        """
        try:
            logger.info(f"🔄 Fetching economic calendar data for {len(target_dates)} dates...")
            
            # Make request
            session = requests.Session()
            session.headers.update(self.headers)
            response = session.get(self.url, timeout=30)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Debug: Check what we found
            logger.info(f"📄 Page title: {soup.title.string if soup.title else 'No title'}")
            
            # Look for calendar rows directly (no table wrapper needed)
            all_rows = soup.find_all('tr')
            logger.info(f"🔍 Found {len(all_rows)} total rows in page")
            
            # Filter for calendar-related rows
            date_rows = soup.find_all('tr', class_='economicCalendarDateRow')
            event_rows = soup.find_all('tr', class_='economicCalendarRow')
            
            logger.info(f"📅 Found {len(date_rows)} date header rows")
            logger.info(f"📋 Found {len(event_rows)} event rows")
            
            # Extract events for target dates
            events = self._extract_events_by_date(soup, target_dates, date_rows, event_rows)
            
            logger.info(f"✅ Successfully scraped {len(events)} economic events")
            return events
            
        except requests.RequestException as e:
            logger.error(f"❌ Network error scraping calendar: {e}")
            return []
        except Exception as e:
            logger.error(f"❌ Error scraping calendar: {e}")
            return []
    
    def _extract_events_by_date(self, soup: BeautifulSoup, target_dates: List[str], 
                               date_rows: List, event_rows: List) -> List[Dict]:
        """Extract events for specific target dates"""
        events = []
        
        try:
            # Create a mapping of date positions in the page
            date_positions = {}
            
            # Find positions of target dates
            all_rows = soup.find_all('tr')
            for i, row in enumerate(all_rows):
                if 'economicCalendarDateRow' in row.get('class', []):
                    date_cell = row.find('td')
                    if date_cell:
                        date_text = date_cell.get_text(strip=True)
                        logger.info(f"🔍 Found date header: '{date_text}'")
                        if date_text in target_dates:
                            date_positions[date_text] = i
                            logger.info(f"📅 Target date found: {date_text} at position {i}")
            
            # If no target dates found, let's see what dates are available
            if not date_positions:
                logger.warning("⚠️ No target dates found. Available dates:")
                for row in date_rows:
                    date_cell = row.find('td')
                    if date_cell:
                        available_date = date_cell.get_text(strip=True)
                        logger.info(f"   Available: '{available_date}'")
                return []
            
            # Extract events following each target date
            for target_date, date_pos in date_positions.items():
                logger.info(f"🔄 Processing events for {target_date}")
                
                # Find events after this date header until next date header
                current_date = target_date
                events_found = 0
                
                for i in range(date_pos + 1, len(all_rows)):
                    row = all_rows[i]
                    
                    # Stop if we hit another date header
                    if 'economicCalendarDateRow' in row.get('class', []):
                        break
                    
                    # Process if this is an event row
                    if 'economicCalendarRow' in row.get('class', []):
                        event_data = self._extract_event_data(row, current_date)
                        if event_data:
                            events.append(event_data)
                            events_found += 1
                
                logger.info(f"📊 Found {events_found} events for {target_date}")
            
            return events
            
        except Exception as e:
            logger.error(f"❌ Error extracting events by date: {e}")
            return []
    
    def _extract_event_data(self, row, event_date: str) -> Optional[Dict]:
        """Extract data from a single event row"""
        try:
            # Get row ID
            row_id = row.get('id', '').replace('calRow', '') if row.get('id') else 'unknown'
            
            cells = row.find_all('td')
            if len(cells) < 6:
                return None
            
            event_data = {
                'row_id': row_id,
                'event_date': event_date,
                'time': 'N/A',
                'time_until': 'N/A',
                'country': 'N/A',
                'currency': 'N/A',
                'event_name': 'N/A',
                'impact': 'N/A',
                'previous': 'N/A',
                'consensus': 'N/A',
                'actual': 'N/A',
                'scraped_timestamp': datetime.now().isoformat()
            }
            
            # Extract time (first cell - div with class calendarDateTd)
            time_div = cells[0].find('div', class_='calendarDateTd')
            if time_div:
                event_data['time'] = time_div.get_text(strip=True)
            
            # Extract time until (second cell - span with class calendarLeft)
            time_until_span = cells[1].find('span', class_='calendarLeft')
            if time_until_span:
                event_data['time_until'] = time_until_span.get_text(strip=True)
            
            # Extract country (third cell - look for flag class)
            flag_span = cells[2].find('span', class_='flag')
            if flag_span:
                flag_classes = flag_span.get('class', [])
                for cls in flag_classes:
                    if cls.startswith('flag-icon-'):
                        country_code = cls.replace('flag-icon-', '').upper()
                        event_data['country'] = country_code
                        break
            
            # Extract currency (fourth cell)
            event_data['currency'] = cells[3].get_text(strip=True)
            
            # Extract event name (fifth cell - link text + span)
            event_link = cells[4].find('a', class_='calendar-event-link')
            if event_link:
                event_name = event_link.get_text(strip=True)
                # Add any additional span text (like "(Jul/25)")
                additional_span = cells[4].find('span')
                if additional_span:
                    additional_text = additional_span.get_text(strip=True)
                    if additional_text:
                        event_name += f" {additional_text}"
                event_data['event_name'] = event_name
            
            # Extract impact (sixth cell - look for impact_low/medium/high classes)
            impact_div = cells[5].find('div')
            if impact_div:
                impact_classes = impact_div.get('class', [])
                for cls in impact_classes:
                    if cls.startswith('impact_'):
                        impact_level = cls.replace('impact_', '').capitalize()
                        event_data['impact'] = impact_level
                        break
                # Fallback to text content
                if event_data['impact'] == 'N/A':
                    event_data['impact'] = impact_div.get_text(strip=True)
            
            # Extract previous value (seventh cell)
            if len(cells) > 6:
                previous_span = cells[6].find('span', class_='previousCell')
                if previous_span:
                    event_data['previous'] = previous_span.get_text(strip=True)
            
            # Extract consensus (eighth cell)
            if len(cells) > 7:
                # Look for div with align-center class first
                consensus_div = cells[7].find('div', class_='align-center')
                if consensus_div:
                    event_data['consensus'] = consensus_div.get_text(strip=True)
                else:
                    # Fallback to cell text
                    consensus_text = cells[7].get_text(strip=True)
                    if consensus_text:
                        event_data['consensus'] = consensus_text
            
            # Extract actual value (ninth cell)
            if len(cells) > 8:
                actual_span = cells[8].find('span', class_='actualCell')
                if actual_span:
                    actual_text = actual_span.get_text(strip=True)
                    if actual_text:
                        event_data['actual'] = actual_text
            
            # Log what we extracted
            logger.debug(f"📋 Extracted: {event_data['time']} - {event_data['currency']} - {event_data['event_name']} ({event_data['impact']})")
            
            return event_data
            
        except Exception as e:
            logger.warning(f"⚠️ Error extracting event data: {e}")
            return None

# Test function
def test_fixed_scraper():
    """Test the fixed scraper"""
    logger.info("🧪 Testing fixed economic calendar scraper...")
    
    # Generate target dates
    target_dates = []
    current_date = datetime.now()
    
    for i in range(3):  # Next 3 trading days
        while current_date.weekday() >= 5:  # Skip weekends
            current_date += timedelta(days=1)
        
        formatted_date = current_date.strftime("%A, %b %d, %Y")
        target_dates.append(formatted_date)
        current_date += timedelta(days=1)
    
    logger.info(f"🎯 Target dates: {target_dates}")
    
    # Test scraper
    scraper = FixedEconomicCalendarScraper()
    events = scraper.scrape_calendar_data(target_dates)
    
    logger.info(f"📊 RESULTS:")
    logger.info(f"   Total events found: {len(events)}")
    
    if events:
        # Group by impact
        impact_counts = {}
        currency_counts = {}
        
        for event in events:
            impact = event['impact']
            currency = event['currency']
            
            impact_counts[impact] = impact_counts.get(impact, 0) + 1
            currency_counts[currency] = currency_counts.get(currency, 0) + 1
        
        logger.info(f"   Impact breakdown: {impact_counts}")
        logger.info(f"   Currency breakdown: {currency_counts}")
        
        # Show first few events
        logger.info("📋 Sample events:")
        for event in events[:5]:
            logger.info(f"   {event['event_date']} {event['time']} - {event['currency']} {event['event_name']} ({event['impact']})")
            if event['previous'] != 'N/A':
                logger.info(f"     Previous: {event['previous']}, Consensus: {event['consensus']}")
        
        # Save sample output
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'total_events': len(events),
            'target_dates': target_dates,
            'events': events
        }
        
        with open('test_calendar_output.json', 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info("✅ Test completed successfully - saved to test_calendar_output.json")
        return True
    else:
        logger.error("❌ No events found - check date format or page structure")
        return False

if __name__ == "__main__":
    test_fixed_scraper()