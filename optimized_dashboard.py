# ===== FIXED FLASK DASHBOARD WITH ENHANCED WEBHOOK HANDLING =====
# This version fixes the update delay issues and improves real-time updates

from flask import Flask, render_template, jsonify, request, redirect, url_for
import json
import os
import sys
import time
import threading
from datetime import datetime, timedelta
import logging
from collections import deque
from flask_cors import CORS
import queue

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ===== ENHANCED REAL-TIME DATA STORAGE =====
class EnhancedRealTimeDataStore:
    """Enhanced storage for real-time webhook data with better update tracking"""
    
    def __init__(self):
        # Live data storage
        self.live_data = {
            "timestamp": datetime.now().isoformat(),
            "robot_status": "Disconnected",
            "account": {
                "balance": 0,
                "equity": 0,
                "margin": 0,
                "free_margin": 0,
                "margin_level": 0,
                "profit": 0
            },
            "active_trades": 0,
            "active_batches": 0,
            "total_trades": 0,
            "emergency_stop": False,
            "drawdown_percent": 0,
            "last_signal_time": None,
            "next_analysis": None,
            "batches": [],
            "pairs_status": {},
            "mt5_connected": False
        }
        
        # Account history for charts (last 1000 points)
        self.account_history = {
            "timestamps": [],
            "balance": [],
            "equity": [],
            "profit": [],
            "drawdown": []
        }
        
        # Trade log (last 500 trades)
        self.trade_log = deque(maxlen=500)
        
        # Recent signals (last 100 signals)
        self.recent_signals = deque(maxlen=100)
        
        # Connection status
        self.last_update = datetime.now()
        self.connection_timeout = 30  # seconds
        self.update_count = 0
        self.error_count = 0
        
        # Initialize with some sample data for testing
        self._initialize_sample_data()
        
    def _initialize_sample_data(self):
        """Initialize with sample data for testing"""
        # Add some sample account history
        base_time = datetime.now() - timedelta(hours=1)
        for i in range(10):
            timestamp = base_time + timedelta(minutes=i*6)
            balance = 10000 + (i * 50)
            equity = balance + (i * 25)
            
            self.account_history["timestamps"].append(timestamp.isoformat())
            self.account_history["balance"].append(balance)
            self.account_history["equity"].append(equity)
            self.account_history["profit"].append(equity - balance)
            self.account_history["drawdown"].append(max(0, (10000 - equity) / 10000 * 100))
        
        # Add some sample trades
        for i in range(5):
            trade = {
                'timestamp': (datetime.now() - timedelta(minutes=i*10)).isoformat(),
                'symbol': ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDCAD', 'XAUUSD'][i],
                'direction': ['long', 'short'][i % 2],
                'volume': 0.1 + (i * 0.05),
                'entry_price': 1.0500 + (i * 0.0010),
                'tp': 1.0550 + (i * 0.0010),
                'profit': (i - 2) * 15.50,
                'layer': 1,
                'is_martingale': False,
                'comment': f'BM0{i+1}_TEST_B01'
            }
            self.trade_log.appendleft(trade)
        
        # Add some sample signals
        for i in range(3):
            signal = {
                'timestamp': (datetime.now() - timedelta(minutes=i*15)).isoformat(),
                'symbol': ['BTCUSD', 'US500', 'XAUUSD'][i],
                'direction': ['long', 'short', 'long'][i],
                'entry_price': 50000 + (i * 1000),
                'tp': 50500 + (i * 1000),
                'sl_distance_pips': 20,
                'tp_distance_pips': 50,
                'risk_profile': ['High', 'Medium', 'Low'][i],
                'adx_value': 25 + i * 5,
                'rsi': 45 + i * 10,
                'timeframes_aligned': i + 1,
                'is_initial': True
            }
            self.recent_signals.appendleft(signal)
    
    def update_live_data(self, data):
        """Update live data from webhook"""
        try:
            self.live_data.update(data)
            self.live_data["timestamp"] = datetime.now().isoformat()
            self.last_update = datetime.now()
            self.update_count += 1
            logger.info(f"Live data updated via webhook (update #{self.update_count})")
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error updating live data: {e}")
    
    def add_account_update(self, data):
        """Add account update for chart data"""
        try:
            now = datetime.now()
            
            # Add new data point
            self.account_history["timestamps"].append(now.isoformat())
            self.account_history["balance"].append(data.get("balance", 0))
            self.account_history["equity"].append(data.get("equity", 0))
            self.account_history["profit"].append(data.get("profit", 0))
            self.account_history["drawdown"].append(data.get("drawdown", 0))
            
            # Keep only last 1000 points
            for key in ['timestamps', 'balance', 'equity', 'profit', 'drawdown']:
                if len(self.account_history[key]) > 1000:
                    self.account_history[key] = self.account_history[key][-1000:]
            
            logger.info("Account history updated")
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error adding account update: {e}")
    
    def add_trade_event(self, data):
        """Add trade event to log"""
        try:
            trade_data = {
                'timestamp': data.get('timestamp', datetime.now().isoformat()),
                'event_type': data.get('event_type', 'executed'),
                'symbol': data.get('symbol'),
                'direction': data.get('direction'),
                'volume': data.get('volume'),
                'entry_price': data.get('entry_price'),
                'tp': data.get('tp'),
                'sl': data.get('sl'),
                'order_id': data.get('order_id'),
                'layer': data.get('layer', 1),
                'is_martingale': data.get('is_martingale', False),
                'profit': data.get('profit', 0),
                'comment': data.get('comment', ''),
                'sl_distance_pips': data.get('sl_distance_pips', 0)
            }
            
            self.trade_log.appendleft(trade_data)  # Most recent first
            logger.info(f"Added trade event: {data.get('symbol')} {data.get('direction')}")
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error adding trade event: {e}")
    
    def add_signal(self, data):
        """Add signal to recent signals"""
        try:
            signal_data = {
                'timestamp': data.get('timestamp', datetime.now().isoformat()),
                'symbol': data.get('symbol'),
                'direction': data.get('direction'),
                'entry_price': data.get('entry_price'),
                'tp': data.get('tp'),
                'sl_distance_pips': data.get('sl_distance_pips'),
                'tp_distance_pips': data.get('tp_distance_pips'),
                'risk_profile': data.get('risk_profile'),
                'adx_value': data.get('adx_value'),
                'rsi': data.get('rsi'),
                'timeframes_aligned': data.get('timeframes_aligned', 1),
                'is_initial': data.get('is_initial', True)
            }
            
            self.recent_signals.appendleft(signal_data)  # Most recent first
            logger.info(f"Added signal: {data.get('symbol')} {data.get('direction')}")
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error adding signal: {e}")
    
    def is_connected(self):
        """Check if bot is connected based on last update time"""
        time_since_update = (datetime.now() - self.last_update).total_seconds()
        return time_since_update < self.connection_timeout
    
    def get_connection_status(self):
        """Get connection status info"""
        time_since_update = (datetime.now() - self.last_update).total_seconds()
        return {
            'connected': self.is_connected(),
            'last_update': self.last_update.isoformat(),
            'seconds_since_update': int(time_since_update),
            'update_count': self.update_count,
            'error_count': self.error_count
        }
    
    def get_status_summary(self):
        """Get status summary for debugging"""
        return {
            'total_updates': self.update_count,
            'total_errors': self.error_count,
            'last_update': self.last_update.isoformat(),
            'connected': self.is_connected(),
            'trade_count': len(self.trade_log),
            'signal_count': len(self.recent_signals),
            'chart_points': len(self.account_history['timestamps'])
        }

# Initialize data store
data_store = EnhancedRealTimeDataStore()

# ===== ENHANCED WEBHOOK ENDPOINTS =====
@app.route('/webhook/live_data', methods=['POST'])
def webhook_live_data():
    """Receive live data updates from bot"""
    try:
        data = request.json
        if not data:
            logger.warning("No data received in live_data webhook")
            return jsonify({"status": "error", "message": "No data received"}), 400
        
        data_store.update_live_data(data)
        logger.info("Live data webhook received and processed successfully")
        return jsonify({"status": "success", "message": "Live data updated"})
        
    except Exception as e:
        data_store.error_count += 1
        logger.error(f"Webhook live_data error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/webhook/account_update', methods=['POST'])
def webhook_account_update():
    """Receive account updates for charts"""
    try:
        data = request.json
        if not data:
            logger.warning("No data received in account_update webhook")
            return jsonify({"status": "error", "message": "No data received"}), 400
        
        data_store.add_account_update(data)
        logger.info("Account update webhook received and processed successfully")
        return jsonify({"status": "success", "message": "Account updated"})
        
    except Exception as e:
        data_store.error_count += 1
        logger.error(f"Webhook account_update error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/webhook/trade_event', methods=['POST'])
def webhook_trade_event():
    """Receive trade execution events"""
    try:
        data = request.json
        if not data:
            logger.warning("No data received in trade_event webhook")
            return jsonify({"status": "error", "message": "No data received"}), 400
        
        data_store.add_trade_event(data)
        logger.info(f"Trade event webhook: {data.get('symbol')} {data.get('direction')}")
        return jsonify({"status": "success", "message": "Trade event added"})
        
    except Exception as e:
        data_store.error_count += 1
        logger.error(f"Webhook trade_event error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/webhook/signal_generated', methods=['POST'])
def webhook_signal_generated():
    """Receive signal generation events"""
    try:
        data = request.json
        if not data:
            logger.warning("No data received in signal_generated webhook")
            return jsonify({"status": "error", "message": "No data received"}), 400
        
        data_store.add_signal(data)
        logger.info(f"Signal webhook: {data.get('symbol')} {data.get('direction')}")
        return jsonify({"status": "success", "message": "Signal added"})
        
    except Exception as e:
        data_store.error_count += 1
        logger.error(f"Webhook signal_generated error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# ===== ENHANCED API ENDPOINTS =====
@app.route('/api/live_data')
def api_live_data():
    """API endpoint for live data"""
    try:
        # Add connection status and update info
        live_data = data_store.live_data.copy()
        live_data['connection_status'] = data_store.get_connection_status()
        live_data['data_store_status'] = data_store.get_status_summary()
        
        return jsonify(live_data)
    except Exception as e:
        logger.error(f"Error in live_data API: {e}")
        return jsonify({"error": "Failed to get live data"}), 500

@app.route('/api/account_history')
def api_account_history():
    """API endpoint for account history"""
    try:
        return jsonify(data_store.account_history)
    except Exception as e:
        logger.error(f"Error in account_history API: {e}")
        return jsonify({"error": "Failed to get account history"}), 500

@app.route('/api/trade_log')
def api_trade_log():
    """API endpoint for trade log"""
    try:
        # Convert deque to list for JSON serialization
        trades = list(data_store.trade_log)
        return jsonify(trades)
    except Exception as e:
        logger.error(f"Error in trade_log API: {e}")
        return jsonify({"error": "Failed to get trade log"}), 500

@app.route('/api/recent_signals')
def api_recent_signals():
    """API endpoint for recent signals"""
    try:
        # Convert deque to list for JSON serialization
        signals = list(data_store.recent_signals)
        return jsonify(signals)
    except Exception as e:
        logger.error(f"Error in recent_signals API: {e}")
        return jsonify({"error": "Failed to get signals"}), 500

@app.route('/api/connection_status')
def api_connection_status():
    """API endpoint for connection status"""
    try:
        return jsonify(data_store.get_connection_status())
    except Exception as e:
        logger.error(f"Error in connection_status API: {e}")
        return jsonify({"error": "Failed to get connection status"}), 500

@app.route('/api/mt5_status')
def api_mt5_status():
    """API endpoint for MT5 connection status"""
    try:
        # This would normally check actual MT5 connection
        # For now, return based on recent webhook activity
        is_connected = data_store.is_connected()
        return jsonify({
            'mt5_connected': is_connected,
            'last_update': data_store.last_update.isoformat(),
            'update_count': data_store.update_count
        })
    except Exception as e:
        logger.error(f"Error in mt5_status API: {e}")
        return jsonify({"error": "Failed to get MT5 status"}), 500

# ===== DEBUG AND STATUS ENDPOINTS =====
@app.route('/api/dashboard_status')
def api_dashboard_status():
    """Dashboard status endpoint"""
    try:
        status = {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': int(time.time() - app.config.get('START_TIME', time.time())),
            'data_store': data_store.get_status_summary(),
            'connection': data_store.get_connection_status(),
            'webhook_endpoints': [
                '/webhook/live_data',
                '/webhook/account_update', 
                '/webhook/trade_event',
                '/webhook/signal_generated'
            ],
            'api_endpoints': [
                '/api/live_data',
                '/api/account_history',
                '/api/trade_log',
                '/api/recent_signals'
            ]
        }
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting dashboard status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/debug/data_store')
def debug_data_store():
    """Debug endpoint to check data store contents"""
    try:
        debug_info = {
            'live_data': data_store.live_data,
            'account_history_count': len(data_store.account_history['timestamps']),
            'trade_log_count': len(data_store.trade_log),
            'signals_count': len(data_store.recent_signals),
            'last_update': data_store.last_update.isoformat(),
            'connection_status': data_store.get_connection_status(),
            'status_summary': data_store.get_status_summary()
        }
        return jsonify(debug_info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===== TEST ENDPOINTS FOR DEBUGGING =====
@app.route('/test/webhook', methods=['POST'])
def test_webhook():
    """Test webhook endpoint for debugging"""
    try:
        logger.info("Test webhook called")
        data = request.json or {"test": "data"}
        
        # Simulate a live data update
        test_data = {
            "timestamp": datetime.now().isoformat(),
            "robot_status": "Running",
            "account": {
                "balance": 10000.00,
                "equity": 10125.50,
                "profit": 125.50
            },
            "active_trades": 3,
            "test_mode": True
        }
        
        data_store.update_live_data(test_data)
        
        return jsonify({
            "status": "success", 
            "message": "Test webhook processed",
            "data_received": data,
            "update_count": data_store.update_count
        })
    except Exception as e:
        logger.error(f"Test webhook error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/test/simulate_trade')
def test_simulate_trade():
    """Simulate a trade for testing"""
    try:
        trade_data = {
            'timestamp': datetime.now().isoformat(),
            'symbol': 'EURUSD',
            'direction': 'long',
            'volume': 0.10,
            'entry_price': 1.0950,
            'tp': 1.1000,
            'profit': 45.50,
            'layer': 1,
            'is_martingale': False,
            'comment': 'TEST_TRADE'
        }
        
        data_store.add_trade_event(trade_data)
        
        return jsonify({
            "status": "success",
            "message": "Test trade added",
            "trade": trade_data
        })
    except Exception as e:
        logger.error(f"Test trade error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# ===== CHART GENERATION (Fixed) =====
def create_balance_chart():
    """Create balance/equity chart from webhook data"""
    try:
        import plotly.graph_objects as go
        import plotly.utils
        
        history = data_store.account_history
        
        if not history['timestamps']:
            return json.dumps({})
        
        fig = go.Figure()
        
        # Balance line
        fig.add_trace(go.Scatter(
            x=history['timestamps'],
            y=history['balance'],
            mode='lines',
            name='Balance',
            line=dict(color='#2E86AB', width=2)
        ))
        
        # Equity line
        fig.add_trace(go.Scatter(
            x=history['timestamps'],
            y=history['equity'],
            mode='lines',
            name='Equity',
            line=dict(color='#A23B72', width=2)
        ))
        
        fig.update_layout(
            title='Account Performance',
            xaxis_title='Time',
            yaxis_title='Amount ($)',
            template='plotly_dark',
            height=400,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
    except Exception as e:
        logger.error(f"Error creating balance chart: {e}")
        return json.dumps({})

def create_drawdown_chart():
    """Create drawdown chart from webhook data"""
    try:
        import plotly.graph_objects as go
        import plotly.utils
        
        history = data_store.account_history
        
        if not history['timestamps']:
            return json.dumps({})
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=history['timestamps'],
            y=history['drawdown'],
            mode='lines',
            name='Drawdown %',
            fill='tozeroy',
            fillcolor='rgba(255, 99, 132, 0.2)',
            line=dict(color='#FF6384', width=2)
        ))
        
        fig.update_layout(
            title='Drawdown',
            xaxis_title='Time',
            yaxis_title='Drawdown (%)',
            template='plotly_dark',
            height=300,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
    except Exception as e:
        logger.error(f"Error creating drawdown chart: {e}")
        return json.dumps({})

@app.route('/api/balance_chart')
def api_balance_chart():
    """API endpoint for balance chart"""
    try:
        chart_data = create_balance_chart()
        return chart_data
    except Exception as e:
        logger.error(f"Error in balance_chart API: {e}")
        return jsonify({"error": "Failed to create balance chart"}), 500

@app.route('/api/drawdown_chart')
def api_drawdown_chart():
    """API endpoint for drawdown chart"""
    try:
        chart_data = create_drawdown_chart()
        return chart_data
    except Exception as e:
        logger.error(f"Error in drawdown_chart API: {e}")
        return jsonify({"error": "Failed to create drawdown chart"}), 500

# ===== PAGE ROUTES =====
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/trades')
def trades_page():
    """Trades page"""
    return render_template('trades.html')

@app.route('/signals')
def signals_page():
    """Signals page"""
    return render_template('signals.html')

@app.route('/config')
def config_page():
    """Configuration page"""
    # Load config from file if it exists
    config = {}
    try:
        if os.path.exists("bot_config.json"):
            with open("bot_config.json", 'r') as f:
                config = json.load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
    
    return render_template('config.html', config=config)

@app.route('/trading_intelligence')
def trading_intelligence_dashboard():
    """Serve the trading intelligence dashboard"""
    return render_template('trading_intelligence_dashboard.html')

# ===== CONFIG API =====
@app.route('/api/config')
def api_config():
    """API endpoint for configuration"""
    try:
        if os.path.exists("bot_config.json"):
            with open("bot_config.json", 'r') as f:
                config = json.load(f)
            return jsonify(config)
        else:
            return jsonify({"error": "Config file not found"}), 404
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return jsonify({"error": "Failed to load config"}), 500

@app.route('/api/config', methods=['POST'])
def api_update_config():
    """API endpoint to update configuration"""
    try:
        new_config = request.json
        if not new_config:
            return jsonify({"status": "error", "message": "No config data received"})
        
        # Save config to file
        with open("bot_config.json", 'w') as f:
            json.dump(new_config, f, indent=2)
        
        # Create reload flag for bot
        os.makedirs("gui_data", exist_ok=True)
        with open("gui_data/reload_config.flag", 'w') as f:
            f.write(datetime.now().isoformat())
        
        logger.info("Configuration updated and reload flag created")
        return jsonify({"status": "success", "message": "Configuration updated"})
        
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        return jsonify({"status": "error", "message": str(e)})

# ===== TRADING INTELLIGENCE ENDPOINTS =====
@app.route('/trading_intelligence.json')
def trading_intelligence_json():
    """Serve the trading intelligence JSON file"""
    try:
        json_file_path = os.path.join('scrapers', 'trading_intelligence.json')
        
        # Check if file exists
        if not os.path.exists(json_file_path):
            return jsonify({"error": "Trading intelligence file not found"}), 404
        
        # Read and return the JSON file with UTF-8 encoding
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"✅ Successfully loaded JSON data with {len(data)} keys")
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"Error serving trading intelligence: {e}")
        return jsonify({"error": "Failed to load trading intelligence data", "details": str(e)}), 500

@app.route('/api/trading_intelligence')
def api_trading_intelligence():
    """API endpoint for trading intelligence data"""
    try:
        json_file_path = os.path.join('scrapers', 'trading_intelligence.json')
        
        if not os.path.exists(json_file_path):
            return jsonify({"error": "Trading intelligence file not found"}), 404
        
        # Read with UTF-8 encoding
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error in trading intelligence API: {e}")
        return jsonify({"error": "Failed to get trading intelligence data", "details": str(e)}), 500

# ===== BACKGROUND TASKS =====
def start_background_tasks():
    """Start background maintenance tasks"""
    def maintenance_task():
        """Periodic maintenance"""
        while True:
            try:
                # Log status every 5 minutes
                logger.info(f"📊 Dashboard Status: {data_store.update_count} updates, {data_store.error_count} errors")
                
                # Clean old data periodically
                now = datetime.now()
                
                # Clean account history older than 24 hours
                if data_store.account_history['timestamps']:
                    cutoff_time = now - timedelta(hours=24)
                    
                    # Find first index after cutoff
                    for i, timestamp_str in enumerate(data_store.account_history['timestamps']):
                        if datetime.fromisoformat(timestamp_str) >= cutoff_time:
                            break
                    else:
                        i = len(data_store.account_history['timestamps'])
                    
                    # Keep only recent data
                    if i > 0:
                        for key in ['timestamps', 'balance', 'equity', 'profit', 'drawdown']:
                            data_store.account_history[key] = data_store.account_history[key][i:]
                
                # Update connection status
                if not data_store.is_connected():
                    data_store.live_data['robot_status'] = 'Disconnected'
                
            except Exception as e:
                logger.error(f"Error in maintenance task: {e}")
            
            time.sleep(300)  # Run every 5 minutes
    
    # Start maintenance thread
    maintenance_thread = threading.Thread(target=maintenance_task, daemon=True)
    maintenance_thread.start()
    logger.info("Background maintenance tasks started")

# ===== APPLICATION STARTUP =====
if __name__ == '__main__':
    # Record start time
    app.config['START_TIME'] = time.time()
    
    # Start background tasks
    start_background_tasks()
    
    print("="*80)
    print("BM TRADING ROBOT - ENHANCED FLASK DASHBOARD")
    print("="*80)
    print("🌟 Enhanced Features:")
    print("  ✅ Real-time webhook data processing")
    print("  ✅ Live updates for all dashboard pages")
    print("  ✅ Enhanced connection status monitoring")
    print("  ✅ Improved error handling and logging")
    print("  ✅ Debug endpoints for troubleshooting")
    print("  ✅ Test endpoints for development")
    print("")
    print("📊 Dashboard URLs:")
    print("  Main Dashboard: http://localhost:5000")
    print("  Trades Page:    http://localhost:5000/trades")
    print("  Signals Page:   http://localhost:5000/signals")
    print("  Config Page:    http://localhost:5000/config")
    print("  Intelligence:   http://localhost:5000/trading_intelligence")
    print("")
    print("🔧 API Endpoints:")
    print("  Live Data:      http://localhost:5000/api/live_data")
    print("  Trades:         http://localhost:5000/api/trade_log")
    print("  Signals:        http://localhost:5000/api/recent_signals")
    print("  Status:         http://localhost:5000/api/dashboard_status")
    print("")
    print("📡 Webhook Endpoints (for bot):")
    print("  Live Data:      POST http://localhost:5000/webhook/live_data")
    print("  Account Update: POST http://localhost:5000/webhook/account_update")
    print("  Trade Event:    POST http://localhost:5000/webhook/trade_event")
    print("  Signal Event:   POST http://localhost:5000/webhook/signal_generated")
    print("")
    print("🧪 Test Endpoints:")
    print("  Test Webhook:   POST http://localhost:5000/test/webhook")
    print("  Simulate Trade: GET  http://localhost:5000/test/simulate_trade")
    print("")
    print("🔍 Debug Endpoints:")
    print("  Data Store:     http://localhost:5000/debug/data_store")
    print("  Connection:     http://localhost:5000/api/connection_status")
    print("="*80)
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)