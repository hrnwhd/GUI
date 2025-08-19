# ===== ENHANCED FLASK DASHBOARD WITH JSON FILE BACKUP =====
# This version uses webhooks for real-time data but also writes to JSON files as backup

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

# ===== ENHANCED DATA STORAGE WITH JSON BACKUP =====
class HybridDataStore:
    """Enhanced storage that uses both webhooks and JSON files for reliability"""
    
    def __init__(self):
        # Create data directory
        self.data_dir = "dashboard_data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # JSON file paths
        self.live_data_file = os.path.join(self.data_dir, "live_data.json")
        self.account_history_file = os.path.join(self.data_dir, "account_history.json")
        self.trades_file = os.path.join(self.data_dir, "trades.json")
        self.signals_file = os.path.join(self.data_dir, "signals.json")
        
        # In-memory data (faster access)
        self.live_data = {
            "timestamp": datetime.now().isoformat(),
            "robot_status": "Disconnected",
            "account": {
                "balance": 0, "equity": 0, "margin": 0,
                "free_margin": 0, "margin_level": 0, "profit": 0
            },
            "active_trades": 0, "active_batches": 0,
            "total_trades": 0, "emergency_stop": False,
            "drawdown_percent": 0, "batches": [],
            "pairs_status": {}, "mt5_connected": False
        }
        
        # Limited in-memory storage
        self.account_history = {"timestamps": [], "balance": [], "equity": [], "profit": [], "drawdown": []}
        self.trade_log = deque(maxlen=500)
        self.recent_signals = deque(maxlen=100)
        
        # Connection tracking
        self.last_update = datetime.now()
        self.update_count = 0
        self.error_count = 0
        
        # Load existing data
        self.load_all_data()
        
        # Start background JSON writer
        self.start_json_writer()
        
    def load_all_data(self):
        """Load all data from JSON files on startup"""
        try:
            # Load live data
            if os.path.exists(self.live_data_file):
                with open(self.live_data_file, 'r') as f:
                    self.live_data.update(json.load(f))
                logger.info("✅ Loaded live data from JSON")
            
            # Load account history
            if os.path.exists(self.account_history_file):
                with open(self.account_history_file, 'r') as f:
                    history_data = json.load(f)
                    self.account_history.update(history_data)
                logger.info(f"✅ Loaded {len(self.account_history['timestamps'])} account history points")
            
            # Load trades
            if os.path.exists(self.trades_file):
                with open(self.trades_file, 'r') as f:
                    trades_data = json.load(f)
                    self.trade_log.extend(trades_data)
                logger.info(f"✅ Loaded {len(self.trade_log)} trades")
            
            # Load signals
            if os.path.exists(self.signals_file):
                with open(self.signals_file, 'r') as f:
                    signals_data = json.load(f)
                    self.recent_signals.extend(signals_data)
                logger.info(f"✅ Loaded {len(self.recent_signals)} signals")
                
        except Exception as e:
            logger.error(f"Error loading data from JSON files: {e}")
    
    def start_json_writer(self):
        """Start background thread to write data to JSON files"""
        def json_writer():
            while True:
                try:
                    time.sleep(30)  # Write every 30 seconds
                    self.write_all_data_to_json()
                except Exception as e:
                    logger.error(f"Error in JSON writer: {e}")
        
        writer_thread = threading.Thread(target=json_writer, daemon=True)
        writer_thread.start()
        logger.info("📁 JSON writer started")
    
    def write_all_data_to_json(self):
        """Write all data to JSON files"""
        try:
            # Write live data
            with open(self.live_data_file, 'w') as f:
                json.dump(self.live_data, f, indent=2, default=str)
            
            # Write account history (keep last 1000 points)
            history_to_save = {}
            for key in self.account_history:
                history_to_save[key] = list(self.account_history[key])[-1000:]
            
            with open(self.account_history_file, 'w') as f:
                json.dump(history_to_save, f, indent=2)
            
            # Write trades (keep last 500)
            trades_to_save = list(self.trade_log)[-500:]
            with open(self.trades_file, 'w') as f:
                json.dump(trades_to_save, f, indent=2, default=str)
            
            # Write signals (keep last 100)
            signals_to_save = list(self.recent_signals)[-100:]
            with open(self.signals_file, 'w') as f:
                json.dump(signals_to_save, f, indent=2, default=str)
            
            logger.debug("💾 Data written to JSON files")
            
        except Exception as e:
            logger.error(f"Error writing to JSON files: {e}")
    
    # ===== WEBHOOK METHODS (Same as before but with JSON backup) =====
    def update_live_data(self, data):
        """Update live data from webhook and save to JSON"""
        try:
            self.live_data.update(data)
            self.live_data["timestamp"] = datetime.now().isoformat()
            self.last_update = datetime.now()
            self.update_count += 1
            
            # Immediate JSON write for critical data
            threading.Thread(target=self.write_live_data_immediate, daemon=True).start()
            
            logger.debug(f"Live data updated via webhook (#{self.update_count})")
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error updating live data: {e}")
    
    def write_live_data_immediate(self):
        """Immediate write of live data to JSON"""
        try:
            with open(self.live_data_file, 'w') as f:
                json.dump(self.live_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error in immediate JSON write: {e}")
    
    def add_account_update(self, data):
        """Add account update for chart data"""
        try:
            now = datetime.now()
            
            self.account_history["timestamps"].append(now.isoformat())
            self.account_history["balance"].append(data.get("balance", 0))
            self.account_history["equity"].append(data.get("equity", 0))
            self.account_history["profit"].append(data.get("profit", 0))
            self.account_history["drawdown"].append(data.get("drawdown", 0))
            
            # Keep only last 1000 points in memory
            for key in ['timestamps', 'balance', 'equity', 'profit', 'drawdown']:
                if len(self.account_history[key]) > 1000:
                    self.account_history[key] = self.account_history[key][-1000:]
            
            logger.debug("Account history updated")
                
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
            
            self.trade_log.appendleft(trade_data)
            logger.debug(f"Added trade event: {data.get('symbol')} {data.get('direction')}")
            
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
            
            self.recent_signals.appendleft(signal_data)
            logger.debug(f"Added signal: {data.get('symbol')} {data.get('direction')}")
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error adding signal: {e}")
    
    def is_connected(self):
        """Check if bot is connected"""
        time_since_update = (datetime.now() - self.last_update).total_seconds()
        return time_since_update < 30
    
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

# Initialize data store
data_store = HybridDataStore()

# ===== WEBHOOKS (Same as before) =====
@app.route('/webhook/live_data', methods=['POST'])
def webhook_live_data():
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data received"}), 400
        
        data_store.update_live_data(data)
        return jsonify({"status": "success", "message": "Live data updated"})
        
    except Exception as e:
        logger.error(f"Webhook live_data error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/webhook/account_update', methods=['POST'])
def webhook_account_update():
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data received"}), 400
        
        data_store.add_account_update(data)
        return jsonify({"status": "success", "message": "Account updated"})
        
    except Exception as e:
        logger.error(f"Webhook account_update error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/webhook/trade_event', methods=['POST'])
def webhook_trade_event():
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data received"}), 400
        
        data_store.add_trade_event(data)
        return jsonify({"status": "success", "message": "Trade event added"})
        
    except Exception as e:
        logger.error(f"Webhook trade_event error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/webhook/signal_generated', methods=['POST'])
def webhook_signal_generated():
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data received"}), 400
        
        data_store.add_signal(data)
        return jsonify({"status": "success", "message": "Signal added"})
        
    except Exception as e:
        logger.error(f"Webhook signal_generated error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# ===== FAST API ENDPOINTS FOR FRONTEND =====
@app.route('/api/live_data')
def api_live_data():
    try:
        live_data = data_store.live_data.copy()
        live_data['connection_status'] = data_store.get_connection_status()
        return jsonify(live_data)
    except Exception as e:
        logger.error(f"Error in live_data API: {e}")
        return jsonify({"error": "Failed to get live data"}), 500

@app.route('/api/account_history')
def api_account_history():
    try:
        return jsonify(data_store.account_history)
    except Exception as e:
        logger.error(f"Error in account_history API: {e}")
        return jsonify({"error": "Failed to get account history"}), 500

@app.route('/api/trade_log')
def api_trade_log():
    try:
        trades = list(data_store.trade_log)
        return jsonify(trades)
    except Exception as e:
        logger.error(f"Error in trade_log API: {e}")
        return jsonify({"error": "Failed to get trade log"}), 500

@app.route('/api/recent_signals')
def api_recent_signals():
    try:
        signals = list(data_store.recent_signals)
        return jsonify(signals)
    except Exception as e:
        logger.error(f"Error in recent_signals API: {e}")
        return jsonify({"error": "Failed to get signals"}), 500

# ===== SIMPLIFIED CHART ENDPOINTS =====
@app.route('/api/balance_chart')
def api_balance_chart():
    try:
        history = data_store.account_history
        if not history['timestamps']:
            return jsonify({})
        
        # Return simple data for frontend to render
        chart_data = {
            'timestamps': history['timestamps'][-100:],  # Last 100 points
            'balance': history['balance'][-100:],
            'equity': history['equity'][-100:]
        }
        return jsonify(chart_data)
    except Exception as e:
        logger.error(f"Error in balance_chart API: {e}")
        return jsonify({"error": "Failed to create balance chart"}), 500

@app.route('/api/drawdown_chart')
def api_drawdown_chart():
    try:
        history = data_store.account_history
        if not history['timestamps']:
            return jsonify({})
        
        chart_data = {
            'timestamps': history['timestamps'][-100:],
            'drawdown': history['drawdown'][-100:]
        }
        return jsonify(chart_data)
    except Exception as e:
        logger.error(f"Error in drawdown_chart API: {e}")
        return jsonify({"error": "Failed to create drawdown chart"}), 500

# ===== JSON FILE API ENDPOINTS (BACKUP ACCESS) =====
@app.route('/api/json/live_data')
def api_json_live_data():
    """Direct access to JSON file data"""
    try:
        if os.path.exists(data_store.live_data_file):
            with open(data_store.live_data_file, 'r') as f:
                return jsonify(json.load(f))
        return jsonify({"error": "No JSON data available"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/json/trades')
def api_json_trades():
    """Direct access to trades JSON file"""
    try:
        if os.path.exists(data_store.trades_file):
            with open(data_store.trades_file, 'r') as f:
                return jsonify(json.load(f))
        return jsonify([])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===== PAGE ROUTES =====
@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/trades')
def trades_page():
    return render_template('trades.html')

@app.route('/signals')
def signals_page():
    return render_template('signals.html')

@app.route('/config')
def config_page():
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
    return render_template('trading_intelligence_dashboard.html')

# ===== TRADING INTELLIGENCE (Same as before) =====
@app.route('/trading_intelligence.json')
def trading_intelligence_json():
    try:
        json_file_path = os.path.join('scrapers', 'trading_intelligence.json')
        
        if not os.path.exists(json_file_path):
            return jsonify({"error": "Trading intelligence file not found"}), 404
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"Error serving trading intelligence: {e}")
        return jsonify({"error": "Failed to load trading intelligence data"}), 500

# ===== BACKGROUND MAINTENANCE =====
def start_background_tasks():
    def maintenance_task():
        while True:
            try:
                # Clean old data
                now = datetime.now()
                cutoff_time = now - timedelta(hours=24)
                
                # Clean account history older than 24 hours
                if data_store.account_history['timestamps']:
                    new_timestamps = []
                    new_balance = []
                    new_equity = []
                    new_profit = []
                    new_drawdown = []
                    
                    for i, timestamp_str in enumerate(data_store.account_history['timestamps']):
                        try:
                            timestamp = datetime.fromisoformat(timestamp_str)
                            if timestamp >= cutoff_time:
                                new_timestamps.append(timestamp_str)
                                new_balance.append(data_store.account_history['balance'][i])
                                new_equity.append(data_store.account_history['equity'][i])
                                new_profit.append(data_store.account_history['profit'][i])
                                new_drawdown.append(data_store.account_history['drawdown'][i])
                        except:
                            continue
                    
                    data_store.account_history['timestamps'] = new_timestamps
                    data_store.account_history['balance'] = new_balance
                    data_store.account_history['equity'] = new_equity
                    data_store.account_history['profit'] = new_profit
                    data_store.account_history['drawdown'] = new_drawdown
                
                # Force JSON write
                data_store.write_all_data_to_json()
                
                logger.info(f"🧹 Maintenance: {data_store.update_count} updates, {data_store.error_count} errors")
                
            except Exception as e:
                logger.error(f"Error in maintenance: {e}")
            
            time.sleep(300)  # Every 5 minutes
    
    maintenance_thread = threading.Thread(target=maintenance_task, daemon=True)
    maintenance_thread.start()
    logger.info("🧹 Background maintenance started")

if __name__ == '__main__':
    # Record start time
    app.config['START_TIME'] = time.time()
    
    # Start background tasks
    start_background_tasks()
    
    print("="*80)
    print("BM TRADING ROBOT - HYBRID DASHBOARD (WEBHOOKS + JSON)")
    print("="*80)
    print("🔄 Features:")
    print("  ✅ Real-time webhook data processing")
    print("  ✅ JSON file backup for reliability")
    print("  ✅ Automatic data persistence")
    print("  ✅ Fast frontend access")
    print("  ✅ Data recovery on restart")
    print("")
    print("📊 Dashboard URLs:")
    print("  Main Dashboard: http://localhost:5000")
    print("  Trades Page:    http://localhost:5000/trades")
    print("  Signals Page:   http://localhost:5000/signals")
    print("  Config Page:    http://localhost:5000/config")
    print("  Intelligence:   http://localhost:5000/trading_intelligence")
    print("")
    print("📁 Data Storage:")
    print("  JSON Files:     ./dashboard_data/")
    print("  Live Data:      ./dashboard_data/live_data.json")
    print("  Trades:         ./dashboard_data/trades.json")
    print("  Signals:        ./dashboard_data/signals.json")
    print("="*80)
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)