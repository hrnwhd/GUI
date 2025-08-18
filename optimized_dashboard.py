# ===== OPTIMIZED FLASK WEB DASHBOARD FOR BM TRADING ROBOT =====
# optimized_dashboard.py - High-performance real-time dashboard

from flask import Flask, render_template, jsonify, request, redirect, url_for
import json
import os
import sys
import time
import requests
import threading
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.utils
import pandas as pd
import logging
from collections import deque
from flask_cors import CORS


# Try to import MetaTrader5, but don't fail if not available
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("⚠️ MetaTrader5 not available - running in file-only mode")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Replace your routes section with this corrected version:

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

@app.route('/trading_intelligence')
def trading_intelligence_dashboard():
    """Serve the trading intelligence dashboard"""
    return render_template('trading_intelligence_dashboard.html')

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

@app.route('/debug/file_check')
def debug_file_check():
    """Debug endpoint to check file system"""
    try:
        current_dir = os.getcwd()
        
        debug_info = {
            "current_working_directory": current_dir,
            "scrapers_directory_exists": os.path.exists('scrapers'),
            "full_scrapers_path": os.path.join(current_dir, 'scrapers'),
            "files_in_current_directory": os.listdir('.') if os.path.exists('.') else [],
            "files_in_scrapers": os.listdir('scrapers') if os.path.exists('scrapers') else []
        }
        
        # Check specific file
        target_file = os.path.join('scrapers', 'trading_intelligence.json')
        debug_info["target_file_path"] = target_file
        debug_info["target_file_exists"] = os.path.exists(target_file)
        
        if os.path.exists(target_file):
            debug_info["file_size"] = os.path.getsize(target_file)
            debug_info["file_modified"] = os.path.getmtime(target_file)
        
        return jsonify(debug_info)
        
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/test_json')
def test_json():
    """Test endpoint to verify JSON loading"""
    try:
        json_file_path = os.path.join('scrapers', 'trading_intelligence.json')
        
        # Check if file exists
        if not os.path.exists(json_file_path):
            return f"❌ File not found: {json_file_path}"
        
        # Try to read the file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check file size
        file_size = len(content)
        
        # Try to parse JSON
        try:
            data = json.loads(content)
            json_keys = list(data.keys()) if isinstance(data, dict) else "Not a dict"
            
            return f"""
            ✅ SUCCESS!<br>
            File path: {json_file_path}<br>
            File size: {file_size} bytes<br>
            JSON parsed successfully<br>
            Top-level keys: {json_keys}<br>
            Data type: {type(data)}<br>
            <br>
            <a href="/trading_intelligence.json">Test JSON endpoint</a><br>
            <a href="/trading_intelligence">Test Dashboard</a>
            """
            
        except json.JSONDecodeError as e:
            return f"❌ JSON parse error: {e}<br>File size: {file_size} bytes"
            
    except Exception as e:
        return f"❌ Error: {e}"

class OptimizedDashboardManager:
    def __init__(self):
        self.gui_data_dir = "gui_data"
        self.config_file = "bot_config.json"
        self.ensure_directories()
        
        # Data files
        self.live_data_file = os.path.join(self.gui_data_dir, "live_data.json")
        self.account_history_file = os.path.join(self.gui_data_dir, "account_history.json")
        self.trade_log_file = os.path.join(self.gui_data_dir, "trade_log.json")
        self.signals_file = os.path.join(self.gui_data_dir, "recent_signals.json")
        self.reload_flag_file = os.path.join(self.gui_data_dir, "reload_config.flag")
        
        # Real-time data caching
        self.live_data_cache = {}
        self.account_cache = {}
        self.trades_cache = []
        self.signals_cache = []
        self.last_cache_update = datetime.now()
        self.cache_expiry = 2  # seconds
        
        # MT5 Integration for real-time data
        self.mt5_connected = False
        self.magic_number = None
        
        # Load config to get magic number
        self.load_magic_number()
        
        # Initialize MT5 connection if available
        if MT5_AVAILABLE:
            self.init_mt5_connection()
        
        self.initialize_data_files()
        
        # Start background data updater
        self.start_background_updater()
    
    def load_magic_number(self):
        """Load magic number from config"""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            self.magic_number = config.get('account_settings', {}).get('magic_number', 50515253)
            logger.info(f"Loaded magic number: {self.magic_number}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self.magic_number = 50515253  # Default
    
    def init_mt5_connection(self):
        """Initialize MT5 connection for real-time data"""
        if not MT5_AVAILABLE:
            return False
            
        try:
            if not mt5.initialize():
                logger.warning("MT5 initialization failed - using file-based data only")
                return False
            
            account_info = mt5.account_info()
            if account_info:
                logger.info(f"✅ MT5 connected: Account {account_info.login}")
                self.mt5_connected = True
                return True
            else:
                logger.warning("MT5 connection failed - no account info")
                return False
                
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False
    
    def start_background_updater(self):
        """Start background thread for continuous data updates"""
        def update_loop():
            while True:
                try:
                    self.update_live_cache()
                    time.sleep(1)  # Update every second
                except Exception as e:
                    logger.error(f"Background update error: {e}")
                    time.sleep(5)
        
        thread = threading.Thread(target=update_loop, daemon=True)
        thread.start()
        logger.info("✅ Background updater started")
    
    def update_live_cache(self):
        """Update live data cache with real-time information"""
        try:
            current_time = datetime.now()
            
            # Update from MT5 if connected
            if self.mt5_connected and MT5_AVAILABLE:
                account_info = mt5.account_info()
                if account_info:
                    # Get active positions
                    positions = mt5.positions_get()
                    our_positions = []
                    
                    if positions:
                        our_positions = [pos for pos in positions if pos.magic == self.magic_number]
                    
                    # Calculate profit
                    profit = account_info.equity - account_info.balance
                    
                    # Update live cache with real MT5 data
                    self.live_data_cache = {
                        "timestamp": current_time.isoformat(),
                        "robot_status": "Running",
                        "account": {
                            "balance": round(account_info.balance, 2),
                            "equity": round(account_info.equity, 2),
                            "margin": round(account_info.margin, 2),
                            "free_margin": round(account_info.margin_free, 2),
                            "margin_level": round((account_info.equity / account_info.margin * 100) if account_info.margin > 0 else 0, 2),
                            "profit": round(profit, 2)
                        },
                        "active_trades": len(our_positions),
                        "active_batches": self.count_active_batches(our_positions),
                        "total_trades": self.get_total_trades_count(),
                        "emergency_stop": False,
                        "drawdown_percent": self.calculate_drawdown(account_info),
                        "last_signal_time": current_time.isoformat(),
                        "next_analysis": (current_time + timedelta(minutes=5)).isoformat(),
                        "batches": self.get_batch_info(our_positions),
                        "pairs_status": self.get_pairs_status(),
                        "mt5_connected": True
                    }
                    
                    # Update account history for charts
                    self.update_account_history(account_info, profit)
                    
                    # Update active trades list
                    self.update_active_trades(our_positions)
                    
                    self.last_cache_update = current_time
                    return True
            
            # Fallback to file-based data
            file_data = self.load_json(self.live_data_file)
            if file_data:
                file_data["mt5_connected"] = False
                file_data["last_updated"] = current_time.isoformat()
                self.live_data_cache = file_data
                self.last_cache_update = current_time
                
        except Exception as e:
            logger.error(f"Cache update error: {e}")
            return False
    
    def count_active_batches(self, positions):
        """Count unique batch IDs from positions"""
        try:
            batch_ids = set()
            for pos in positions:
                comment = pos.comment
                if comment and comment.startswith('BM'):
                    # Extract batch ID from comment like BM01_EURUSD_SL01
                    parts = comment.split('_')
                    if len(parts) >= 1:
                        batch_id = parts[0]  # BM01
                        batch_ids.add(batch_id)
            
            return len(batch_ids)
        except Exception as e:
            logger.error(f"Error counting batches: {e}")
            return 0
    
    def get_total_trades_count(self):
        """Get total trades count from file or estimate"""
        try:
            state_file = "BM_bot_state.json"
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state = json.load(f)
                return state.get('total_trades', 0)
            
            # Estimate from trade log
            trade_data = self.load_json(self.trade_log_file)
            if trade_data and 'trades' in trade_data:
                return len(trade_data['trades'])
                
            return 0
        except Exception as e:
            logger.error(f"Error getting total trades: {e}")
            return 0
    
    def calculate_drawdown(self, account_info):
        """Calculate current drawdown percentage"""
        try:
            # Try to get initial balance from state file
            state_file = "BM_bot_state.json"
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state = json.load(f)
                initial_balance = state.get('initial_balance')
                
                if initial_balance and account_info.equity < initial_balance:
                    drawdown = ((initial_balance - account_info.equity) / initial_balance) * 100
                    return round(drawdown, 2)
            
            return 0
        except Exception as e:
            logger.error(f"Error calculating drawdown: {e}")
            return 0
    
    def get_batch_info(self, positions):
        """Get batch information from positions"""
        try:
            batches = {}
            
            # Group positions by batch
            for pos in positions:
                comment = pos.comment
                if comment and comment.startswith('BM'):
                    # Parse comment: BM01_EURUSD_SL01
                    parts = comment.split('_')
                    if len(parts) >= 3:
                        batch_id = parts[0][2:]  # Remove BM prefix
                        symbol = parts[1]
                        direction_layer = parts[2]
                        
                        # Determine direction
                        direction = 'long' if direction_layer.startswith('B') else 'short'
                        layer = int(direction_layer[1:]) if direction_layer[1:].isdigit() else 1
                        
                        batch_key = f"{symbol}_{direction}"
                        
                        if batch_key not in batches:
                            batches[batch_key] = {
                                'batch_id': int(batch_id) if batch_id.isdigit() else 0,
                                'symbol': symbol,
                                'direction': direction,
                                'current_layer': 0,
                                'total_volume': 0,
                                'breakeven_price': 0,
                                'positions': []
                            }
                        
                        batches[batch_key]['positions'].append({
                            'layer': layer,
                            'volume': pos.volume,
                            'entry_price': pos.price_open,
                            'current_price': pos.price_current,
                            'profit': pos.profit
                        })
                        
                        # Update batch totals
                        batch = batches[batch_key]
                        batch['current_layer'] = max(batch['current_layer'], layer)
                        batch['total_volume'] += pos.volume
            
            # Calculate breakeven prices
            for batch_key, batch in batches.items():
                if batch['total_volume'] > 0:
                    total_invested = sum(p['volume'] * p['entry_price'] for p in batch['positions'])
                    batch['breakeven_price'] = total_invested / batch['total_volume']
            
            return list(batches.values())
            
        except Exception as e:
            logger.error(f"Error getting batch info: {e}")
            return []
    
    def get_pairs_status(self):
        """Get trading pairs status"""
        try:
            config = self.load_json(self.config_file)
            pairs = config.get('trading_pairs', [])
            return {pair: "Active" for pair in pairs}
        except Exception as e:
            logger.error(f"Error getting pairs status: {e}")
            return {}
    
    def update_account_history(self, account_info, profit):
        """Update account history for charts"""
        try:
            history = self.load_json(self.account_history_file)
            if not history:
                history = {
                    "timestamps": [],
                    "balance": [],
                    "equity": [],
                    "profit": [],
                    "drawdown": []
                }
            
            current_time = datetime.now()
            
            # Only add new data point if enough time has passed (avoid spam)
            if (not history["timestamps"] or 
                (current_time - datetime.fromisoformat(history["timestamps"][-1])).total_seconds() > 60):
                
                history["timestamps"].append(current_time.isoformat())
                history["balance"].append(round(account_info.balance, 2))
                history["equity"].append(round(account_info.equity, 2))
                history["profit"].append(round(profit, 2))
                history["drawdown"].append(self.calculate_drawdown(account_info))
                
                # Keep only last 1000 points
                max_points = 1000
                for key in ['timestamps', 'balance', 'equity', 'profit', 'drawdown']:
                    if len(history[key]) > max_points:
                        history[key] = history[key][-max_points:]
                
                # Save updated history
                self.save_json(self.account_history_file, history)
                
        except Exception as e:
            logger.error(f"Error updating account history: {e}")
    
    def update_active_trades(self, positions):
        """Update active trades cache"""
        try:
            trades = []
            for pos in positions:
                trade = {
                    'timestamp': datetime.fromtimestamp(pos.time).isoformat(),
                    'symbol': pos.symbol,
                    'direction': 'long' if pos.type == 0 else 'short',
                    'volume': pos.volume,
                    'entry_price': pos.price_open,
                    'current_price': pos.price_current,
                    'tp': pos.tp,
                    'sl': pos.sl,
                    'profit': pos.profit,
                    'order_id': pos.ticket,
                    'comment': pos.comment
                }
                trades.append(trade)
            
            self.trades_cache = sorted(trades, key=lambda x: x['timestamp'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error updating active trades: {e}")
    
    def ensure_directories(self):
        """Create necessary directories"""
        if not os.path.exists(self.gui_data_dir):
            os.makedirs(self.gui_data_dir)
        if not os.path.exists("templates"):
            os.makedirs("templates")
        if not os.path.exists("static"):
            os.makedirs("static")
    
    def initialize_data_files(self):
        """Initialize data files with default structure if they don't exist"""
        
        # Live data
        if not os.path.exists(self.live_data_file):
            default_live = {
                "timestamp": datetime.now().isoformat(),
                "robot_status": "Connecting",
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
                "pairs_status": {},
                "batches": []
            }
            self.save_json(self.live_data_file, default_live)
        
        # Other files...
        for filename, default_data in [
            (self.account_history_file, {"timestamps": [], "balance": [], "equity": [], "profit": [], "drawdown": []}),
            (self.trade_log_file, {"trades": []}),
            (self.signals_file, {"signals": []})
        ]:
            if not os.path.exists(filename):
                self.save_json(filename, default_data)
    
    def load_json(self, file_path):
        """Safely load JSON file"""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
        return {}
    
    def save_json(self, file_path, data):
        """Safely save JSON file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"Error saving {file_path}: {e}")
            return False
    
    def get_live_data(self):
        """Get current live data (from cache)"""
        if not self.live_data_cache:
            return self.load_json(self.live_data_file)
        return self.live_data_cache
    
    def get_account_history(self):
        """Get account history for charts"""
        return self.load_json(self.account_history_file)
    
    def get_trade_log(self):
        """Get trade log - return both historical and active trades"""
        historical_data = self.load_json(self.trade_log_file)
        historical_trades = historical_data.get('trades', [])
        
        # Combine with active trades from cache
        all_trades = historical_trades + self.trades_cache
        
        # Remove duplicates and sort
        seen_ids = set()
        unique_trades = []
        
        for trade in all_trades:
            trade_id = trade.get('order_id', f"{trade.get('symbol')}_{trade.get('timestamp')}")
            if trade_id not in seen_ids:
                seen_ids.add(trade_id)
                unique_trades.append(trade)
        
        # Sort by timestamp (newest first) and return last 100
        sorted_trades = sorted(unique_trades, key=lambda x: x.get('timestamp', ''), reverse=True)
        return sorted_trades[:100]
    
    def get_recent_signals(self):
        """Get recent signals"""
        data = self.load_json(self.signals_file)
        signals = data.get('signals', [])
        return signals[-50:] if len(signals) > 50 else signals
    
    def get_config(self):
        """Get bot configuration"""
        return self.load_json(self.config_file)
    
    def update_config(self, new_config):
        """Update bot configuration"""
        if self.save_json(self.config_file, new_config):
            with open(self.reload_flag_file, 'w') as f:
                f.write(datetime.now().isoformat())
            return True
        return False
    
    def create_balance_chart(self):
        """Create balance/equity chart"""
        history = self.get_account_history()
        
        if not history.get('timestamps'):
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
    
    def create_drawdown_chart(self):
        """Create drawdown chart"""
        history = self.get_account_history()
        
        if not history.get('timestamps'):
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

# Initialize dashboard manager
dashboard_manager = OptimizedDashboardManager()

# ===== FLASK ROUTES =====

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/live_data')
def api_live_data():
    """API endpoint for live data - optimized"""
    try:
        live_data = dashboard_manager.get_live_data()
        return jsonify(live_data)
    except Exception as e:
        logger.error(f"Error in live_data API: {e}")
        return jsonify({"error": "Failed to get live data"}), 500

@app.route('/api/account_history')
def api_account_history():
    """API endpoint for account history"""
    try:
        return jsonify(dashboard_manager.get_account_history())
    except Exception as e:
        logger.error(f"Error in account_history API: {e}")
        return jsonify({"error": "Failed to get account history"}), 500

@app.route('/api/trade_log')
def api_trade_log():
    """API endpoint for trade log - optimized"""
    try:
        return jsonify(dashboard_manager.get_trade_log())
    except Exception as e:
        logger.error(f"Error in trade_log API: {e}")
        return jsonify({"error": "Failed to get trade log"}), 500

@app.route('/api/recent_signals')
def api_recent_signals():
    """API endpoint for recent signals"""
    try:
        return jsonify(dashboard_manager.get_recent_signals())
    except Exception as e:
        logger.error(f"Error in recent_signals API: {e}")
        return jsonify({"error": "Failed to get signals"}), 500

@app.route('/api/balance_chart')
def api_balance_chart():
    """API endpoint for balance chart"""
    try:
        chart_data = dashboard_manager.create_balance_chart()
        return chart_data
    except Exception as e:
        logger.error(f"Error in balance_chart API: {e}")
        return jsonify({"error": "Failed to create balance chart"}), 500

@app.route('/api/drawdown_chart')
def api_drawdown_chart():
    """API endpoint for drawdown chart"""
    try:
        chart_data = dashboard_manager.create_drawdown_chart()
        return chart_data
    except Exception as e:
        logger.error(f"Error in drawdown_chart API: {e}")
        return jsonify({"error": "Failed to create drawdown chart"}), 500

@app.route('/config')
def config_page():
    """Configuration page"""
    config = dashboard_manager.get_config()
    return render_template('config.html', config=config)

@app.route('/api/config')
def api_config():
    """API endpoint for configuration"""
    return jsonify(dashboard_manager.get_config())

@app.route('/api/config', methods=['POST'])
def api_update_config():
    """API endpoint to update configuration"""
    try:
        new_config = request.json
        if dashboard_manager.update_config(new_config):
            return jsonify({"status": "success", "message": "Configuration updated"})
        else:
            return jsonify({"status": "error", "message": "Failed to update configuration"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/trades')
def trades_page():
    """Trades page"""
    return render_template('trades.html')

@app.route('/signals')
def signals_page():
    """Signals page"""
    return render_template('signals.html')

# Webhook endpoints (for compatibility)
@app.route('/webhook/live_data', methods=['POST'])
def webhook_live_data():
    try:
        data = request.json
        dashboard_manager.save_json(dashboard_manager.live_data_file, data)
        return jsonify({"status": "success"})
    except Exception as e:
        logger.error(f"Webhook live_data error: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/webhook/trade_event', methods=['POST'])
def webhook_trade_event():
    try:
        data = request.json
        trade_log = dashboard_manager.load_json(dashboard_manager.trade_log_file)
        
        if 'trades' not in trade_log:
            trade_log['trades'] = []
        
        trade_log['trades'].append(data)
        
        if len(trade_log['trades']) > 500:
            trade_log['trades'] = trade_log['trades'][-500:]
        
        dashboard_manager.save_json(dashboard_manager.trade_log_file, trade_log)
        return jsonify({"status": "success"})
    except Exception as e:
        logger.error(f"Webhook trade_event error: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/webhook/signal_generated', methods=['POST'])
def webhook_signal_generated():
    try:
        data = request.json
        signals_log = dashboard_manager.load_json(dashboard_manager.signals_file)
        
        if 'signals' not in signals_log:
            signals_log['signals'] = []
        
        signals_log['signals'].append(data)
        
        if len(signals_log['signals']) > 200:
            signals_log['signals'] = signals_log['signals'][-200:]
        
        dashboard_manager.save_json(dashboard_manager.signals_file, signals_log)
        return jsonify({"status": "success"})
    except Exception as e:
        logger.error(f"Webhook signal_generated error: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/mt5_status')
def api_mt5_status():
    """Check MT5 connection status"""
    return jsonify({
        "mt5_connected": dashboard_manager.mt5_connected,
        "mt5_available": MT5_AVAILABLE,
        "last_cache_update": dashboard_manager.last_cache_update.isoformat(),
        "cache_age_seconds": (datetime.now() - dashboard_manager.last_cache_update).total_seconds()
    })

if __name__ == '__main__':
    print("="*60)
    print("BM TRADING ROBOT - OPTIMIZED WEB DASHBOARD")
    print("="*60)
    print("Dashboard URL: http://localhost:5000")
    print("Config URL: http://localhost:5000/config")
    print("Trades URL: http://localhost:5000/trades")
    print("Signals URL: http://localhost:5000/signals")
    print(f"MT5 Available: {MT5_AVAILABLE}")
    print(f"MT5 Connected: {dashboard_manager.mt5_connected}")
    print("="*60)
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
