# ===== FLASK WEB DASHBOARD FOR BM TRADING ROBOT =====
# dashboard.py - Standalone web interface

from flask import Flask, render_template, jsonify, request, redirect, url_for
import json
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.utils
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class DashboardManager:
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
        
        self.initialize_data_files()
    
    def ensure_directories(self):
        """Create necessary directories"""
        if not os.path.exists(self.gui_data_dir):
            os.makedirs(self.gui_data_dir)
        if not os.path.exists("templates"):
            os.makedirs("templates")
        if not os.path.exists("static"):
            os.makedirs("static")
    
    def initialize_data_files(self):
        """Initialize data files with default structure"""
        
        # Live data
        if not os.path.exists(self.live_data_file):
            default_live = {
                "timestamp": datetime.now().isoformat(),
                "robot_status": "Stopped",
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
        
        # Account history
        if not os.path.exists(self.account_history_file):
            default_history = {
                "timestamps": [],
                "balance": [],
                "equity": [],
                "profit": [],
                "drawdown": []
            }
            self.save_json(self.account_history_file, default_history)
        
        # Trade log
        if not os.path.exists(self.trade_log_file):
            default_log = {
                "trades": []
            }
            self.save_json(self.trade_log_file, default_log)
        
        # Recent signals
        if not os.path.exists(self.signals_file):
            default_signals = {
                "signals": []
            }
            self.save_json(self.signals_file, default_signals)
    
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
        """Get current live data"""
        return self.load_json(self.live_data_file)
    
    def get_account_history(self):
        """Get account history for charts"""
        return self.load_json(self.account_history_file)
    
    def get_trade_log(self):
        """Get trade log"""
        data = self.load_json(self.trade_log_file)
        # Return last 100 trades
        trades = data.get('trades', [])
        return trades[-100:] if len(trades) > 100 else trades
    
    def get_recent_signals(self):
        """Get recent signals"""
        data = self.load_json(self.signals_file)
        signals = data.get('signals', [])
        # Return last 50 signals
        return signals[-50:] if len(signals) > 50 else signals
    
    def get_config(self):
        """Get bot configuration"""
        return self.load_json(self.config_file)
    
    def update_config(self, new_config):
        """Update bot configuration"""
        if self.save_json(self.config_file, new_config):
            # Create reload flag
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
        
        # Profit/Loss area
        fig.add_trace(go.Scatter(
            x=history['timestamps'],
            y=history['profit'],
            mode='lines',
            name='P&L',
            fill='tozeroy',
            fillcolor='rgba(46, 134, 171, 0.2)',
            line=dict(color='#F18F01', width=1)
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
dashboard_manager = DashboardManager()

# ===== FLASK ROUTES =====

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/live_data')
def api_live_data():
    """API endpoint for live data"""
    return jsonify(dashboard_manager.get_live_data())

@app.route('/api/account_history')
def api_account_history():
    """API endpoint for account history"""
    return jsonify(dashboard_manager.get_account_history())

@app.route('/api/trade_log')
def api_trade_log():
    """API endpoint for trade log"""
    return jsonify(dashboard_manager.get_trade_log())

@app.route('/api/recent_signals')
def api_recent_signals():
    """API endpoint for recent signals"""
    return jsonify(dashboard_manager.get_recent_signals())

@app.route('/api/balance_chart')
def api_balance_chart():
    """API endpoint for balance chart"""
    chart_data = dashboard_manager.create_balance_chart()
    return chart_data

@app.route('/api/drawdown_chart')
def api_drawdown_chart():
    """API endpoint for drawdown chart"""
    chart_data = dashboard_manager.create_drawdown_chart()
    return chart_data

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

# ===== WEBHOOK ENDPOINTS FOR BOT =====

@app.route('/webhook/live_data', methods=['POST'])
def webhook_live_data():
    """Webhook to receive live data from bot"""
    try:
        data = request.json
        if dashboard_manager.save_json(dashboard_manager.live_data_file, data):
            return jsonify({"status": "success"})
        else:
            return jsonify({"status": "error"})
    except Exception as e:
        logger.error(f"Webhook live_data error: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/webhook/account_update', methods=['POST'])
def webhook_account_update():
    """Webhook to receive account updates from bot"""
    try:
        data = request.json
        history = dashboard_manager.get_account_history()
        
        # Add new data point
        history['timestamps'].append(data['timestamp'])
        history['balance'].append(data['balance'])
        history['equity'].append(data['equity'])
        history['profit'].append(data['profit'])
        history['drawdown'].append(data['drawdown'])
        
        # Keep only last 1000 points
        max_points = 1000
        for key in ['timestamps', 'balance', 'equity', 'profit', 'drawdown']:
            if len(history[key]) > max_points:
                history[key] = history[key][-max_points:]
        
        if dashboard_manager.save_json(dashboard_manager.account_history_file, history):
            return jsonify({"status": "success"})
        else:
            return jsonify({"status": "error"})
    except Exception as e:
        logger.error(f"Webhook account_update error: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/webhook/trade_event', methods=['POST'])
def webhook_trade_event():
    """Webhook to receive trade events from bot"""
    try:
        data = request.json
        trade_log = dashboard_manager.load_json(dashboard_manager.trade_log_file)
        
        if 'trades' not in trade_log:
            trade_log['trades'] = []
        
        trade_log['trades'].append(data)
        
        # Keep only last 500 trades
        if len(trade_log['trades']) > 500:
            trade_log['trades'] = trade_log['trades'][-500:]
        
        if dashboard_manager.save_json(dashboard_manager.trade_log_file, trade_log):
            return jsonify({"status": "success"})
        else:
            return jsonify({"status": "error"})
    except Exception as e:
        logger.error(f"Webhook trade_event error: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/webhook/signal_generated', methods=['POST'])
def webhook_signal_generated():
    """Webhook to receive signal events from bot"""
    try:
        data = request.json
        signals_log = dashboard_manager.load_json(dashboard_manager.signals_file)
        
        if 'signals' not in signals_log:
            signals_log['signals'] = []
        
        signals_log['signals'].append(data)
        
        # Keep only last 200 signals
        if len(signals_log['signals']) > 200:
            signals_log['signals'] = signals_log['signals'][-200:]
        
        if dashboard_manager.save_json(dashboard_manager.signals_file, signals_log):
            return jsonify({"status": "success"})
        else:
            return jsonify({"status": "error"})
    except Exception as e:
        logger.error(f"Webhook signal_generated error: {e}")
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    print("="*60)
    print("BM TRADING ROBOT - WEB DASHBOARD")
    print("="*60)
    print("Dashboard URL: http://localhost:5000")
    print("Config URL: http://localhost:5000/config")
    print("Trades URL: http://localhost:5000/trades")
    print("Signals URL: http://localhost:5000/signals")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)