# ===== WEBHOOK INTEGRATION FOR EXISTING BOT =====
# webhook_integration.py - Add this to your existing bot code

import requests
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Add to your existing bot code

class WebhookManager:
    """Manages webhook communications with the dashboard"""
    
    def __init__(self, dashboard_url="http://localhost:5000"):
        self.dashboard_url = dashboard_url
        self.enabled = True
        self.logger = logging.getLogger(__name__)
        
    def _send_webhook(self, endpoint: str, data: Dict[str, Any]) -> bool:
        """Send data to webhook endpoint"""
        if not self.enabled:
            return False
            
        try:
            url = f"{self.dashboard_url}/webhook/{endpoint}"
            response = requests.post(
                url, 
                json=data, 
                timeout=5,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                return True
            else:
                self.logger.warning(f"Webhook {endpoint} failed: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.debug(f"Webhook {endpoint} error: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected webhook error: {e}")
            return False
    
    def send_live_data(self, trade_manager, account_info=None):
        """Send current live data to dashboard"""
        try:
            if account_info is None:
                account_info = mt5.account_info()
                
            if account_info is None:
                return False
            
            # Calculate profit
            profit = account_info.equity - account_info.balance
            
            # Calculate drawdown
            drawdown = 0
            if trade_manager.initial_balance and account_info.equity < trade_manager.initial_balance:
                drawdown = ((trade_manager.initial_balance - account_info.equity) / trade_manager.initial_balance) * 100
            
            # Prepare batch data
            batches_data = []
            for batch_key, batch in trade_manager.martingale_batches.items():
                if batch.trades:
                    # Calculate next trigger price
                    next_trigger = None
                    try:
                        next_trigger = batch.get_next_trigger_price()
                    except:
                        pass
                    
                    batch_data = {
                        "batch_id": batch.batch_id,
                        "symbol": batch.symbol,
                        "direction": batch.direction,
                        "current_layer": batch.current_layer,
                        "total_volume": round(batch.total_volume, 2),
                        "breakeven_price": round(batch.breakeven_price, 5),
                        "initial_entry_price": round(batch.initial_entry_price, 5),
                        "next_trigger": round(next_trigger, 5) if next_trigger else None,
                        "created_time": batch.created_time.isoformat()
                    }
                    batches_data.append(batch_data)
            
            live_data = {
                "timestamp": datetime.now().isoformat(),
                "robot_status": "Running" if not trade_manager.emergency_stop_active else "Emergency Stop",
                "account": {
                    "balance": round(account_info.balance, 2),
                    "equity": round(account_info.equity, 2),
                    "margin": round(account_info.margin, 2),
                    "free_margin": round(account_info.margin_free, 2),
                    "margin_level": round((account_info.equity / account_info.margin * 100) if account_info.margin > 0 else 0, 2),
                    "profit": round(profit, 2)
                },
                "active_trades": len(trade_manager.active_trades),
                "active_batches": len([b for b in trade_manager.martingale_batches.values() if b.trades]),
                "total_trades": trade_manager.total_trades,
                "emergency_stop": trade_manager.emergency_stop_active,
                "drawdown_percent": round(drawdown, 2),
                "last_signal_time": datetime.now().isoformat(),
                "next_analysis": (datetime.now() + timedelta(minutes=5)).isoformat(),
                "batches": batches_data
            }
            
            return self._send_webhook("live_data", live_data)
            
        except Exception as e:
            self.logger.error(f"Error preparing live data: {e}")
            return False
    
    def send_account_update(self, account_info, trade_manager):
        """Send account update for chart data"""
        try:
            if account_info is None:
                return False
            
            profit = account_info.equity - account_info.balance
            drawdown = 0
            
            if trade_manager.initial_balance and account_info.equity < trade_manager.initial_balance:
                drawdown = ((trade_manager.initial_balance - account_info.equity) / trade_manager.initial_balance) * 100
            
            update_data = {
                "timestamp": datetime.now().isoformat(),
                "balance": round(account_info.balance, 2),
                "equity": round(account_info.equity, 2),
                "profit": round(profit, 2),
                "drawdown": round(drawdown, 2)
            }
            
            return self._send_webhook("account_update", update_data)
            
        except Exception as e:
            self.logger.error(f"Error sending account update: {e}")
            return False
    
    def send_trade_event(self, trade_info, event_type="executed"):
        """Send trade execution event"""
        try:
            trade_data = {
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                "symbol": trade_info.get('symbol'),
                "direction": trade_info.get('direction'),
                "volume": trade_info.get('volume'),
                "entry_price": trade_info.get('entry_price'),
                "tp": trade_info.get('tp'),
                "sl": trade_info.get('sl'),
                "order_id": trade_info.get('order_id'),
                "layer": trade_info.get('layer', 1),
                "is_martingale": trade_info.get('is_martingale', False),
                "profit": trade_info.get('profit', 0),
                "comment": trade_info.get('enhanced_comment', '')
            }
            
            return self._send_webhook("trade_event", trade_data)
            
        except Exception as e:
            self.logger.error(f"Error sending trade event: {e}")
            return False
    
    def send_signal_generated(self, signal):
        """Send signal generation event"""
        try:
            signal_data = {
                "timestamp": datetime.now().isoformat(),
                "symbol": signal.get('symbol'),
                "direction": signal.get('direction'),
                "entry_price": signal.get('entry_price'),
                "tp": signal.get('tp'),
                "sl_distance_pips": signal.get('sl_distance_pips'),
                "tp_distance_pips": signal.get('tp_distance_pips'),
                "risk_profile": signal.get('risk_profile'),
                "adx_value": signal.get('adx_value'),
                "rsi": signal.get('rsi'),
                "timeframes_aligned": signal.get('timeframes_aligned', 1),
                "is_initial": signal.get('is_initial', True)
            }
            
            return self._send_webhook("signal_generated", signal_data)
            
        except Exception as e:
            self.logger.error(f"Error sending signal: {e}")
            return False
    
    def check_config_reload(self) -> bool:
        """Check if configuration reload is requested"""
        try:
            reload_flag_file = "gui_data/reload_config.flag"
            if os.path.exists(reload_flag_file):
                # Read flag file to get timestamp
                with open(reload_flag_file, 'r') as f:
                    flag_time = f.read().strip()
                
                # Remove flag file
                os.remove(reload_flag_file)
                
                self.logger.info(f"Configuration reload requested at {flag_time}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error checking config reload: {e}")
            
        return False

# ===== INTEGRATION POINTS FOR EXISTING BOT =====

# Add this to your existing EnhancedTradeManager class:
def integrate_webhooks_to_trade_manager():
    """
    Add these methods to your existing EnhancedTradeManager class
    """
    
    # Add to __init__ method:
    def __init__(self):
        # ... existing code ...
        self.webhook_manager = WebhookManager()
        self.last_webhook_update = datetime.now()
        self.webhook_update_interval = 10  # seconds
    
    # Add to add_trade method (after successful trade addition):
    def add_trade_webhook_integration(self, trade_info):
        # ... existing add_trade code ...
        
        # Send trade event webhook
        try:
            self.webhook_manager.send_trade_event(trade_info, "executed")
        except Exception as e:
            logger.error(f"Webhook error in add_trade: {e}")
    
    # Add new method for periodic updates:
    def send_periodic_webhook_updates(self):
        """Send periodic updates to dashboard"""
        try:
            now = datetime.now()
            if (now - self.last_webhook_update).total_seconds() >= self.webhook_update_interval:
                
                # Get account info
                account_info = mt5.account_info()
                if account_info:
                    # Send live data
                    self.webhook_manager.send_live_data(self, account_info)
                    
                    # Send account update for charts (less frequently)
                    if (now - self.last_webhook_update).total_seconds() >= 30:
                        self.webhook_manager.send_account_update(account_info, self)
                
                self.last_webhook_update = now
                
        except Exception as e:
            logger.error(f"Error in periodic webhook updates: {e}")
    
    # Add configuration reload check:
    def check_and_reload_config(self):
        """Check for configuration reload request"""
        try:
            if self.webhook_manager.check_config_reload():
                logger.info("🔄 Configuration reload requested from dashboard")
                
                # Reload configuration
                global CONFIG
                CONFIG = load_config()
                
                # Update global variables
                self.update_global_config_variables()
                
                logger.info("✅ Configuration reloaded successfully")
                return True
                
        except Exception as e:
            logger.error(f"Error reloading configuration: {e}")
            
        return False
    
    def update_global_config_variables(self):
        """Update global configuration variables after reload"""
        global MARTINGALE_ENABLED, MAX_MARTINGALE_LAYERS, LOT_SIZE_MODE, MANUAL_LOT_SIZE
        global PAIRS, ENHANCED_PAIR_RISK_PROFILES, PARAM_SETS
        
        # Update martingale settings
        MARTINGALE_ENABLED = CONFIG['martingale_settings']['enabled']
        MAX_MARTINGALE_LAYERS = CONFIG['martingale_settings']['max_layers']
        
        # Update lot size settings
        LOT_SIZE_MODE = CONFIG['lot_size_settings']['mode']
        MANUAL_LOT_SIZE = CONFIG['lot_size_settings']['manual_lot_size']
        
        # Update pairs and profiles
        PAIRS = CONFIG['trading_pairs']
        ENHANCED_PAIR_RISK_PROFILES = CONFIG['pair_risk_profiles']
        PARAM_SETS = CONFIG['risk_parameters']
        
        logger.info("Global configuration variables updated")

# ===== INTEGRATION POINTS FOR MAIN ROBOT LOOP =====

def integrate_webhooks_to_main_loop():
    """
    Add these calls to your main robot loop in run_simplified_robot()
    """
    
    # Add this after each successful signal execution:
    def after_signal_execution(signal, execution_success):
        if execution_success:
            try:
                trade_manager.webhook_manager.send_signal_generated(signal)
            except Exception as e:
                logger.error(f"Webhook error after signal execution: {e}")
    
    # Add this in your main loop (every cycle):
    def main_loop_webhook_calls(trade_manager):
        try:
            # Send periodic updates
            trade_manager.send_periodic_webhook_updates()
            
            # Check for configuration reload
            trade_manager.check_and_reload_config()
            
        except Exception as e:
            logger.error(f"Webhook integration error in main loop: {e}")

# ===== SAMPLE INTEGRATION EXAMPLE =====

def sample_integration_in_execute_trade():
    """
    Example of how to integrate webhooks in your execute_trade function
    """
    
    # At the end of successful execute_trade function, add:
    def execute_trade_with_webhooks(signal, trade_manager):
        # ... existing execute_trade code ...
        
        # After successful execution (before return True):
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            # ... existing trade_info creation ...
            
            # Add trade to manager (existing code)
            trade_manager.add_trade(trade_info)
            
            # Send webhook notification
            try:
                trade_manager.webhook_manager.send_trade_event(trade_info, "executed")
            except Exception as e:
                logger.error(f"Webhook error: {e}")
            
            return True

# ===== SAMPLE INTEGRATION IN SIGNAL GENERATION =====

def sample_integration_in_signal_generation():
    """
    Example of how to integrate webhooks in signal generation
    """
    
    # In your generate_enhanced_signals function, after creating each signal:
    def generate_signals_with_webhooks(pairs, trade_manager):
        signals = []
        
        # ... existing signal generation code ...
        
        for signal in generated_signals:
            # Send signal webhook
            try:
                trade_manager.webhook_manager.send_signal_generated(signal)
            except Exception as e:
                logger.error(f"Signal webhook error: {e}")
            
            signals.append(signal)
        
        return signals

# ===== EASY INTEGRATION GUIDE =====

"""
INTEGRATION STEPS:

1. Add WebhookManager to your imports and trade manager initialization
2. Add these method calls to your main robot loop:

   # In run_simplified_robot(), add to main while loop:
   
   try:
       # ... existing cycle code ...
       
       # Add periodic webhook updates
       trade_manager.send_periodic_webhook_updates()
       
       # Check for config reload
       trade_manager.check_and_reload_config()
       
       # After executing signals:
       for signal in signals:
           if execute_trade(signal, trade_manager):
               # Send signal webhook
               trade_manager.webhook_manager.send_signal_generated(signal)
       
   except Exception as e:
       # ... existing error handling ...

3. Modify your execute_trade function to send trade events:

   # At the end of execute_trade, after trade_manager.add_trade():
   try:
       trade_manager.webhook_manager.send_trade_event(trade_info, "executed")
   except Exception as e:
       logger.error(f"Webhook error: {e}")

4. Make sure gui_data directory exists for config reload functionality

That's it! Your bot will now communicate with the dashboard.
"""