"""
Enhanced Webhook Integration for Trading Bot
Handles sending trading data to external webhooks with retry logic and error handling
"""

import requests
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import time

# Set up logging
logger = logging.getLogger(__name__)

class WebhookManager:
    """Manages webhook communications with the dashboard"""
    
    def __init__(self, dashboard_url="http://localhost:5000"):
        self.dashboard_url = dashboard_url
        self.enabled = True
        self.logger = logging.getLogger(__name__)
        
    def _send_webhook(self, endpoint: str, data: dict) -> bool:
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
                self.logger.debug(f"✅ Webhook {endpoint} sent successfully")
                return True
            else:
                self.logger.warning(f"⚠️ Webhook {endpoint} failed: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.debug(f"🔌 Webhook {endpoint} connection error: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Unexpected webhook error: {e}")
            return False
    
    def send_live_data(self, trade_manager, account_info=None):
        """Send current live data to dashboard"""
        try:
            if account_info is None:
                import MetaTrader5 as mt5
                account_info = mt5.account_info()
                
            if account_info is None:
                return False
            
            # Calculate profit
            profit = account_info.equity - account_info.balance
            
            # Calculate drawdown
            drawdown = 0
            if hasattr(trade_manager, 'initial_balance') and trade_manager.initial_balance and account_info.equity < trade_manager.initial_balance:
                drawdown = ((trade_manager.initial_balance - account_info.equity) / trade_manager.initial_balance) * 100
            
            # Prepare batch data
            batches_data = []
            if hasattr(trade_manager, 'martingale_batches'):
                for batch_key, batch in trade_manager.martingale_batches.items():
                    if hasattr(batch, 'trades') and batch.trades:
                        # Calculate next trigger price
                        next_trigger = None
                        try:
                            if hasattr(batch, 'get_next_trigger_price'):
                                next_trigger = batch.get_next_trigger_price()
                        except:
                            pass
                        
                        batch_data = {
                            "batch_id": getattr(batch, 'batch_id', 0),
                            "symbol": getattr(batch, 'symbol', 'Unknown'),
                            "direction": getattr(batch, 'direction', 'Unknown'),
                            "current_layer": getattr(batch, 'current_layer', 0),
                            "total_volume": round(getattr(batch, 'total_volume', 0), 2),
                            "breakeven_price": round(getattr(batch, 'breakeven_price', 0), 5),
                            "initial_entry_price": round(getattr(batch, 'initial_entry_price', 0), 5),
                            "next_trigger": round(next_trigger, 5) if next_trigger else None,
                            "created_time": getattr(batch, 'created_time', datetime.now()).isoformat()
                        }
                        batches_data.append(batch_data)
            
            live_data = {
                "timestamp": datetime.now().isoformat(),
                "robot_status": "Running" if not getattr(trade_manager, 'emergency_stop_active', False) else "Emergency Stop",
                "account": {
                    "balance": round(account_info.balance, 2),
                    "equity": round(account_info.equity, 2),
                    "margin": round(account_info.margin, 2),
                    "free_margin": round(account_info.margin_free, 2),
                    "margin_level": round((account_info.equity / account_info.margin * 100) if account_info.margin > 0 else 0, 2),
                    "profit": round(profit, 2)
                },
                "active_trades": len(getattr(trade_manager, 'active_trades', [])),
                "active_batches": len([b for b in getattr(trade_manager, 'martingale_batches', {}).values() if hasattr(b, 'trades') and b.trades]),
                "total_trades": getattr(trade_manager, 'total_trades', 0),
                "emergency_stop": getattr(trade_manager, 'emergency_stop_active', False),
                "drawdown_percent": round(drawdown, 2),
                "last_signal_time": datetime.now().isoformat(),
                "next_analysis": (datetime.now() + timedelta(minutes=5)).isoformat(),
                "batches": batches_data,
                "pairs_status": {}
            }
            
            success = self._send_webhook("live_data", live_data)
            if success:
                self.logger.debug(f"📊 Live data sent: Balance=${account_info.balance:.2f}")
            return success
            
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
            
            if hasattr(trade_manager, 'initial_balance') and trade_manager.initial_balance and account_info.equity < trade_manager.initial_balance:
                drawdown = ((trade_manager.initial_balance - account_info.equity) / trade_manager.initial_balance) * 100
            
            update_data = {
                "timestamp": datetime.now().isoformat(),
                "balance": round(account_info.balance, 2),
                "equity": round(account_info.equity, 2),
                "profit": round(profit, 2),
                "drawdown": round(drawdown, 2)
            }
            
            success = self._send_webhook("account_update", update_data)
            if success:
                self.logger.debug(f"📈 Chart data sent")
            return success
            
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
                "comment": trade_info.get('enhanced_comment', ''),
                "sl_distance_pips": trade_info.get('sl_distance_pips', 0)
            }
            
            success = self._send_webhook("trade_event", trade_data)
            if success:
                self.logger.info(f"🎯 Trade event sent: {trade_info.get('symbol')} {trade_info.get('direction')}")
            return success
            
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
            
            success = self._send_webhook("signal_generated", signal_data)
            if success:
                self.logger.info(f"📡 Signal sent: {signal.get('symbol')} {signal.get('direction')}")
            return success
            
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


class WebhookIntegration:
    def __init__(self, config_file: str = 'bot_config.json'):
        """Initialize webhook integration with configuration"""
        self.config = self.load_config(config_file)
        # Set defaults for webhook configuration
        webhook_config = self.config.get('webhook', {})
        self.webhook_urls = webhook_config.get('webhook_urls', [])
        self.webhook_enabled = webhook_config.get('webhook_enabled', False)
        self.retry_attempts = webhook_config.get('webhook_retry_attempts', 3)
        self.timeout = webhook_config.get('webhook_timeout', 10)
        self.rate_limit_delay = webhook_config.get('webhook_rate_limit_delay', 1)
        
        logger.info(f"Webhook integration initialized:")
        logger.info(f"  - Enabled: {self.webhook_enabled}")
        logger.info(f"  - URLs configured: {len(self.webhook_urls)}")
        logger.info(f"  - Retry attempts: {self.retry_attempts}")
    
    def load_config(self, config_file: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_file} not found, using defaults")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing config file: {e}")
            return {}
    
    def send_webhook(self, data: Dict[str, Any]) -> bool:
        """Send data to all configured webhook URLs"""
        if not self.webhook_enabled:
            logger.debug("Webhooks disabled, skipping send")
            return True
        
        if not self.webhook_urls:
            logger.warning("No webhook URLs configured")
            return False
        
        success_count = 0
        total_urls = len(self.webhook_urls)
        
        for url in self.webhook_urls:
            if self.send_single_webhook(url, data):
                success_count += 1
            
            # Rate limiting between webhook calls
            if len(self.webhook_urls) > 1:
                time.sleep(self.rate_limit_delay)
        
        success_rate = success_count / total_urls if total_urls > 0 else 0
        logger.info(f"Webhook batch sent: {success_count}/{total_urls} successful ({success_rate:.1%})")
        
        return success_count > 0
    
    def send_single_webhook(self, url: str, data: Dict[str, Any]) -> bool:
        """Send data to a single webhook URL with retry logic"""
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'BM-Trading-Robot-Webhook/1.0'
        }
        
        # Add timestamp if not present
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().isoformat()
        
        # Add webhook metadata
        webhook_data = {
            'webhook_version': '1.0',
            'sent_at': datetime.now().isoformat(),
            'data': data
        }
        
        for attempt in range(self.retry_attempts):
            try:
                response = requests.post(
                    url,
                    json=webhook_data,
                    headers=headers,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    logger.debug(f"✅ Webhook sent successfully to {self.mask_url(url)}")
                    return True
                else:
                    logger.warning(f"⚠️ Webhook failed with status {response.status_code}: {response.text}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"⏰ Webhook timeout (attempt {attempt + 1}/{self.retry_attempts})")
            except requests.exceptions.ConnectionError:
                logger.warning(f"🔌 Webhook connection error (attempt {attempt + 1}/{self.retry_attempts})")
            except Exception as e:
                logger.error(f"❌ Webhook error (attempt {attempt + 1}/{self.retry_attempts}): {e}")
            
            # Wait before retry (exponential backoff)
            if attempt < self.retry_attempts - 1:
                wait_time = (2 ** attempt) * self.rate_limit_delay
                time.sleep(wait_time)
        
        logger.error(f"❌ Webhook failed after {self.retry_attempts} attempts to {self.mask_url(url)}")
        return False
    
    def mask_url(self, url: str) -> str:
        """Mask URL for logging security"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return f"{parsed.scheme}://{parsed.netloc}/***"
        except:
            return "***masked***"
    
    def send_trade_signal(self, signal_data: Dict[str, Any]) -> bool:
        """Send trading signal webhook"""
        webhook_data = {
            'type': 'trade_signal',
            'signal': signal_data,
            'timestamp': datetime.now().isoformat()
        }
        return self.send_webhook(webhook_data)
    
    def send_trade_execution(self, execution_data: Dict[str, Any]) -> bool:
        """Send trade execution webhook"""
        webhook_data = {
            'type': 'trade_execution',
            'execution': execution_data,
            'timestamp': datetime.now().isoformat()
        }
        return self.send_webhook(webhook_data)
    
    def send_batch_update(self, batch_data: Dict[str, Any]) -> bool:
        """Send batch update webhook"""
        webhook_data = {
            'type': 'batch_update',
            'batch': batch_data,
            'timestamp': datetime.now().isoformat()
        }
        return self.send_webhook(webhook_data)
    
    def send_account_update(self, account_data: Dict[str, Any]) -> bool:
        """Send account update webhook"""
        webhook_data = {
            'type': 'account_update',
            'account': account_data,
            'timestamp': datetime.now().isoformat()
        }
        return self.send_webhook(webhook_data)
    
    def send_error_alert(self, error_data: Dict[str, Any]) -> bool:
        """Send error alert webhook"""
        webhook_data = {
            'type': 'error_alert',
            'error': error_data,
            'timestamp': datetime.now().isoformat(),
            'severity': error_data.get('severity', 'medium')
        }
        return self.send_webhook(webhook_data)
    
    def send_heartbeat(self) -> bool:
        """Send heartbeat webhook to confirm system is running"""
        webhook_data = {
            'type': 'heartbeat',
            'timestamp': datetime.now().isoformat(),
            'status': 'running'
        }
        return self.send_webhook(webhook_data)
    
    def test_webhook_connection(self) -> Dict[str, Any]:
        """Test webhook connections and return status"""
        if not self.webhook_enabled:
            return {
                'enabled': False,
                'message': 'Webhooks are disabled'
            }
        
        if not self.webhook_urls:
            return {
                'enabled': True,
                'urls_configured': 0,
                'message': 'No webhook URLs configured'
            }
        
        test_data = {
            'type': 'connection_test',
            'message': 'Testing webhook connection',
            'timestamp': datetime.now().isoformat()
        }
        
        results = []
        for i, url in enumerate(self.webhook_urls):
            success = self.send_single_webhook(url, test_data)
            results.append({
                'url_index': i,
                'url_masked': self.mask_url(url),
                'success': success
            })
        
        successful_tests = sum(1 for r in results if r['success'])
        
        return {
            'enabled': True,
            'urls_configured': len(self.webhook_urls),
            'successful_tests': successful_tests,
            'total_tests': len(self.webhook_urls),
            'success_rate': successful_tests / len(self.webhook_urls) if self.webhook_urls else 0,
            'results': results
        }


# Convenience function for quick webhook sending
def send_quick_webhook(data: Dict[str, Any], config_file: str = 'bot_config.json') -> bool:
    """Quick function to send webhook data"""
    webhook = WebhookIntegration(config_file)
    return webhook.send_webhook(data)


# Example usage and testing
if __name__ == "__main__":
    # Test the webhook integration
    logging.basicConfig(level=logging.INFO)
    
    webhook = WebhookIntegration()
    
    # Test connection
    test_result = webhook.test_webhook_connection()
    print(f"Webhook test result: {json.dumps(test_result, indent=2)}")
    
    # Send test data
    test_data = {
        'message': 'Test webhook from BM Trading Robot',
        'account': 'TEST123',
        'balance': 10000.00,
        'timestamp': datetime.now().isoformat()
    }
    
    success = webhook.send_webhook(test_data)
    print(f"Test webhook sent: {'Success' if success else 'Failed'}")