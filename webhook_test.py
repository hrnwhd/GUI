# ===== WEBHOOK CONNECTION TEST SCRIPT =====
# Run this to test if your dashboard is receiving webhooks properly

import requests
import json
import time
from datetime import datetime

class WebhookTester:
    """Test webhook connections to dashboard"""
    
    def __init__(self, dashboard_url="http://localhost:5000"):
        self.dashboard_url = dashboard_url
        
    def test_connection(self):
        """Test basic connection to dashboard"""
        try:
            print("🔍 Testing dashboard connection...")
            response = requests.get(f"{self.dashboard_url}/api/dashboard_status", timeout=5)
            
            if response.status_code == 200:
                print("✅ Dashboard is running and responding")
                data = response.json()
                print(f"   Uptime: {data.get('uptime_seconds', 0)} seconds")
                return True
            else:
                print(f"❌ Dashboard responded with status {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to dashboard - make sure it's running on port 5000")
            return False
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
    
    def test_webhook_endpoints(self):
        """Test all webhook endpoints"""
        endpoints = [
            ('live_data', self._get_test_live_data()),
            ('account_update', self._get_test_account_update()),
            ('trade_event', self._get_test_trade_event()),
            ('signal_generated', self._get_test_signal())
        ]
        
        print("\n🔧 Testing webhook endpoints...")
        results = {}
        
        for endpoint, test_data in endpoints:
            print(f"\n   Testing /webhook/{endpoint}...")
            
            try:
                url = f"{self.dashboard_url}/webhook/{endpoint}"
                response = requests.post(
                    url,
                    json=test_data,
                    headers={'Content-Type': 'application/json'},
                    timeout=10
                )
                
                if response.status_code == 200:
                    print(f"   ✅ {endpoint}: SUCCESS")
                    results[endpoint] = True
                else:
                    print(f"   ❌ {endpoint}: FAILED (HTTP {response.status_code})")
                    print(f"      Response: {response.text[:100]}")
                    results[endpoint] = False
                    
            except Exception as e:
                print(f"   ❌ {endpoint}: ERROR - {e}")
                results[endpoint] = False
        
        return results
    
    def test_api_endpoints(self):
        """Test API endpoints to see if they return webhook data"""
        endpoints = [
            'live_data',
            'account_history', 
            'trade_log',
            'recent_signals'
        ]
        
        print("\n📊 Testing API endpoints...")
        
        for endpoint in endpoints:
            try:
                url = f"{self.dashboard_url}/api/{endpoint}"
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"   ✅ /api/{endpoint}: {len(str(data))} characters")
                    
                    # Check if data looks recent
                    if isinstance(data, dict) and 'timestamp' in data:
                        timestamp = data['timestamp']
                        print(f"      Last update: {timestamp}")
                    elif isinstance(data, list) and len(data) > 0:
                        print(f"      Records: {len(data)}")
                    
                else:
                    print(f"   ❌ /api/{endpoint}: HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ /api/{endpoint}: ERROR - {e}")
    
    def send_test_data_sequence(self):
        """Send a sequence of test data to simulate bot activity"""
        print("\n🎯 Sending test data sequence...")
        
        # 1. Send live data
        print("   1. Sending live data...")
        self._send_webhook('live_data', self._get_test_live_data())
        time.sleep(1)
        
        # 2. Send account update
        print("   2. Sending account update...")
        self._send_webhook('account_update', self._get_test_account_update())
        time.sleep(1)
        
        # 3. Send signal
        print("   3. Sending signal...")
        self._send_webhook('signal_generated', self._get_test_signal())
        time.sleep(1)
        
        # 4. Send trade event
        print("   4. Sending trade event...")
        self._send_webhook('trade_event', self._get_test_trade_event())
        time.sleep(1)
        
        print("   ✅ Test sequence completed")
        
        # 5. Check if data appears in API
        print("\n   Checking if data appears in API endpoints...")
        time.sleep(2)  # Give dashboard time to process
        
        try:
            # Check live data
            response = requests.get(f"{self.dashboard_url}/api/live_data")
            if response.status_code == 200:
                data = response.json()
                if data.get('robot_status') == 'Test Mode':
                    print("   ✅ Live data updated successfully")
                else:
                    print("   ⚠️ Live data may not be updated")
            
            # Check trade log
            response = requests.get(f"{self.dashboard_url}/api/trade_log")
            if response.status_code == 200:
                trades = response.json()
                if trades and trades[0].get('symbol') == 'EURUSD':
                    print("   ✅ Trade log updated successfully")
                else:
                    print("   ⚠️ Trade log may not be updated")
                    
        except Exception as e:
            print(f"   ❌ Error checking API updates: {e}")
    
    def _send_webhook(self, endpoint, data):
        """Send webhook data"""
        try:
            url = f"{self.dashboard_url}/webhook/{endpoint}"
            response = requests.post(url, json=data, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _get_test_live_data(self):
        """Generate test live data"""
        return {
            "timestamp": datetime.now().isoformat(),
            "robot_status": "Test Mode",
            "account": {
                "balance": 10000.00,
                "equity": 10150.00,
                "margin": 500.00,
                "free_margin": 9650.00,
                "margin_level": 2030.00,
                "profit": 150.00
            },
            "active_trades": 2,
            "active_batches": 1,
            "total_trades": 25,
            "emergency_stop": False,
            "drawdown_percent": 0.0,
            "last_signal_time": datetime.now().isoformat(),
            "batches": [{
                "batch_id": 1,
                "symbol": "EURUSD",
                "direction": "long",
                "current_layer": 2,
                "total_volume": 0.03,
                "breakeven_price": 1.08500
            }],
            "mt5_connected": True
        }
    
    def _get_test_account_update(self):
        """Generate test account update"""
        return {
            "timestamp": datetime.now().isoformat(),
            "balance": 10000.00,
            "equity": 10150.00,
            "profit": 150.00,
            "drawdown": 0.0
        }
    
    def _get_test_trade_event(self):
        """Generate test trade event"""
        return {
            "timestamp": datetime.now().isoformat(),
            "event_type": "executed",
            "symbol": "EURUSD",
            "direction": "long",
            "volume": 0.01,
            "entry_price": 1.08500,
            "tp": 1.08700,
            "sl": None,
            "order_id": 123456,
            "layer": 1,
            "is_martingale": False,
            "profit": 0.0,
            "comment": "BM01_EURUSD_B01",
            "sl_distance_pips": 20.0
        }
    
    def _get_test_signal(self):
        """Generate test signal"""
        return {
            "timestamp": datetime.now().isoformat(),
            "symbol": "GBPUSD",
            "direction": "short",
            "entry_price": 1.25500,
            "tp": 1.25200,
            "sl_distance_pips": 25.0,
            "tp_distance_pips": 30.0,
            "risk_profile": "Medium",
            "adx_value": 35.5,
            "rsi": 72.0,
            "timeframes_aligned": 2,
            "is_initial": True
        }

def run_full_test():
    """Run complete webhook test suite"""
    print("="*60)
    print("BM TRADING ROBOT - WEBHOOK CONNECTION TEST")
    print("="*60)
    
    tester = WebhookTester()
    
    # Test 1: Basic connection
    if not tester.test_connection():
        print("\n❌ CRITICAL: Dashboard not accessible")
        print("   Make sure the dashboard is running: python optimized_dashboard.py")
        return False
    
    # Test 2: Webhook endpoints
    webhook_results = tester.test_webhook_endpoints()
    webhook_success = all(webhook_results.values())
    
    if webhook_success:
        print("\n✅ All webhook endpoints working")
    else:
        print("\n⚠️ Some webhook endpoints failed")
        for endpoint, success in webhook_results.items():
            status = "✅" if success else "❌"
            print(f"   {status} {endpoint}")
    
    # Test 3: API endpoints
    tester.test_api_endpoints()
    
    # Test 4: Full data flow test
    if webhook_success:
        tester.send_test_data_sequence()
    
    # Test 5: Final status
    print("\n📊 FINAL RESULTS:")
    print("="*40)
    
    if webhook_success:
        print("✅ Webhook system is working correctly")
        print("✅ Your bot should be able to send data to dashboard")
        print("\nNEXT STEPS:")
        print("1. Make sure your bot is using the FIXED WebhookManager")
        print("2. Start your bot and check for webhook success messages")
        print("3. Monitor dashboard for real-time updates")
    else:
        print("❌ Webhook system has issues")
        print("\nTROUBLESHOOTING:")
        print("1. Make sure you're using the FIXED dashboard version")
        print("2. Check if port 5000 is available")
        print("3. Look for error messages in dashboard logs")
    
    return webhook_success

if __name__ == "__main__":
    run_full_test()