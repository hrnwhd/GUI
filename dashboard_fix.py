#!/usr/bin/env python3
"""
Quick Dashboard Fix Script
Run this to immediately fix your dashboard data issues
"""

import json
import os
import requests
import time
import threading
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_realistic_test_data():
    """Create realistic test data for dashboard"""
    print("📊 Creating realistic test data...")
    
    # Ensure directory exists
    os.makedirs("gui_data", exist_ok=True)
    
    # Current time
    now = datetime.now()
    
    # 1. Live Data
    live_data = {
        "timestamp": now.isoformat(),
        "robot_status": "Running",
        "account": {
            "balance": 10856.73,
            "equity": 10946.27,
            "margin": 250.00,
            "free_margin": 10696.27,
            "margin_level": 4378.51,
            "profit": 89.54
        },
        "active_trades": 3,
        "active_batches": 2,
        "total_trades": 12,
        "emergency_stop": False,
        "drawdown_percent": 0.84,
        "last_signal_time": (now - timedelta(minutes=8)).isoformat(),
        "next_analysis": (now + timedelta(minutes=2)).isoformat(),
        "batches": [
            {
                "batch_id": 1,
                "symbol": "EURUSD",
                "direction": "long",
                "current_layer": 3,
                "total_volume": 0.18,
                "breakeven_price": 1.08275,
                "initial_entry_price": 1.08450,
                "next_trigger": 1.08050,
                "created_time": (now - timedelta(hours=1, minutes=25)).isoformat()
            },
            {
                "batch_id": 2,
                "symbol": "GBPUSD",
                "direction": "short",
                "current_layer": 2,
                "total_volume": 0.12,
                "breakeven_price": 1.26685,
                "initial_entry_price": 1.26750,
                "next_trigger": 1.26850,
                "created_time": (now - timedelta(minutes=45)).isoformat()
            }
        ],
        "pairs_status": {
            "EURUSD": "Active",
            "GBPUSD": "Active",
            "AUDUSD": "Waiting",
            "USDCAD": "Active",
            "XAUUSD": "Paused",
            "BTCUSD": "Active"
        }
    }
    
    # Save live data
    with open("gui_data/live_data.json", "w") as f:
        json.dump(live_data, f, indent=2)
    
    # 2. Account History (for charts)
    timestamps = []
    balances = []
    equities = []
    profits = []
    drawdowns = []
    
    base_balance = 10800.00
    
    # Generate 100 data points over last 8 hours
    for i in range(100):
        time_point = now - timedelta(minutes=i*5)
        timestamps.append(time_point.isoformat())
        
        # Simulate realistic account progression
        balance = base_balance + (i * 0.8) + (i % 10) * 2
        equity_variance = 15 + (i % 20) * 3 - (i % 7) * 5
        equity = balance + equity_variance
        
        balances.append(round(balance, 2))
        equities.append(round(equity, 2))
        profits.append(round(equity - balance, 2))
        
        # Calculate drawdown from peak
        peak_equity = 10950
        dd = max(0, (peak_equity - equity) / peak_equity * 100)
        drawdowns.append(round(dd, 2))
    
    history = {
        "timestamps": list(reversed(timestamps)),
        "balance": list(reversed(balances)),
        "equity": list(reversed(equities)),
        "profit": list(reversed(profits)),
        "drawdown": list(reversed(drawdowns))
    }
    
    with open("gui_data/account_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    # 3. Trade Log
    trades = [
        {
            "timestamp": (now - timedelta(hours=1, minutes=25)).isoformat(),
            "symbol": "EURUSD",
            "direction": "long",
            "volume": 0.06,
            "entry_price": 1.08450,
            "tp": 1.08600,
            "profit": 0,  # Still open
            "layer": 1,
            "comment": "BM01_EURUSD_B01",
            "enhanced_comment": "BM01_EURUSD_B01"
        },
        {
            "timestamp": (now - timedelta(hours=1, minutes=15)).isoformat(),
            "symbol": "EURUSD",
            "direction": "long",
            "volume": 0.06,
            "entry_price": 1.08250,
            "tp": 1.08400,
            "profit": 0,  # Still open
            "layer": 2,
            "comment": "BM01_EURUSD_B02",
            "enhanced_comment": "BM01_EURUSD_B02"
        },
        {
            "timestamp": (now - timedelta(minutes=55)).isoformat(),
            "symbol": "EURUSD",
            "direction": "long",
            "volume": 0.06,
            "entry_price": 1.08125,
            "tp": 1.08275,
            "profit": 0,  # Still open
            "layer": 3,
            "comment": "BM01_EURUSD_B03",
            "enhanced_comment": "BM01_EURUSD_B03"
        },
        {
            "timestamp": (now - timedelta(minutes=45)).isoformat(),
            "symbol": "GBPUSD",
            "direction": "short",
            "volume": 0.08,
            "entry_price": 1.26750,
            "tp": 1.26600,
            "profit": 0,  # Still open
            "layer": 1,
            "comment": "BM02_GBPUSD_S01",
            "enhanced_comment": "BM02_GBPUSD_S01"
        },
        {
            "timestamp": (now - timedelta(minutes=35)).isoformat(),
            "symbol": "GBPUSD",
            "direction": "short",
            "volume": 0.04,
            "entry_price": 1.26800,
            "tp": 1.26650,
            "profit": 0,  # Still open
            "layer": 2,
            "comment": "BM02_GBPUSD_S02",
            "enhanced_comment": "BM02_GBPUSD_S02"
        },
        {
            "timestamp": (now - timedelta(hours=2)).isoformat(),
            "symbol": "AUDUSD",
            "direction": "long",
            "volume": 0.10,
            "entry_price": 0.66250,
            "tp": 0.66400,
            "profit": 75.50,  # Closed profit
            "layer": 1,
            "comment": "BM03_AUDUSD_B01",
            "enhanced_comment": "BM03_AUDUSD_B01"
        }
    ]
    
    with open("gui_data/trade_log.json", "w") as f:
        json.dump({"trades": trades}, f, indent=2)
    
    # 4. Recent Signals
    signals = [
        {
            "timestamp": (now - timedelta(minutes=8)).isoformat(),
            "symbol": "USDCAD",
            "direction": "short",
            "entry_price": 1.42850,
            "tp": 1.42700,
            "sl_distance_pips": 25,
            "tp_distance_pips": 15,
            "risk_profile": "Low",
            "adx_value": 32.8,
            "rsi": 72.5,
            "timeframes_aligned": 3,
            "is_initial": True
        },
        {
            "timestamp": (now - timedelta(minutes=25)).isoformat(),
            "symbol": "XAUUSD",
            "direction": "long",
            "entry_price": 2045.75,
            "tp": 2048.25,
            "sl_distance_pips": 20,
            "tp_distance_pips": 25,
            "risk_profile": "High",
            "adx_value": 28.2,
            "rsi": 28.9,
            "timeframes_aligned": 2,
            "is_initial": True
        }
    ]
    
    with open("gui_data/recent_signals.json", "w") as f:
        json.dump({"signals": signals}, f, indent=2)
    
    print("✅ Realistic test data created successfully")
    return True

def send_webhook_data():
    """Send data directly to dashboard webhooks"""
    print("📡 Sending webhook data to dashboard...")
    
    dashboard_url = "http://localhost:5000"
    
    try:
        # Load the data we just created
        with open("gui_data/live_data.json", "r") as f:
            live_data = json.load(f)
        
        with open("gui_data/account_history.json", "r") as f:
            history = json.load(f)
        
        # Send live data
        response = requests.post(
            f"{dashboard_url}/webhook/live_data",
            json=live_data,
            timeout=5
        )
        if response.status_code == 200:
            print("✅ Live data sent successfully")
        else:
            print(f"❌ Live data failed: {response.status_code}")
        
        # Send account update (for charts)
        if history['timestamps']:
            latest_idx = -1  # Get the latest data point
            account_update = {
                "timestamp": history['timestamps'][latest_idx],
                "balance": history['balance'][latest_idx],
                "equity": history['equity'][latest_idx],
                "profit": history['profit'][latest_idx],
                "drawdown": history['drawdown'][latest_idx]
            }
            
            response = requests.post(
                f"{dashboard_url}/webhook/account_update",
                json=account_update,
                timeout=5
            )
            if response.status_code == 200:
                print("✅ Account update sent successfully")
            else:
                print(f"❌ Account update failed: {response.status_code}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to dashboard. Is it running on http://localhost:5000?")
        return False
    except Exception as e:
        print(f"❌ Error sending webhook data: {e}")
        return False

def start_live_data_sender():
    """Start background thread to send live updates"""
    def update_worker():
        print("🔄 Starting live data updates...")
        while True:
            try:
                # Update live data with current timestamp
                now = datetime.now()
                
                # Load current live data
                try:
                    with open("gui_data/live_data.json", "r") as f:
                        live_data = json.load(f)
                except:
                    return  # Exit if file doesn't exist
                
                # Update timestamp and some dynamic values
                live_data["timestamp"] = now.isoformat()
                live_data["last_signal_time"] = (now - timedelta(minutes=2, seconds=30)).isoformat()
                live_data["next_analysis"] = (now + timedelta(minutes=2, seconds=45)).isoformat()
                
                # Simulate slight balance changes
                base_balance = 10856.73
                base_equity = 10946.27
                
                # Add small random variations
                import random
                balance_var = random.uniform(-2, 3)
                equity_var = random.uniform(-5, 8)
                
                live_data["account"]["balance"] = round(base_balance + balance_var, 2)
                live_data["account"]["equity"] = round(base_equity + equity_var, 2)
                live_data["account"]["profit"] = round(live_data["account"]["equity"] - live_data["account"]["balance"], 2)
                
                # Update margin level
                if live_data["account"]["margin"] > 0:
                    live_data["account"]["margin_level"] = round(
                        (live_data["account"]["equity"] / live_data["account"]["margin"]) * 100, 2
                    )
                
                # Save updated data
                with open("gui_data/live_data.json", "w") as f:
                    json.dump(live_data, f, indent=2)
                
                # Send webhook
                try:
                    response = requests.post(
                        "http://localhost:5000/webhook/live_data",
                        json=live_data,
                        timeout=2
                    )
                    if response.status_code == 200:
                        print(f"✅ Live update sent: Balance=${live_data['account']['balance']:.2f}")
                except:
                    pass  # Silent fail for webhook issues
                
                time.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                print(f"❌ Error in live updates: {e}")
                time.sleep(30)  # Wait longer on error
    
    # Start background thread
    update_thread = threading.Thread(target=update_worker, daemon=True)
    update_thread.start()
    print("✅ Live data sender started")

def main():
    """Main function to fix dashboard"""
    print("🚀 BM Trading Robot Dashboard Quick Fix")
    print("=" * 50)
    
    # Step 1: Create test data
    if not create_realistic_test_data():
        print("❌ Failed to create test data")
        return
    
    # Step 2: Try to send webhook data
    if send_webhook_data():
        print("✅ Initial webhook data sent")
    else:
        print("⚠️ Webhook send failed - make sure dashboard is running")
    
    # Step 3: Start live updates
    start_live_data_sender()
    
    print("\n" + "=" * 50)
    print("DASHBOARD FIX COMPLETE")
    print("=" * 50)
    print("✅ Test data created")
    print("✅ Live data sender started")
    print("\n📋 Next steps:")
    print("1. Open http://localhost:5000 in your browser")
    print("2. You should now see live data updating")
    print("3. The dashboard will refresh every 10 seconds")
    print("4. Press Ctrl+C to stop live updates")
    print("=" * 50)
    
    try:
        print("\n⏰ Sending live updates... (Press Ctrl+C to stop)")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Live updates stopped")

if __name__ == "__main__":
    main()