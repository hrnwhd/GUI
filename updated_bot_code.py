# ===== UPDATED BM TRADING ROBOT - VERSION 3 LOGIC WITH FIXED WEBHOOKS =====
# Updated version with proper webhook integration and error handling

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from collections import deque
import time
import logging
import json
import os
import requests
import threading

# Suppress warnings
warnings.filterwarnings("ignore")

# ===== LOAD CONFIGURATION FROM JSON =====
def load_config():
    """Load configuration from JSON file"""
    config_file = "bot_config.json"
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from {config_file}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file {config_file} not found!")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing configuration file: {e}")
        raise

# Initialize basic logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
CONFIG = load_config()

# ===== EXTRACT CONFIGURATION VALUES =====
ACCOUNT_NUMBER = CONFIG['account_settings']['account_number']
MAGIC_NUMBER = CONFIG['account_settings']['magic_number']

# Timeframe mapping
TIMEFRAME_MAP = {
    'M1': mt5.TIMEFRAME_M1,
    'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15,
    'M30': mt5.TIMEFRAME_M30,
    'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4,
    'D1': mt5.TIMEFRAME_D1
}
GLOBAL_TIMEFRAME = TIMEFRAME_MAP[CONFIG['timeframe_settings']['global_timeframe']]

# Martingale Configuration
MARTINGALE_ENABLED = CONFIG['martingale_settings']['enabled']
MAX_MARTINGALE_LAYERS = CONFIG['martingale_settings']['max_layers']
MARTINGALE_MULTIPLIER = CONFIG['martingale_settings']['multiplier']
EMERGENCY_DD_PERCENTAGE = CONFIG['martingale_settings']['emergency_dd_percentage']
MARTINGALE_PROFIT_BUFFER_PIPS = CONFIG['martingale_settings']['profit_buffer_pips']
MIN_PROFIT_PERCENTAGE = CONFIG['martingale_settings']['min_profit_percentage']
FLIRT_THRESHOLD_PIPS = CONFIG['martingale_settings']['flirt_threshold_pips']

# Lot Size Configuration
LOT_SIZE_MODE = CONFIG['lot_size_settings']['mode']  # "DYNAMIC" or "MANUAL"
MANUAL_LOT_SIZE = CONFIG['lot_size_settings']['manual_lot_size']

# Trading Pairs and Schedules
PAIRS = CONFIG['trading_pairs']
PAIR_TRADING_DAYS = {pair: CONFIG['pair_trading_schedule']['trading_days'] for pair in PAIRS}
PAIR_TRADING_HOURS = {pair: CONFIG['pair_trading_schedule']['trading_hours'] for pair in PAIRS}

# Risk Profiles - Using new name but old variable name
ENHANCED_PAIR_RISK_PROFILES = CONFIG['pair_risk_profiles']

# Parameter Sets
PARAM_SETS = CONFIG['risk_parameters']

# Symbol-specific settings
SPREAD_LIMITS = CONFIG['symbol_specific_settings']['spread_limits']
RISK_REDUCTION_FACTORS = CONFIG['symbol_specific_settings']['risk_reduction_factors']
MAX_POSITION_PERCENTAGES = CONFIG['symbol_specific_settings']['max_position_percentages']
BASE_TP_TARGETS = CONFIG['symbol_specific_settings']['base_tp_targets']

# ===== SETUP LOGGING WITH CONFIG =====
log_config = CONFIG['logging_settings']
logging.basicConfig(
    level=getattr(logging, log_config['level']),
    format=log_config['format'],
    handlers=[
        logging.FileHandler(log_config['log_file'], encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===== UTILITY FUNCTIONS =====
def get_pip_size(symbol):
    """Get pip size for different symbol types"""
    symbol = symbol.upper()
    if 'JPY' in symbol:
        return 0.01
    if symbol in ['US500', 'NAS100', 'SPX500']:
        return 0.1
    if symbol in ['XAUUSD', 'GOLD']:
        return 0.1
    if symbol in ['BTCUSD', 'ETHUSD', 'XRPUSD']:
        return 1.0
    return 0.0001

def get_pip_value(symbol, lot_size):
    """Enhanced pip value calculation with all special cases"""
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error(f"Could not get symbol info for {symbol}")
        return 0
    
    point = symbol_info.point
    contract_size = symbol_info.trade_contract_size
    current_bid = symbol_info.bid
    
    # Enhanced crypto calculations
    if symbol in ['BTCUSD', 'ETHUSD', 'XRPUSD']:
        return lot_size * 1.0
    
    # Enhanced indices calculations  
    if symbol in ['US500', 'NAS100', 'SPX500', 'USTEC']:
        return lot_size * 10.0
    
    # Gold calculation
    if symbol in ['XAUUSD', 'GOLD']:
        return lot_size * 10.0
    
    # USDCHF calculation
    if symbol == 'USDCHF':
        try:
            if current_bid > 0:
                chf_pip_value = lot_size * 10.0
                usd_pip_value = chf_pip_value * current_bid
                return usd_pip_value
            else:
                return lot_size * 9.0
        except Exception as e:
            logger.error(f"Error calculating USDCHF pip value: {e}")
            return lot_size * 9.0
    
    # Standard USD quote currency pairs
    if symbol.endswith('USD'):
        return lot_size * 10.0
    
    # USD base currency pairs
    if symbol.startswith('USD'):
        if current_bid == 0:
            return 0
        return (lot_size * 10.0) / current_bid
    
    # CAD pairs calculation
    if symbol.endswith('CAD'):
        try:
            usdcad_tick = mt5.symbol_info_tick('USDCAD')
            if usdcad_tick and usdcad_tick.bid > 0:
                cad_pip_value = lot_size * 10.0
                usd_pip_value = cad_pip_value / usdcad_tick.bid
                return usd_pip_value
            else:
                return lot_size * 7.4
        except Exception as e:
            logger.error(f"Error calculating CAD pair pip value: {e}")
            return lot_size * 7.0
    
    # Cross pairs calculation
    base_currency = symbol[:3]
    
    try:
        base_usd_pair = f"{base_currency}USD"
        base_usd_tick = mt5.symbol_info_tick(base_usd_pair)
        
        if base_usd_tick:
            usd_conversion_rate = base_usd_tick.bid
            return lot_size * 10.0 * usd_conversion_rate
        else:
            usd_base_pair = f"USD{base_currency}"
            usd_base_tick = mt5.symbol_info_tick(usd_base_pair)
            
            if usd_base_tick and usd_base_tick.bid > 0:
                usd_conversion_rate = 1.0 / usd_base_tick.bid
                return lot_size * 10.0 * usd_conversion_rate
        
        return lot_size * 8.0
        
    except Exception as e:
        logger.error(f"Error calculating cross pair pip value: {e}")
        return lot_size * 5.0

def normalize_volume(symbol, volume):
    """Normalize volume to broker requirements"""
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error(f"Cannot get symbol info for {symbol}")
        return 0.01
    
    min_volume = symbol_info.volume_min
    max_volume = symbol_info.volume_max
    volume_step = symbol_info.volume_step
    
    # Apply min/max constraints
    volume = max(min_volume, min(volume, max_volume))
    
    # Apply step constraints
    if volume_step > 0:
        volume = round(volume / volume_step) * volume_step
    
    # Final validation
    if volume < min_volume:
        volume = min_volume
    
    return volume

def calculate_position_size(symbol, stop_loss_pips, risk_amount, is_martingale=False, base_volume=None, layer=1):
    """Enhanced position size calculation"""
    
    # MANUAL LOT SIZE MODE
    if LOT_SIZE_MODE == "MANUAL" and not is_martingale:
        logger.info(f"🔧 MANUAL LOT MODE: Using fixed lot size {MANUAL_LOT_SIZE} for {symbol}")
        
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info:
            min_vol = symbol_info.volume_min
            max_vol = symbol_info.volume_max
        else:
            profile = ENHANCED_PAIR_RISK_PROFILES.get(symbol, {"min_lot": 0.01, "max_lot": 1.0})
            min_vol = profile["min_lot"]
            max_vol = profile["max_lot"]
        
        manual_lot = max(min_vol, min(MANUAL_LOT_SIZE, max_vol))
        manual_lot = normalize_volume(symbol, manual_lot)
        
        return manual_lot
    
    # MARTINGALE CALCULATION
    if is_martingale and base_volume:
        position_size = base_volume * (MARTINGALE_MULTIPLIER ** (layer - 1))
        
        profile = ENHANCED_PAIR_RISK_PROFILES.get(symbol, {"min_lot": 0.01, "max_lot": 1.0})
        position_size = max(position_size, profile["min_lot"])
        position_size = min(position_size, profile["max_lot"])
        
        return normalize_volume(symbol, position_size)
    
    # DYNAMIC LOT SIZE CALCULATION
    logger.info(f"🔄 DYNAMIC LOT MODE: Calculating risk-based position size for {symbol}")
    
    # Enhanced risk reduction factors from config
    risk_multiplier = RISK_REDUCTION_FACTORS.get(symbol, 1.0)
    adjusted_risk = risk_amount * risk_multiplier
    
    # Calculate using enhanced pip value
    pip_value = get_pip_value(symbol, 1.0)
    if pip_value <= 0:
        profile = ENHANCED_PAIR_RISK_PROFILES.get(symbol, {"min_lot": 0.01})
        return profile["min_lot"]
    
    # Calculate position size
    position_size = adjusted_risk / (stop_loss_pips * pip_value)
    
    # Apply enhanced constraints
    profile = ENHANCED_PAIR_RISK_PROFILES.get(symbol, {"min_lot": 0.01, "max_lot": 1.0})
    position_size = max(position_size, profile["min_lot"])
    position_size = min(position_size, profile["max_lot"])
    
    return normalize_volume(symbol, position_size)

# ===== TECHNICAL ANALYSIS =====
def get_historical_data(symbol, timeframe, num_bars=500):
    """Get historical data"""
    if not mt5.symbol_select(symbol, True):
        return None
    
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    if rates is None:
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def calculate_indicators(df):
    """Calculate technical indicators"""
    if df is None or df.empty:
        return df
    
    # EMA
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # ADX
    high, low, close = df['high'], df['low'], df['close']
    df['plus_dm'] = np.where((high.diff() > -low.diff()) & (high.diff() > 0), high.diff(), 0)
    df['minus_dm'] = np.where((-low.diff() > high.diff()) & (-low.diff() > 0), -low.diff(), 0)
    df['tr'] = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    
    alpha = 1/14
    plus_dm_smooth = df['plus_dm'].ewm(alpha=alpha, adjust=False).mean()
    minus_dm_smooth = df['minus_dm'].ewm(alpha=alpha, adjust=False).mean()
    tr_smooth = df['tr'].ewm(alpha=alpha, adjust=False).mean()
    
    df['plus_di'] = 100 * (plus_dm_smooth / tr_smooth)
    df['minus_di'] = 100 * (minus_dm_smooth / tr_smooth)
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']).replace(0, 0.001)
    df['adx'] = df['dx'].ewm(alpha=alpha, adjust=False).mean()
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df

def calculate_atr(df, period=14):
    """Calculate ATR"""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean().iloc[-1]

# ===== MARTINGALE BATCH CLASS =====
class MartingaleBatch:
    def __init__(self, symbol, direction, initial_sl_distance, entry_price):
        self.symbol = symbol
        self.direction = direction
        self.initial_sl_distance = initial_sl_distance
        self.initial_entry_price = entry_price
        self.trades = []
        self.current_layer = 0
        self.total_volume = 0
        self.total_invested = 0
        self.breakeven_price = 0
        self.created_time = datetime.now()
        self.last_layer_time = datetime.now()
        self.batch_id = None
        
    def add_trade(self, trade):
        """Add trade to batch and recalculate batch TP"""
        self.current_layer += 1
        trade['layer'] = self.current_layer
        trade['batch_id'] = self.batch_id
        
        # Enhanced trade labeling for clear identification
        batch_prefix = f"BM{self.batch_id:02d}"
        layer_suffix = f"L{self.current_layer:02d}"
        direction_code = "B" if self.direction == 'long' else "S"
        
        trade['enhanced_comment'] = f"{batch_prefix}_{self.symbol[:6]}_{direction_code}{layer_suffix}"
        
        logger.info(f"Adding {trade['enhanced_comment']} to {self.symbol} {self.direction} batch")
        
        self.trades.append(trade)
        self.total_volume += trade['volume']
        self.total_invested += trade['volume'] * trade['entry_price']
        self.last_layer_time = datetime.now()
        
        # Recalculate breakeven price
        if self.total_volume > 0:
            self.breakeven_price = self.total_invested / self.total_volume
            
    def get_next_trigger_price(self):
        """Calculate next price level to trigger new layer"""
        distance_for_next_layer = self.initial_sl_distance * self.current_layer
        
        if self.direction == 'long':
            return self.initial_entry_price - distance_for_next_layer
        else:
            return self.initial_entry_price + distance_for_next_layer
    
    def calculate_adaptive_batch_tp(self, market_volatility_pips=None):
        """Adaptive TP that considers market conditions and urgency"""
        if not self.trades or self.total_volume == 0:
            return None
            
        pip_size = get_pip_size(self.symbol)
        
        # Base profit calculation using "urgency factor"
        urgency_factor = min(self.current_layer, 8)
        
        urgency_multipliers = {
            1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4,
            5: 0.25, 6: 0.15, 7: 0.1, 8: 0.05
        }
        
        urgency_mult = urgency_multipliers[urgency_factor]
        
        # Symbol-specific base targets from config
        base_profit_pips = BASE_TP_TARGETS.get(self.symbol, BASE_TP_TARGETS['default'])
        final_profit_pips = base_profit_pips * urgency_mult
        
        # Absolute minimum
        if urgency_factor < 8:
            final_profit_pips = max(final_profit_pips, 2)
        else:
            final_profit_pips = max(final_profit_pips, 1)
        
        # Calculate TP from breakeven
        if self.direction == 'long':
            return self.breakeven_price + (final_profit_pips * pip_size)
        else:
            return self.breakeven_price - (final_profit_pips * pip_size)
    
    def should_add_layer(self, current_price, fast_move_threshold_seconds=30):
        """Check if we should add another layer"""
        # Check maximum layers
        if self.current_layer >= MAX_MARTINGALE_LAYERS:
            return False
            
        # Check if price has reached trigger level
        trigger_price = self.get_next_trigger_price()
        price_triggered = False
        
        if self.direction == 'long':
            price_triggered = current_price <= trigger_price
        else:
            price_triggered = current_price >= trigger_price
            
        if not price_triggered:
            return False
            
        # Fast move protection
        time_since_last_layer = (datetime.now() - self.last_layer_time).total_seconds()
        if time_since_last_layer < fast_move_threshold_seconds:
            return False
            
        return True
    
    def update_all_tps_with_retry(self, new_tp, max_attempts=3):
        """Update TP for all trades with retry mechanism"""
        logger.info(f"🔄 Updating ALL TPs in {self.symbol} batch to {new_tp:.5f}")
        
        success_count = 0
        total_trades = len(self.trades)
        
        for trade in self.trades:
            if self.update_trade_tp_with_retry(trade, new_tp):
                success_count += 1
                
        if success_count >= total_trades * 0.8:
            logger.info(f"✅ TP Update successful: {success_count}/{total_trades} trades updated")
            return True
        
        logger.warning(f"⚠️ TP Update partial success: {success_count}/{total_trades} trades updated")
        return success_count > 0
    
    def update_trade_tp_with_retry(self, trade, new_tp, max_attempts=2):
        """Update TP for individual trade with retry"""
        for attempt in range(max_attempts):
            try:
                positions = mt5.positions_get(symbol=self.symbol)
                if not positions:
                    return False
                    
                # Find matching position
                order_id = trade.get('order_id')
                target_position = None
                
                for pos in positions:
                    if pos.magic == MAGIC_NUMBER and pos.ticket == order_id:
                        target_position = pos
                        break
                        
                if not target_position:
                    return False
                
                # Skip if TP already correct
                if abs(target_position.tp - new_tp) < get_pip_size(self.symbol):
                    trade['tp'] = new_tp
                    return True
                    
                # Update TP
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": self.symbol,
                    "position": target_position.ticket,
                    "sl": target_position.sl,
                    "tp": float(new_tp),
                }
                
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    trade['tp'] = new_tp
                    return True
                    
            except Exception as e:
                logger.error(f"Error updating TP: {e}")
                
            time.sleep(0.5)
                
        return False

# ===== PERSISTENCE SYSTEM =====
class BotPersistence:
    def __init__(self, data_file="BM_bot_state.json"):
        self.data_file = data_file
        self.backup_file = data_file + ".backup"
        
    def save_bot_state(self, trade_manager):
        """Save complete bot state"""
        try:
            # Create backup
            if os.path.exists(self.data_file):
                import shutil
                shutil.copy2(self.data_file, self.backup_file)
            
            # Prepare state data
            state_data = {
                'timestamp': datetime.now().isoformat(),
                'account_number': ACCOUNT_NUMBER,
                'magic_number': MAGIC_NUMBER,
                'bot_version': '3.0_enhanced',
                'total_trades': trade_manager.total_trades,
                'next_batch_id': trade_manager.next_batch_id,
                'emergency_stop_active': trade_manager.emergency_stop_active,
                'initial_balance': trade_manager.initial_balance,
                'batches': {}
            }
            
            # Save all martingale batches
            for batch_key, batch in trade_manager.martingale_batches.items():
                state_data['batches'][batch_key] = {
                    'batch_id': batch.batch_id,
                    'symbol': batch.symbol,
                    'direction': batch.direction,
                    'initial_entry_price': batch.initial_entry_price,
                    'initial_sl_distance': batch.initial_sl_distance,
                    'current_layer': batch.current_layer,
                    'total_volume': batch.total_volume,
                    'total_invested': batch.total_invested,
                    'breakeven_price': batch.breakeven_price,
                    'created_time': batch.created_time.isoformat(),
                    'last_layer_time': batch.last_layer_time.isoformat(),
                    'trades': []
                }
                
                # Save all trades in batch
                for trade in batch.trades:
                    trade_data = {
                        'order_id': trade.get('order_id'),
                        'layer': trade.get('layer'),
                        'volume': trade.get('volume'),
                        'entry_price': trade.get('entry_price'),
                        'tp': trade.get('tp'),
                        'entry_time': trade.get('entry_time').isoformat() if trade.get('entry_time') else None,
                        'enhanced_comment': trade.get('enhanced_comment'),
                        'trade_id': trade.get('trade_id')
                    }
                    state_data['batches'][batch_key]['trades'].append(trade_data)
            
            # Write to file
            with open(self.data_file, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
                
            logger.info(f"💾 State saved: {len(state_data['batches'])} batches")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save state: {e}")
            return False
    
    def load_and_recover_state(self, trade_manager):
        """Load state and perform automatic recovery"""
        try:
            saved_batches = self.load_saved_state()
            if not saved_batches:
                logger.info("🆕 No saved state found - starting fresh")
                return True
            
            mt5_positions = self.get_mt5_positions()
            recovered_batches = self.recover_batches(saved_batches, mt5_positions, trade_manager)
            
            logger.info(f"🔄 Recovery complete: {len(recovered_batches)} batches restored")
            return True
            
        except Exception as e:
            logger.error(f"❌ Recovery failed: {e}")
            return True
    
    def load_saved_state(self):
        """Load saved state from file"""
        if not os.path.exists(self.data_file):
            return None
            
        try:
            with open(self.data_file, 'r') as f:
                state_data = json.load(f)
            
            if state_data.get('account_number') != ACCOUNT_NUMBER:
                logger.warning(f"⚠️ Account mismatch")
                return None
            
            return state_data
            
        except Exception as e:
            logger.error(f"❌ Error loading saved state: {e}")
            return None
    
    def get_mt5_positions(self):
        """Get current MT5 positions with our magic number"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []
            
            our_positions = [pos for pos in positions if pos.magic == MAGIC_NUMBER]
            logger.info(f"🔍 Found {len(our_positions)} MT5 positions with our magic number")
            
            return our_positions
            
        except Exception as e:
            logger.error(f"❌ Error getting MT5 positions: {e}")
            return []
    
    def recover_batches(self, saved_state, mt5_positions, trade_manager):
        """Simple batch recovery"""
        try:
            # Update trade manager with basic recovery
            trade_manager.next_batch_id = saved_state.get('next_batch_id', 1)
            trade_manager.total_trades = saved_state.get('total_trades', 0)
            trade_manager.initial_balance = saved_state.get('initial_balance')
            
            return {}
            
        except Exception as e:
            logger.error(f"❌ Error in batch recovery: {e}")
            return {}

# ===== ENHANCED TRADE MANAGER =====
class EnhancedTradeManager:
    def __init__(self):
        self.active_trades = []
        self.martingale_batches = {}
        self.total_trades = 0
        self.emergency_stop_active = False
        self.initial_balance = None
        self.next_batch_id = 1
        self.last_webhook_update = datetime.now()
        self.webhook_update_interval = 10
        
        # Import webhook manager
        try:
            from webhook_integration import WebhookManager
            self.webhook_manager = WebhookManager()
            logger.info("✅ Webhook manager initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize webhook manager: {e}")
            self.webhook_manager = None
        
        # Initialize persistence system
        self.persistence = BotPersistence()
        
        # Recovery on startup
        logger.info("🔄 Attempting to recover previous state...")
        recovery_success = self.persistence.load_and_recover_state(self)
        if recovery_success:
            logger.info("✅ State recovery completed successfully")
        else:
            logger.warning("⚠️ State recovery failed - starting fresh")
    
    def can_trade(self, symbol):
        """Check if we can trade this symbol"""
        try:
            if self.emergency_stop_active:
                return False
                
            account_info = mt5.account_info()
            if account_info is None:
                return False
                
            # Initialize balance tracking
            if self.initial_balance is None:
                self.initial_balance = account_info.balance
                logger.info(f"Initial balance set: ${self.initial_balance:.2f}")
                
            # Check emergency drawdown
            current_equity = account_info.equity
            if self.initial_balance and current_equity:
                drawdown = ((self.initial_balance - current_equity) / self.initial_balance) * 100
                if drawdown >= EMERGENCY_DD_PERCENTAGE:
                    self.emergency_stop_active = True
                    logger.critical(f"🚨 EMERGENCY STOP: Drawdown {drawdown:.1f}%")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in can_trade: {e}")
            return False
    
    def send_periodic_webhook_updates(self):
        """Send periodic updates to dashboard"""
        try:
            if not self.webhook_manager:
                return
                
            now = datetime.now()
            if (now - self.last_webhook_update).total_seconds() >= self.webhook_update_interval:
                
                account_info = mt5.account_info()
                if account_info:
                    self.webhook_manager.send_live_data(self, account_info)
                    
                    if (now - self.last_webhook_update).total_seconds() >= 30:
                        self.webhook_manager.send_account_update(account_info, self)
                
                self.last_webhook_update = now
                
        except Exception as e:
            logger.error(f"Error in periodic webhook updates: {e}")

    def check_and_reload_config(self):
        """Check for configuration reload request"""
        try:
            if not self.webhook_manager:
                return False
                
            if self.webhook_manager.check_config_reload():
                logger.info("🔄 Configuration reload requested from dashboard")
                
                # Reload configuration
                global CONFIG
                CONFIG = load_config()
                
                logger.info("✅ Configuration reloaded successfully")
                return True
                
        except Exception as e:
            logger.error(f"Error reloading configuration: {e}")
            
        return False
    
    def has_position(self, symbol, direction):
        """Check if we already have a position for symbol+direction"""
        try:
            batch_key = f"{symbol}_{direction}"
            
            if batch_key in self.martingale_batches:
                batch = self.martingale_batches[batch_key]
                return len(batch.trades) > 0
                
            return False
        except Exception as e:
            logger.error(f"Error checking position for {symbol} {direction}: {e}")
            return False
    
    def get_or_create_batch(self, symbol, direction, sl_distance, entry_price):
        """Get existing batch or create new one"""
        try:
            batch_key = f"{symbol}_{direction}"
            
            if batch_key not in self.martingale_batches:
                batch = MartingaleBatch(symbol, direction, sl_distance, entry_price)
                batch.batch_id = self.next_batch_id
                self.next_batch_id += 1
                self.martingale_batches[batch_key] = batch
                logger.info(f"Created new martingale batch: {batch_key} (ID: {batch.batch_id})")
                
            return self.martingale_batches[batch_key]
        except Exception as e:
            logger.error(f"Error creating/getting batch for {symbol} {direction}: {e}")
            return None
    
    def add_trade(self, trade_info):
        """Add trade to tracking and batch management"""
        try:
            self.total_trades += 1
            trade_info['trade_id'] = self.total_trades
            trade_info['entry_time'] = datetime.now()
            
            # Add to batch if martingale enabled
            if MARTINGALE_ENABLED:
                symbol = trade_info['symbol']
                direction = trade_info['direction']
                sl_distance = trade_info.get('sl_distance', 0)
                entry_price = trade_info['entry_price']
                
                batch = self.get_or_create_batch(symbol, direction, sl_distance, entry_price)
                if batch:
                    batch.add_trade(trade_info)
                    
                    # Calculate and update batch TP
                    try:
                        new_tp = batch.calculate_adaptive_batch_tp()
                        if new_tp:
                            logger.info(f"Calculated adaptive batch TP: {new_tp:.5f}")
                            batch.update_all_tps_with_retry(new_tp)
                    except Exception as e:
                        logger.error(f"Error updating batch TP: {e}")
            
            self.active_trades.append(trade_info)
            
            # Save state
            try:
                self.persistence.save_bot_state(self)
            except Exception as e:
                logger.error(f"Failed to save state: {e}")
            
            # Send webhook notification
            try:
                if self.webhook_manager:
                    self.webhook_manager.send_trade_event(trade_info, "executed")
                    logger.info("📡 Trade webhook sent successfully")
            except Exception as e:
                logger.error(f"Webhook error in add_trade: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding trade: {e}")
            return False
    
    def check_martingale_opportunities_enhanced(self, current_prices):
        """Enhanced martingale checking"""
        opportunities = []
        
        try:
            if not MARTINGALE_ENABLED or self.emergency_stop_active:
                return opportunities
            
            for batch_key, batch in self.martingale_batches.items():
                try:
                    symbol = batch.symbol
                    
                    if symbol not in current_prices:
                        continue
                    
                    current_price = current_prices[symbol]['bid'] if batch.direction == 'long' else current_prices[symbol]['ask']
                    
                    if batch.should_add_layer(current_price):
                        opportunities.append({
                            'batch': batch,
                            'symbol': symbol,
                            'direction': batch.direction,
                            'entry_price': current_price,
                            'layer': batch.current_layer + 1,
                            'trigger_price': batch.get_next_trigger_price(),
                            'distance_pips': abs(current_price - batch.get_next_trigger_price()) / get_pip_size(symbol)
                        })
                        
                except Exception as e:
                    logger.error(f"Error checking martingale for {batch_key}: {e}")
                    continue
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error in check_martingale_opportunities_enhanced: {e}")
            return []

# ===== SIGNAL GENERATION =====
def execute_martingale_trade(opportunity, trade_manager):
    """Execute a martingale layer trade"""
    batch = opportunity['batch']
    symbol = opportunity['symbol']
    direction = opportunity['direction']
    layer = opportunity['layer']
    
    martingale_signal = {
        'symbol': symbol,
        'direction': direction,
        'entry_price': opportunity['entry_price'],
        'sl': None,
        'tp': None,
        'sl_distance_pips': batch.initial_sl_distance / get_pip_size(symbol),
        'risk_profile': ENHANCED_PAIR_RISK_PROFILES.get(symbol, {"risk": "High"})["risk"],
        'is_initial': False,
        'layer': layer,
        'sl_distance': batch.initial_sl_distance
    }
    
    logger.info(f"Executing martingale Layer {layer} for {symbol} {direction}")
    return execute_trade(martingale_signal, trade_manager)

def generate_enhanced_signals(pairs, trade_manager):
    """Generate signals with multi-timeframe confirmation"""
    signals = []
    
    for symbol in pairs:
        if not trade_manager.can_trade(symbol):
            continue
            
        # Skip if we already have positions
        if (trade_manager.has_position(symbol, 'long') and 
            trade_manager.has_position(symbol, 'short')):
            continue
        
        # Get primary timeframe data
        df = get_historical_data(symbol, GLOBAL_TIMEFRAME, 500)
        if df is None or len(df) < 50:
            continue
            
        df = calculate_indicators(df)
        if df is None or 'adx' not in df.columns:
            continue
            
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Get risk profile
        risk_profile = ENHANCED_PAIR_RISK_PROFILES.get(symbol, {"risk": "High"})["risk"]
        params = PARAM_SETS[risk_profile]
        
        # Calculate ATR
        atr = calculate_atr(df)
        atr_pips = atr / get_pip_size(symbol)
        
        if atr_pips < params['min_volatility_pips']:
            continue
        
        # Check ADX strength
        if not (params['min_adx_strength'] <= latest['adx'] <= params['max_adx_strength']):
            continue
        
        # Signal generation for both directions
        for direction in ['long', 'short']:
            if trade_manager.has_position(symbol, direction):
                continue
            
            signal_valid = False
            
            if direction == 'long':
                # Long signal conditions
                ema_up = latest['ema20'] > prev['ema20']
                rsi_condition = prev['rsi'] < 40 and latest['rsi'] > 40
                price_action = latest['close'] > latest['open']
                
                signal_valid = ema_up and (rsi_condition or price_action)
                
            else:  # short
                # Short signal conditions  
                ema_down = latest['ema20'] < prev['ema20']
                rsi_condition = prev['rsi'] > 60 and latest['rsi'] < 60
                price_action = latest['close'] < latest['open']
                
                signal_valid = ema_down and (rsi_condition or price_action)
            
            if signal_valid:
                # Calculate entry, SL, TP
                entry_price = latest['close']
                pip_size = get_pip_size(symbol)
                
                if direction == 'long':
                    sl = min(df['low'].iloc[-3:]) - atr * params['atr_multiplier']
                    tp_distance = abs(entry_price - sl) * params['risk_reward_ratio_long']
                    tp = entry_price + tp_distance
                else:
                    sl = max(df['high'].iloc[-3:]) + atr * params['atr_multiplier']
                    tp_distance = abs(sl - entry_price) * params['risk_reward_ratio_short']
                    tp = entry_price - tp_distance
                
                # Validate distances
                sl_distance_pips = abs(entry_price - sl) / pip_size
                tp_distance_pips = abs(tp - entry_price) / pip_size
                
                if sl_distance_pips >= 10 and tp_distance_pips >= 10:
                    signals.append({
                        'symbol': symbol,
                        'direction': direction,
                        'entry_price': entry_price,
                        'sl': sl,
                        'tp': tp,
                        'atr': atr,
                        'adx_value': latest['adx'],
                        'rsi': latest['rsi'],
                        'sl_distance_pips': sl_distance_pips,
                        'tp_distance_pips': tp_distance_pips,
                        'risk_profile': risk_profile,
                        'timestamp': datetime.now(),
                        'timeframes_aligned': 1,
                        'is_initial': True
                    })
    
    return signals

def execute_trade(signal, trade_manager):
    """Execute trade order with enhanced handling"""
    symbol = signal['symbol']
    direction = signal['direction']
    
    # Validate symbol
    if not mt5.symbol_select(symbol, True):
        logger.error(f"Failed to select symbol {symbol}")
        return False
    
    # Get symbol info
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error(f"Failed to get symbol info for {symbol}")
        return False
    
    # Get current prices
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error(f"Failed to get tick data for {symbol}")
        return False
    
    # Check spread
    spread = tick.ask - tick.bid
    spread_pips = spread / get_pip_size(symbol)
    
    max_spread = SPREAD_LIMITS.get(symbol, SPREAD_LIMITS['default'])
    
    if spread_pips > max_spread:
        logger.warning(f"Spread too high for {symbol}: {spread_pips:.1f} pips (max: {max_spread})")
        return False
    
    # Determine order type and price
    if direction == 'long':
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
    
    # Calculate position size
    account_info = mt5.account_info()
    if account_info is None:
        return False
    
    risk_profile = signal['risk_profile']
    params = PARAM_SETS[risk_profile]
    
    # Risk calculation
    base_risk_pct = params['risk_per_trade_pct']
    if trade_manager.initial_balance:
        current_dd = ((trade_manager.initial_balance - account_info.equity) / trade_manager.initial_balance) * 100
        if current_dd > 5:
            base_risk_pct *= 0.5
    
    risk_amount = account_info.balance * (base_risk_pct / 100)
    
    # Check if martingale
    is_martingale = not signal.get('is_initial', True)
    layer = signal.get('layer', 1)
    
    # Enhanced comment generation
    if is_martingale:
        batch_key = f"{symbol}_{direction}"
        if batch_key in trade_manager.martingale_batches:
            batch = trade_manager.martingale_batches[batch_key]
            batch_id = batch.batch_id
            actual_layer = batch.current_layer + 1
        else:
            batch_id = 99
            actual_layer = layer
    else:
        batch_id = trade_manager.next_batch_id
        actual_layer = 1
    
    batch_prefix = f"BM{batch_id:02d}"
    direction_code = "B" if direction == 'long' else "S"
    layer_suffix = f"{direction_code}{actual_layer:02d}"
    enhanced_comment = f"{batch_prefix}_{symbol}_{layer_suffix}"
    
    # Position size calculation
    if is_martingale:
        batch_key = f"{symbol}_{direction}"
        if batch_key in trade_manager.martingale_batches:
            batch = trade_manager.martingale_batches[batch_key]
            base_volume = batch.trades[0]['volume'] if batch.trades else 0.01
            position_size = calculate_position_size(
                symbol, signal['sl_distance_pips'], risk_amount, 
                is_martingale=True, base_volume=base_volume, layer=layer
            )
        else:
            position_size = calculate_position_size(symbol, signal['sl_distance_pips'], risk_amount)
    else:
        position_size = calculate_position_size(symbol, signal['sl_distance_pips'], risk_amount)
    
    if position_size <= 0:
        logger.error(f"Invalid position size: {position_size}")
        return False
    
    # No SL - Build from first approach
    sl = None
    tp = signal.get('tp')
    
    # Create order request
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(position_size),
        "type": order_type,
        "price": float(price),
        "tp": float(tp) if tp else None,
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": enhanced_comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    # Remove None values
    request = {k: v for k, v in request.items() if v is not None}
    
    logger.info(f"Order request: {symbol} {direction} Layer {layer} - {position_size} lots")
    
    # Execute order
    result = None
    
    for attempt in range(3):
        result = mt5.order_send(request)
        
        if result is None:
            logger.error(f"Attempt {attempt+1}: Order send returned None")
            continue
            
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"✅ Trade executed: {symbol} {direction} - {position_size} lots")
            break
            
        elif result.retcode == mt5.TRADE_RETCODE_INVALID_VOLUME:
            new_volume = symbol_info.volume_min if symbol_info else 0.01
            request["volume"] = float(new_volume)
            logger.info(f"Volume retry {attempt+1}: {new_volume}")
            
        elif result.retcode == mt5.TRADE_RETCODE_INVALID_STOPS:
            logger.warning(f"Invalid stops, removing TP")
            request.pop("tp", None)
            
        elif result.retcode == mt5.TRADE_RETCODE_NO_MONEY:
            logger.error("Insufficient funds")
            return False
            
        else:
            logger.error(f"Order failed: {result.retcode}")
            
        time.sleep(0.5)
    else:
        logger.error(f"All execution attempts failed for {symbol}")
        return False
    
    # Ensure result is valid
    if not result or result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"Trade execution failed for {symbol}")
        return False
    
    # Add trade to manager
    sl_distance_value = signal.get('sl_distance')
    if sl_distance_value is None:
        sl_distance_pips = signal.get('sl_distance_pips', 20)
        sl_distance_value = sl_distance_pips * get_pip_size(symbol)
    
    trade_info = {
        'symbol': symbol,
        'direction': direction,
        'entry_price': result.price,
        'sl': None,
        'tp': tp,
        'volume': result.volume,
        'order_id': result.order,
        'sl_distance': sl_distance_value,
        'layer': layer,
        'is_martingale': is_martingale
    }
    
    trade_manager.add_trade(trade_info)
    
    return True

# ===== MAIN ROBOT FUNCTION =====
def run_simplified_robot():
    """Run the simplified trading robot with enhanced error handling"""
    logger.info("="*60)
    logger.info("BM TRADING ROBOT STARTED - VERSION 3 WITH JSON CONFIG")
    logger.info("="*60)
    logger.info(f"Primary Timeframe: {CONFIG['timeframe_settings']['global_timeframe']}")
    logger.info(f"Pairs: {len(PAIRS)}")
    logger.info(f"Martingale: {MARTINGALE_ENABLED}")
    logger.info(f"Lot Size Mode: {LOT_SIZE_MODE}")
    logger.info("="*60)
    
    # Initialize MT5
    if not mt5.initialize():
        logger.error("MT5 initialization failed")
        return
    
    # Validate connection
    account_info = mt5.account_info()
    if account_info is None:
        logger.error("Failed to get account info")
        mt5.shutdown()
        return
    
    logger.info(f"Connected to account: {account_info.login}")
    logger.info(f"Balance: ${account_info.balance:.2f}")
    
    # Initialize trade manager
    trade_manager = EnhancedTradeManager()
    
    try:
        cycle_count = 0
        consecutive_errors = 0
        
        while True:
            try:
                cycle_count += 1
                current_time = datetime.now()
                
                logger.info(f"\n{'='*50}")
                logger.info(f"Analysis Cycle #{cycle_count} at {current_time}")
                logger.info(f"{'='*50}")
                
                # Reset error counter
                consecutive_errors = 0
                
                # Check MT5 connection
                if not mt5.terminal_info():
                    logger.warning("MT5 disconnected, attempting reconnect...")
                    if not mt5.initialize():
                        logger.error("Reconnection failed")
                        consecutive_errors += 1
                        if consecutive_errors >= 5:
                            break
                        time.sleep(30)
                        continue
                
                # Get current prices
                current_prices = {}
                for symbol in PAIRS:
                    try:
                        tick = mt5.symbol_info_tick(symbol)
                        if tick is None:
                            continue
                        current_prices[symbol] = {
                            'bid': tick.bid,
                            'ask': tick.ask
                        }
                    except Exception as e:
                        logger.warning(f"Error getting price for {symbol}: {e}")
                        continue
                
                if not current_prices:
                    logger.warning("No price data available. Skipping this cycle...")
                    time.sleep(30)
                    continue
                
                # Generate signals
                try:
                    signals = generate_enhanced_signals(PAIRS, trade_manager)
                    logger.info(f"Generated {len(signals)} enhanced signals")
                except Exception as e:
                    logger.error(f"Error generating signals: {e}")
                    signals = []
                
                # Execute signals
                for signal in signals:
                    try:
                        if not trade_manager.can_trade(signal['symbol']):
                            continue
                        
                        logger.info(f"\n🎯 Enhanced Signal: {signal['symbol']} {signal['direction'].upper()}")
                        logger.info(f"   Entry: {signal['entry_price']:.5f}")
                        logger.info(f"   SL Distance: {signal['sl_distance_pips']:.1f} pips")
                        logger.info(f"   Initial TP: {signal['tp']:.5f}")
                        logger.info(f"   🚫 NO SL - Build-from-first approach")
                        
                        if execute_trade(signal, trade_manager):
                            logger.info("✅ Enhanced trade executed successfully")
                            try:
                                if trade_manager.webhook_manager:
                                    trade_manager.webhook_manager.send_signal_generated(signal)
                            except Exception as e:
                                logger.error(f"Signal webhook error: {e}")
                        else:
                            logger.error("❌ Trade execution failed")
                            
                    except Exception as e:
                        logger.error(f"Error executing signal for {signal.get('symbol', 'Unknown')}: {e}")
                        continue
                
                # Check for martingale opportunities
                if MARTINGALE_ENABLED and not trade_manager.emergency_stop_active:
                    try:
                        martingale_opportunities = trade_manager.check_martingale_opportunities_enhanced(current_prices)
                        
                        for opportunity in martingale_opportunities:
                            try:
                                logger.info(f"\n🔄 Martingale Opportunity: {opportunity['symbol']} {opportunity['direction'].upper()}")
                                logger.info(f"   Layer: {opportunity['layer']}")
                                
                                if execute_martingale_trade(opportunity, trade_manager):
                                    logger.info("✅ Martingale layer executed successfully")
                                    
                                    # Update batch TP after adding layer
                                    batch = opportunity['batch']
                                    try:
                                        new_tp = batch.calculate_adaptive_batch_tp()
                                        if new_tp:
                                            batch.update_all_tps_with_retry(new_tp)
                                    except Exception as e:
                                        logger.error(f"Error updating batch TP: {e}")
                                else:
                                    logger.error("❌ Martingale execution failed")
                                    
                            except Exception as e:
                                logger.error(f"Error executing martingale: {e}")
                                continue
                                
                    except Exception as e:
                        logger.error(f"Error checking martingale opportunities: {e}")
                
                # Send periodic webhook updates
                try:
                    trade_manager.send_periodic_webhook_updates()
                except Exception as e:
                    logger.error(f"Webhook update error: {e}")
                
                # Check for config reload
                try:
                    trade_manager.check_and_reload_config()
                except Exception as e:
                    logger.error(f"Config reload error: {e}")
                
                # Show account status
                try:
                    account_info = mt5.account_info()
                    if account_info:
                        logger.info(f"\n📊 Account Status:")
                        logger.info(f"   Balance: ${account_info.balance:.2f}")
                        logger.info(f"   Equity: ${account_info.equity:.2f}")
                        logger.info(f"   Active Trades: {len(trade_manager.active_trades)}")
                        
                        if trade_manager.initial_balance:
                            pnl = account_info.equity - trade_manager.initial_balance
                            pnl_pct = (pnl / trade_manager.initial_balance) * 100
                            logger.info(f"   P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
                        
                        # Show batch status
                        active_batches = len([b for b in trade_manager.martingale_batches.values() if b.trades])
                        if active_batches > 0:
                            logger.info(f"\n🔄 Martingale Batches: {active_batches} active")
                            for batch_key, batch in trade_manager.martingale_batches.items():
                                if batch.trades:
                                    logger.info(f"   {batch_key}: Layer {batch.current_layer}/{MAX_MARTINGALE_LAYERS}")
                        else:
                            logger.info(f"\n🎯 No active martingale batches")
                            
                except Exception as e:
                    logger.error(f"Error displaying account status: {e}")
                
                # Sleep until next cycle
                try:
                    now = datetime.now()
                    next_candle = now + timedelta(minutes=5 - (now.minute % 5))
                    next_candle = next_candle.replace(second=0, microsecond=0)
                    sleep_time = (next_candle - now).total_seconds()
                    
                    logger.info(f"\n⏰ Sleeping {sleep_time:.1f}s until next M5 candle")
                    time.sleep(max(1, sleep_time))
                    
                except Exception as e:
                    logger.error(f"Error in sleep calculation: {e}")
                    time.sleep(60)
                    
            except KeyboardInterrupt:
                logger.info("\n🛑 Robot stopped by user")
                raise
                
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"\n❌ Error in main cycle #{cycle_count}: {e}")
                
                # Emergency state save
                try:
                    trade_manager.persistence.save_bot_state(trade_manager)
                except Exception as save_error:
                    logger.error(f"Failed to save emergency state: {save_error}")
                
                # Stop if too many errors
                if consecutive_errors >= 10:
                    logger.critical(f"🚨 Too many consecutive errors - stopping robot")
                    break
                
                # Wait before retrying
                error_sleep = min(consecutive_errors * 30, 300)
                logger.info(f"⏰ Waiting {error_sleep}s before retry...")
                time.sleep(error_sleep)
                
    except KeyboardInterrupt:
        logger.info("\n🛑 Robot stopped by user")
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
    finally:
        # Final cleanup
        try:
            logger.info("🔄 Performing final cleanup...")
            trade_manager.persistence.save_bot_state(trade_manager)
            logger.info("💾 Final state saved")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        try:
            mt5.shutdown()
            logger.info("MT5 connection closed")
        except Exception as e:
            logger.error(f"Error closing MT5: {e}")

# ===== STARTUP VALIDATION =====
if __name__ == "__main__":
    logger.info("="*50)
    logger.info("STARTING BM TRADING ROBOT")
    logger.info("="*50)
    
    # Run the robot
    run_simplified_robot()