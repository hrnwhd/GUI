# ===== COMPLETE SETUP AND STARTUP SCRIPT =====
# setup_and_run.py - Main startup script for the trading system

import os
import sys
import subprocess
import threading
import time
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_directory_structure():
    """Create the complete directory structure"""
    directories = [
        "gui_data",
        "templates", 
        "static",
        "logs",
        "backups"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def create_requirements_file():
    """Create requirements.txt file"""
    requirements = """MetaTrader5==5.0.45
pandas==2.0.3
numpy==1.24.3
flask==2.3.3
plotly==5.17.0
requests==2.31.0
python-dateutil==2.8.2
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    logger.info("Created requirements.txt")

def save_html_templates():
    """Save HTML templates to templates directory"""
    
    # This would normally save the HTML templates
    # For now, we'll create placeholder files
    templates = {
        "dashboard.html": "<!-- Dashboard template goes here -->",
        "config.html": "<!-- Config template goes here -->", 
        "trades.html": "<!-- Trades template goes here -->",
        "signals.html": "<!-- Signals template goes here -->"
    }
    
    for filename, content in templates.items():
        filepath = os.path.join("templates", filename)
        if not os.path.exists(filepath):
            with open(filepath, "w") as f:
                f.write(content)
            logger.info(f"Created template: {filepath}")

def install_requirements():
    """Install required packages"""
    try:
        logger.info("Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        return False

def run_dashboard():
    """Run the Flask dashboard in a separate thread"""
    try:
        logger.info("Starting Flask dashboard...")
        from dashboard import app
        app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
    except Exception as e:
        logger.error(f"Dashboard error: {e}")

def run_trading_bot():
    """Run the main trading bot"""
    try:
        logger.info("Starting trading bot...")
        # Import and run your existing bot
        from updated_bot_code import run_simplified_robot
        run_simplified_robot()
    except Exception as e:
        logger.error(f"Trading bot error: {e}")

def main():
    """Main startup function"""
    print("="*60)
    print("BM TRADING ROBOT - COMPLETE SYSTEM STARTUP")
    print("="*60)
    
    # Create directory structure
    create_directory_structure()
    
    # Create requirements file
    create_requirements_file()
    
    # Save templates
    save_html_templates()
    
    # Install requirements
    if not install_requirements():
        logger.error("Failed to install requirements. Please install manually.")
        return
    
    print("\n" + "="*60)
    print("SYSTEM READY - STARTING COMPONENTS")
    print("="*60)
    
    # Start dashboard in separate thread
    dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
    dashboard_thread.start()
    
    # Wait a moment for dashboard to start
    time.sleep(3)
    
    print(f"Dashboard URL: http://localhost:5000")
    print(f"Started at: {datetime.now()}")
    print("="*60)
    
    # Start the trading bot (this will run in main thread)
    try:
        run_trading_bot()
    except KeyboardInterrupt:
        logger.info("System stopped by user")
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        logger.info("System shutdown complete")

if __name__ == "__main__":
    main()