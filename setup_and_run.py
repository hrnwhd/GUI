# ===== COMPLETE SETUP AND STARTUP SCRIPT - FIXED VERSION =====
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
    """Create requirements.txt file with flexible versions"""
    requirements = """MetaTrader5>=5.0.47
pandas>=2.0.0
numpy>=1.24.0
flask>=2.3.0
plotly>=5.15.0
requests>=2.28.0
python-dateutil>=2.8.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    logger.info("Created requirements.txt with flexible versions")

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("Python 3.8 or higher is required")
        return False
    
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro} ✅")
    return True

def install_requirements_with_fallback():
    """Install required packages with fallback options"""
    try:
        logger.info("Installing requirements...")
        
        # Try to install from requirements.txt first
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Requirements installed successfully from requirements.txt")
            return True
        else:
            logger.warning(f"Requirements.txt installation failed: {result.stderr}")
            logger.info("Trying individual package installation with fallbacks...")
            return install_packages_individually()
            
    except Exception as e:
        logger.error(f"Error installing from requirements.txt: {e}")
        logger.info("Trying individual package installation...")
        return install_packages_individually()

def install_packages_individually():
    """Install packages one by one with fallbacks"""
    packages = [
        ("MetaTrader5", "MetaTrader5"),  # Try latest available
        ("pandas", "pandas>=1.5.0"),    # More flexible pandas
        ("numpy", "numpy>=1.21.0"),     # More flexible numpy
        ("flask", "flask>=2.0.0"),      # More flexible flask
        ("plotly", "plotly>=5.0.0"),    # More flexible plotly
        ("requests", "requests>=2.25.0"), # More flexible requests
        ("python-dateutil", "python-dateutil>=2.8.0")
    ]
    
    installed_packages = []
    failed_packages = []
    
    for package_name, package_spec in packages:
        try:
            logger.info(f"Installing {package_name}...")
            
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package_spec
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                logger.info(f"✅ {package_name} installed successfully")
                installed_packages.append(package_name)
            else:
                logger.warning(f"❌ Failed to install {package_name}: {result.stderr}")
                failed_packages.append(package_name)
                
        except subprocess.TimeoutExpired:
            logger.warning(f"⏰ Timeout installing {package_name}")
            failed_packages.append(package_name)
        except Exception as e:
            logger.warning(f"❌ Error installing {package_name}: {e}")
            failed_packages.append(package_name)
    
    logger.info(f"\nInstallation Summary:")
    logger.info(f"✅ Installed: {len(installed_packages)} packages")
    logger.info(f"❌ Failed: {len(failed_packages)} packages")
    
    if failed_packages:
        logger.warning(f"Failed packages: {failed_packages}")
        logger.info("\nYou can manually install failed packages with:")
        for package in failed_packages:
            logger.info(f"  pip install {package}")
    
    # Return True if critical packages are installed
    critical_packages = ["flask", "requests"]
    critical_installed = [p for p in critical_packages if p in installed_packages]
    
    if len(critical_installed) >= len(critical_packages):
        logger.info("✅ Critical packages installed - system can run")
        return True
    else:
        logger.error("❌ Critical packages missing - system may not work properly")
        return False

def test_imports():
    """Test if critical imports work"""
    logger.info("Testing critical imports...")
    
    tests = [
        ("flask", "from flask import Flask"),
        ("requests", "import requests"),
        ("json", "import json"),
        ("datetime", "from datetime import datetime"),
        ("os", "import os"),
        ("logging", "import logging")
    ]
    
    passed = []
    failed = []
    
    for name, import_code in tests:
        try:
            exec(import_code)
            passed.append(name)
            logger.info(f"✅ {name}")
        except ImportError as e:
            failed.append(name)
            logger.error(f"❌ {name}: {e}")
        except Exception as e:
            failed.append(name)
            logger.error(f"❌ {name}: {e}")
    
    logger.info(f"\nImport Test Results:")
    logger.info(f"✅ Passed: {len(passed)}/{len(tests)}")
    
    if failed:
        logger.warning(f"❌ Failed: {failed}")
        return False
    
    return True

def save_html_templates():
    """Save HTML templates to templates directory"""
    logger.info("Checking HTML templates...")
    
    template_files = [
        "dashboard.html",
        "config.html", 
        "trades.html",
        "signals.html"
    ]
    
    existing_templates = []
    missing_templates = []
    
    for filename in template_files:
        filepath = os.path.join("templates", filename)
        if os.path.exists(filepath):
            existing_templates.append(filename)
        else:
            missing_templates.append(filename)
    
    if existing_templates:
        logger.info(f"✅ Found existing templates: {existing_templates}")
    
    if missing_templates:
        logger.warning(f"⚠️ Missing templates: {missing_templates}")
        logger.info("Templates should be created from the provided template files")
    
    return len(existing_templates) > 0

def run_dashboard():
    """Run the Flask dashboard in a separate thread"""
    try:
        logger.info("Starting Flask dashboard...")
        
        # Import with error handling
        try:
            from dashboard import app
            app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
        except ImportError as e:
            logger.error(f"Cannot import dashboard: {e}")
            logger.error("Make sure dashboard.py exists and all dependencies are installed")
        except Exception as e:
            logger.error(f"Dashboard startup error: {e}")
            
    except Exception as e:
        logger.error(f"Dashboard thread error: {e}")

def run_trading_bot():
    """Run the main trading bot"""
    try:
        logger.info("Starting trading bot...")
        
        # Check if bot file exists
        if not os.path.exists("updated_bot_code.py"):
            logger.error("updated_bot_code.py not found!")
            logger.error("Make sure the trading bot file exists in the current directory")
            return
            
        try:
            from updated_bot_code import run_simplified_robot
            run_simplified_robot()
        except ImportError as e:
            logger.error(f"Cannot import trading bot: {e}")
            logger.error("Make sure updated_bot_code.py exists and all dependencies are installed")
        except Exception as e:
            logger.error(f"Trading bot error: {e}")
            
    except Exception as e:
        logger.error(f"Bot startup error: {e}")

def check_existing_files():
    """Check if required files exist"""
    logger.info("Checking required files...")
    
    required_files = [
        ("bot_config.json", "Bot configuration file"),
        ("updated_bot_code.py", "Main trading bot"),
        ("dashboard.py", "Web dashboard"),
        ("webhook_integration.py", "Webhook integration")
    ]
    
    existing = []
    missing = []
    
    for filename, description in required_files:
        if os.path.exists(filename):
            existing.append((filename, description))
            logger.info(f"✅ {filename} - {description}")
        else:
            missing.append((filename, description))
            logger.warning(f"❌ {filename} - {description}")
    
    if missing:
        logger.warning(f"\nMissing files: {len(missing)}")
        logger.info("You may need to create or copy these files before running the system")
        return False
    
    logger.info(f"✅ All {len(required_files)} required files found")
    return True

def main():
    """Main startup function with better error handling"""
    print("="*70)
    print("BM TRADING ROBOT - COMPLETE SYSTEM STARTUP (FIXED VERSION)")
    print("="*70)
    
    # Check Python version
    if not check_python_version():
        print("❌ Python version check failed")
        input("Press Enter to exit...")
        return
    
    # Create directory structure
    create_directory_structure()
    
    # Create flexible requirements file
    create_requirements_file()
    
    # Check existing files
    files_ok = check_existing_files()
    if not files_ok:
        logger.warning("⚠️ Some required files are missing")
        logger.info("The system may not work properly until all files are present")
        
        user_input = input("\nContinue anyway? (y/n): ")
        if user_input.lower() != 'y':
            logger.info("Setup cancelled by user")
            return
    
    # Install requirements with fallbacks
    install_success = install_requirements_with_fallback()
    if not install_success:
        logger.warning("⚠️ Package installation had issues")
        logger.info("You may need to install packages manually")
        
        user_input = input("\nContinue with current packages? (y/n): ")
        if user_input.lower() != 'y':
            logger.info("Setup cancelled by user")
            return
    
    # Test critical imports
    if not test_imports():
        logger.error("❌ Critical import tests failed")
        logger.info("Please install missing packages manually")
        input("Press Enter to exit...")
        return
    
    # Check templates
    save_html_templates()
    
    print("\n" + "="*70)
    print("SYSTEM READY - STARTING COMPONENTS")
    print("="*70)
    
    # Start dashboard in separate thread
    dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
    dashboard_thread.start()
    
    # Wait a moment for dashboard to start
    logger.info("Waiting for dashboard to initialize...")
    time.sleep(5)
    
    # Check if dashboard started
    try:
        import requests
        response = requests.get("http://localhost:5000", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Dashboard is running successfully")
        else:
            logger.warning(f"⚠️ Dashboard responded with status {response.status_code}")
    except Exception as e:
        logger.warning(f"⚠️ Dashboard health check failed: {e}")
        logger.info("Dashboard may still be starting up...")
    
    print(f"\n🌐 Dashboard URL: http://localhost:5000")
    print(f"📊 Config URL: http://localhost:5000/config")
    print(f"📈 Trades URL: http://localhost:5000/trades")
    print(f"📡 Signals URL: http://localhost:5000/signals")
    print(f"⏰ Started at: {datetime.now()}")
    print("="*70)
    
    # Ask user how to proceed
    print("\nSelect startup option:")
    print("1. Start trading bot (full system)")
    print("2. Dashboard only (no trading)")
    print("3. Exit")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            logger.info("Starting full system with trading bot...")
            try:
                run_trading_bot()
            except KeyboardInterrupt:
                logger.info("Trading bot stopped by user")
            except Exception as e:
                logger.error(f"Trading bot error: {e}")
                
        elif choice == "2":
            logger.info("Running dashboard only...")
            print("Dashboard is running. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Dashboard stopped by user")
                
        elif choice == "3":
            logger.info("Exiting...")
            return
            
        else:
            logger.warning("Invalid choice, running dashboard only...")
            print("Dashboard is running. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Dashboard stopped by user")
                
    except KeyboardInterrupt:
        logger.info("Startup interrupted by user")
    except Exception as e:
        logger.error(f"Startup error: {e}")
    finally:
        logger.info("System shutdown complete")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Fatal startup error: {e}")
        input("Press Enter to exit...")