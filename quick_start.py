# quick_start.py — Quick start script untuk setup dan testing

"""
Script untuk setup environment dan quick testing
Jalankan: python quick_start.py
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header(text):
    """Print header dengan formatting"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def check_python_version():
    """Check Python version"""
    print_header("Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required!")
        return False
    
    print("✅ Python version OK")
    return True

def install_dependencies():
    """Install dependencies dari requirements.txt"""
    print_header("Installing Dependencies")
    
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found!")
        return False
    
    try:
        print("Installing packages... This may take a few minutes.")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt", 
            "--quiet"
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def check_installed_packages():
    """Check jika package penting sudah terinstall"""
    print_header("Checking Installed Packages")
    
    packages = {
        "streamlit": "Streamlit",
        "pandas": "Pandas",
        "numpy": "NumPy",
        "sklearn": "Scikit-learn",
        "rank_bm25": "BM25",
        "sentence_transformers": "Sentence-Transformers",
        "openai": "OpenAI",
        "plotly": "Plotly"
    }
    
    all_ok = True
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"✅ {name:25s} - OK")
        except ImportError:
            print(f"❌ {name:25s} - NOT INSTALLED")
            all_ok = False
    
    return all_ok

def create_directories():
    """Create necessary directories"""
    print_header("Creating Directories")
    
    dirs = ["cache", "exports", "data"]
    
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Created/verified: {dir_name}/")
    
    return True

def generate_sample_data():
    """Generate sample data jika belum ada"""
    print_header("Generating Sample Data")
    
    csv_files = list(Path(".").glob("*.csv"))
    
    if csv_files:
        print(f"ℹ️  Found existing CSV files: {len(csv_files)}")
        response = input("Generate new sample data? (y/n): ").lower()
        if response != 'y':
            print("⏭️  Skipping sample data generation")
            return True
    
    try:
        print("Generating sample data...")
        subprocess.check_call([sys.executable, "generate_sample_data.py"])
        print("✅ Sample data generated")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Could not generate sample data: {e}")
        print("   You can generate it manually later with: python generate_sample_data.py")
        return False
    except FileNotFoundError:
        print("⚠️  generate_sample_data.py not found")
        return False

def run_app():
    """Run Streamlit app"""
    print_header("Launching Application")
    
    if not Path("app_optimized.py").exists():
        print("❌ app_optimized.py not found!")
        return False
    
    print("🚀 Starting Streamlit app...")
    print("   The app will open in your browser at http://localhost:8501")
    print("   Press Ctrl+C to stop the app\n")
    
    try:
        subprocess.call([sys.executable, "-m", "streamlit", "run", "app_optimized.py"])
        return True
    except KeyboardInterrupt:
        print("\n👋 App stopped")
        return True
    except Exception as e:
        print(f"❌ Error running app: {e}")
        return False

def main():
    """Main setup process"""
    print("""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║     🧭 Smart PNS Job Recommender - Quick Start           ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Check Python
    if not check_python_version():
        print("\n❌ Setup failed: Python version too old")
        return
    
    # Step 2: Install dependencies
    response = input("\nInstall dependencies from requirements.txt? (y/n): ").lower()
    if response == 'y':
        if not install_dependencies():
            print("\n❌ Setup failed: Could not install dependencies")
            return
    
    # Step 3: Check packages
    if not check_installed_packages():
        print("\n⚠️  Some packages are missing. Please install them manually:")
        print("    pip install -r requirements.txt")
        response = input("\nContinue anyway? (y/n): ").lower()
        if response != 'y':
            return
    
    # Step 4: Create directories
    create_directories()
    
    # Step 5: Generate sample data
    generate_sample_data()
    
    # Step 6: Launch app
    print_header("Setup Complete!")
    print("✅ All checks passed")
    print("\nOptions:")
    print("  1. Run the application now")
    print("  2. Exit (you can run later with: streamlit run app_optimized.py)")
    
    choice = input("\nEnter your choice (1/2): ").strip()
    
    if choice == '1':
        run_app()
    else:
        print("\n👋 Setup complete! Run the app later with:")
        print("   streamlit run app_optimized.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
