import subprocess
import sys
import importlib.util
import os

class AutoInstaller:
    """Automatically installs missing dependencies"""
    
    REQUIRED_PACKAGES = {
        'streamlit': 'streamlit',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'plotly': 'plotly',
        'sklearn': 'scikit-learn',
        'tensorflow': 'tensorflow',
        'yfinance': 'yfinance',
        'ta': 'ta',
        'prophet': 'prophet',
        'xgboost': 'xgboost',
        'requests': 'requests',
        'beautifulsoup4': 'beautifulsoup4',
        'selenium': 'selenium'
    }
    
    @staticmethod
    def check_package(package_name):
        """Check if a package is installed"""
        spec = importlib.util.find_spec(package_name)
        return spec is not None
    
    @staticmethod
    def install_package(package_name, pip_name=None):
        """Install a package using pip"""
        if pip_name is None:
            pip_name = package_name
        
        print(f"Installing {pip_name}...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pip_name],
                stdout=subprocess.DEVNULL
            )
            print(f"✓ {pip_name} installed successfully")
            return True
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {pip_name}")
            return False
    
    @classmethod
    def install_all_dependencies(cls):
        """Check and install all required dependencies"""
        print("="*60)
        print("JINNI AUTO-INSTALLER - Checking Dependencies")
        print("="*60)
        
        missing_packages = []
        
        for package, pip_name in cls.REQUIRED_PACKAGES.items():
            if not cls.check_package(package):
                missing_packages.append((package, pip_name))
        
        if not missing_packages:
            print("\n✓ All dependencies are already installed!\n")
            return True
        
        print(f"\nFound {len(missing_packages)} missing packages")
        print("Installing missing dependencies...\n")
        
        failed = []
        for package, pip_name in missing_packages:
            if not cls.install_package(package, pip_name):
                failed.append(pip_name)
        
        if failed:
            print(f"\n⚠ Warning: Could not install: {', '.join(failed)}")
            print("Please install them manually using:")
            for pkg in failed:
                print(f"  pip install {pkg}")
            return False
        
        print("\n" + "="*60)
        print("✓ All dependencies installed successfully!")
        print("="*60 + "\n")
        return True
    
    @staticmethod
    def upgrade_pip():
        """Upgrade pip to latest version"""
        print("Upgrading pip...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                stdout=subprocess.DEVNULL
            )
            print("✓ pip upgraded successfully")
            return True
        except:
            return False

def auto_setup():
    """Main setup function to be called on import"""
    installer = AutoInstaller()
    installer.upgrade_pip()
    return installer.install_all_dependencies()

if __name__ == "__main__":
    auto_setup()
