#!/usr/bin/env python3
"""Quick start script for the Multimodal Generative Models project."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Main quick start function."""
    print("🎨 Multimodal Generative Models - Quick Start")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("src").exists():
        print("❌ Please run this script from the project root directory")
        return 1
    
    # Test installation
    print("\n📋 Step 1: Testing Installation")
    if not run_command("python scripts/test_installation.py", "Testing installation"):
        print("❌ Installation test failed. Please check dependencies.")
        return 1
    
    # Create sample data
    print("\n📋 Step 2: Creating Sample Data")
    if not run_command("python -m src.app --create-sample-data", "Creating sample dataset"):
        print("❌ Sample data creation failed.")
        return 1
    
    # Run a quick demo
    print("\n📋 Step 3: Running Quick Demo")
    print("This will generate a few sample images and descriptions...")
    if not run_command("python -m src.app --demo", "Running demonstration"):
        print("❌ Demo failed. Check the logs for details.")
        return 1
    
    print("\n🎉 Quick start completed successfully!")
    print("\n📚 Next Steps:")
    print("1. 🖥️  Launch the interactive demo:")
    print("   streamlit run demo/app.py")
    print("\n2. 🔧 Customize configuration:")
    print("   Edit configs/default.yaml")
    print("\n3. 🧪 Run tests:")
    print("   pytest tests/")
    print("\n4. 📖 Read the documentation:")
    print("   See README.md for detailed usage")
    
    print("\n⚠️  Important Notes:")
    print("- This is for research/educational purposes only")
    print("- Generated content may not be appropriate for all contexts")
    print("- Ensure you have sufficient GPU memory for best performance")
    print("- Check safety settings in the configuration")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
