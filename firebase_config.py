# firebase_config.py
"""
Storage configuration - now uses local SQLite instead of Firebase
Provides backward compatibility with FirebaseManager interface
"""

from local_storage import LocalStorageManager
import os

class FirebaseManager(LocalStorageManager):
    """
    Compatibility wrapper for LocalStorageManager.
    Maintains the same API as the original Firebase implementation
    but uses local SQLite storage instead.
    
    No configuration needed - works out of the box!
    """
    
    def __init__(self):
        # Initialize with default database path
        db_path = os.path.join(".jinni_cache", "jinni_data.db")
        super().__init__(db_path=db_path)
        print("✅ Storage initialized: Local SQLite database")
        print(f"📊 Database location: {self.db_path}")
    
    def is_connected(self) -> bool:
        """Check if storage is available"""
        return os.path.exists(self.db_path)
    
    def get_status(self) -> dict:
        """Get storage status"""
        return {
            "type": "sqlite",
            "path": self.db_path,
            "connected": self.is_connected(),
            "persistent": True
        }

# Backward compatibility
StorageManager = FirebaseManager

if __name__ == "__main__":
    # Test the storage
    manager = FirebaseManager()
    print("Storage Status:", manager.get_status())
    
    # Test prediction logging
    pred_id = manager.log_prediction(
        symbol="RELIANCE.NS",
        predicted_price=1550.0,
        confidence=85.5,
        horizon_days=7
    )
    print(f"✅ Test prediction logged with ID: {pred_id}")
