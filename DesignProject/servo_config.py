"""
Servo configuration module for board positions.
Stores PWM and TIME values for 4 servos at each of the 9 board positions.
"""
import json
import os


class ServoConfig:
    """
    Manages servo configuration for each board position.
    Each position (row, col) has 4 servos with PWM and TIME values.
    """
    
    def __init__(self, config_file='servo_config.json'):
        """
        Initialize servo configuration.
        
        Args:
            config_file: Path to JSON configuration file
        """
        # Get absolute path to ensure we can find the file
        if not os.path.isabs(config_file):
            # If relative path, look in the same directory as this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.config_file = os.path.join(script_dir, config_file)
        else:
            self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self):
        """
        Load configuration from JSON file.
        Creates default structure if file doesn't exist.
        
        Returns:
            Dictionary with configuration data
        """
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config file: {e}")
                print("Creating default configuration structure...")
                return self._create_default_config()
        else:
            print(f"Config file not found: {self.config_file}")
            print("Creating default configuration structure...")
            config = self._create_default_config()
            self._save_config(config)
            return config
    
    def _create_default_config(self):
        """
        Create default configuration structure.
        All PWM and TIME values are set to default values (needs to be configured).
        
        Returns:
            Default configuration dictionary
        """
        config = {
            "prepare_position": {
                "servos": [
                    {"id": 0, "pwm": 1500, "time": 2000},
                    {"id": 1, "pwm": 1500, "time": 2000},
                    {"id": 2, "pwm": 1500, "time": 2000},
                    {"id": 3, "pwm": 1500, "time": 2000}
                ]
            },
            "positions": {}
        }
        
        # Create entries for all 9 positions (0,0) to (2,2)
        # Use format "(row,col)" to match existing JSON file format
        for row in range(3):
            for col in range(3):
                pos_key = f"({row},{col})"
                config["positions"][pos_key] = {
                    "servos": [
                        {"id": 0, "pwm": 1500, "time": 2000},  # Servo 0
                        {"id": 1, "pwm": 1500, "time": 2000},  # Servo 1
                        {"id": 2, "pwm": 1500, "time": 2000},  # Servo 2
                        {"id": 3, "pwm": 1500, "time": 2000}   # Servo 3
                    ]
                }
        
        return config
    
    def _save_config(self, config=None):
        """Save configuration to JSON file"""
        if config is None:
            config = self.config
        
        # Only create directory if config_file has a directory path
        dir_path = os.path.dirname(self.config_file)
        if dir_path:  # Only create directory if path is not empty
            os.makedirs(dir_path, exist_ok=True)
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"Error saving config file: {e}")
    
    def get_servo_config(self, row, col):
        """
        Get servo configuration for a specific board position.
        
        Args:
            row: Row index (0-2)
            col: Column index (0-2)
            
        Returns:
            List of 4 servo configurations, each with id, pwm, and time
        """
        # Try format "(row,col)" first (matches JSON file format)
        pos_key = f"({row},{col})"
        if pos_key in self.config["positions"]:
            return self.config["positions"][pos_key]["servos"]
        
        # Fallback to format "row_col" for backward compatibility
        pos_key_alt = f"{row}_{col}"
        if pos_key_alt in self.config["positions"]:
            return self.config["positions"][pos_key_alt]["servos"]
        
        print(f"Warning: Position ({row}, {col}) not found in config")
        return None
    
    def get_prepare_position_config(self):
        """
        Get servo configuration for prepare position.
        
        Returns:
            List of 4 servo configurations, each with id, pwm, and time
            Returns None if prepare_position not found
        """
        if "prepare_position" in self.config:
            return self.config["prepare_position"]["servos"]
        else:
            print("Warning: prepare_position not found in config")
            return None
    
    def set_servo_config(self, row, col, servo_id, pwm=None, time=None):
        """
        Set PWM and/or TIME values for a specific servo at a specific position.
        
        Args:
            row: Row index (0-2)
            col: Column index (0-2)
            servo_id: Servo ID (0-3)
            pwm: PWM value (500-2500), None to keep current value
            time: Time value in ms (0-9999), None to keep current value
        """
        # Validate inputs
        if row < 0 or row > 2 or col < 0 or col > 2:
            print(f"Error: Invalid position ({row}, {col})")
            return False
        
        if servo_id < 0 or servo_id > 3:
            print(f"Error: Invalid servo ID {servo_id}")
            return False
        
        # Use format "(row,col)" to match JSON file format
        pos_key = f"({row},{col})"
        
        # Initialize position if it doesn't exist
        if pos_key not in self.config["positions"]:
            self.config["positions"][pos_key] = {"servos": []}
            for i in range(4):
                self.config["positions"][pos_key]["servos"].append(
                    {"id": i, "pwm": 1500, "time": 1000}
                )
        
        # Get current values if not provided
        current_pwm = self.config["positions"][pos_key]["servos"][servo_id]["pwm"]
        current_time = self.config["positions"][pos_key]["servos"][servo_id]["time"]
        
        # Use provided values or keep current
        new_pwm = pwm if pwm is not None else current_pwm
        new_time = time if time is not None else current_time
        
        # Validate ranges
        if new_pwm < 500 or new_pwm > 2500:
            print(f"Warning: PWM value {new_pwm} is outside recommended range (500-2500)")
        
        if new_time < 0 or new_time > 9999:
            print(f"Warning: TIME value {new_time} is outside recommended range (0-9999)")
        
        # Update configuration
        self.config["positions"][pos_key]["servos"][servo_id]["id"] = servo_id
        self.config["positions"][pos_key]["servos"][servo_id]["pwm"] = new_pwm
        self.config["positions"][pos_key]["servos"][servo_id]["time"] = new_time
        
        # Save to file
        self._save_config()
        return True
    
    def set_all_servos_for_position(self, row, col, servo_configs):
        """
        Set all 4 servos for a position at once.
        
        Args:
            row: Row index (0-2)
            col: Column index (0-2)
            servo_configs: List of 4 dicts, each with 'id', 'pwm', 'time'
                          Example: [{'id': 0, 'pwm': 1500, 'time': 1000}, ...]
        """
        if len(servo_configs) != 4:
            print(f"Error: Expected 4 servo configurations, got {len(servo_configs)}")
            return False
        
        for i, servo_config in enumerate(servo_configs):
            self.set_servo_config(
                row, col,
                servo_config.get('id', i),
                servo_config.get('pwm', 1500),
                servo_config.get('time', 1000)
            )
        
        return True
    
    def print_config(self):
        """Print current configuration"""
        print("\n=== Servo Configuration ===")
        for row in range(3):
            for col in range(3):
                # Try both formats
                pos_key = f"({row},{col})"
                if pos_key not in self.config["positions"]:
                    pos_key = f"{row}_{col}"
                
                if pos_key in self.config["positions"]:
                    servos = self.config["positions"][pos_key]["servos"]
                    print(f"\nPosition ({row}, {col}):")
                    for servo in servos:
                        print(f"  Servo {servo['id']}: PWM={servo['pwm']}, TIME={servo['time']}ms")
        print("=" * 30)
