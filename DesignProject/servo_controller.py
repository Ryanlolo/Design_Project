"""
Servo Controller Module for ESP32 Robot Arm
Handles coordinate to servo command conversion and serial communication
"""
import serial
import json
import os
import time
from typing import Dict, Tuple, Optional, List


class ServoController:
    """
    Controller for ESP32 servo system.
    Converts board coordinates to servo commands and sends via UART.
    """
    
    def __init__(self, config_file='servo_config.json', serial_port=None, baudrate=115200):
        """
        Initialize servo controller.
        
        Args:
            config_file: Path to JSON config file containing PWM/TIME values for each position
            serial_port: Serial port name (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
            baudrate: Serial communication baudrate
        """
        self.config_file = config_file
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.serial_connection = None
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load servo configuration from JSON file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"Servo configuration loaded from {self.config_file}")
                return config
            except Exception as e:
                print(f"Error loading config file: {e}")
                return self._create_default_config()
        else:
            print(f"Config file not found. Creating default config: {self.config_file}")
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict:
        """Create default configuration template"""
        default_config = {
            "positions": {}
        }
        
        # Create template for all 9 positions
        for row in range(3):
            for col in range(3):
                pos_key = f"({row},{col})"
                default_config["positions"][pos_key] = {
                    "servos": [
                        {"id": 0, "pwm": 1500, "time": 1000},
                        {"id": 1, "pwm": 1500, "time": 1000},
                        {"id": 2, "pwm": 1500, "time": 1000},
                        {"id": 3, "pwm": 1500, "time": 1000}
                    ]
                }
        
        # Save default config
        self._save_config(default_config)
        return default_config
    
    def _save_config(self, config: Dict):
        """Save configuration to JSON file"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"Error saving config file: {e}")
    
    def connect(self, port: Optional[str] = None) -> bool:
        """
        Connect to ESP32 via serial port.
        
        Args:
            port: Serial port name (if None, uses self.serial_port)
            
        Returns:
            True if connection successful, False otherwise
        """
        if port:
            self.serial_port = port
        
        if not self.serial_port:
            print("No serial port specified. Please provide port name.")
            return False
        
        try:
            self.serial_connection = serial.Serial(
                port=self.serial_port,
                baudrate=self.baudrate,
                timeout=1
            )
            time.sleep(2)  # Wait for serial connection to stabilize
            print(f"Connected to ESP32 on {self.serial_port} at {self.baudrate} baud")
            return True
        except serial.SerialException as e:
            print(f"Failed to connect to serial port {self.serial_port}: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error connecting to serial port: {e}")
            return False
    
    def disconnect(self):
        """Close serial connection"""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            print("Serial connection closed")
    
    def format_servo_command(self, servo_id: int, pwm: int, time_ms: int) -> str:
        """
        Format single servo command according to ESP32 protocol.
        
        Args:
            servo_id: Servo ID (0-254)
            pwm: PWM value (500-2500)
            time_ms: Time in milliseconds (0-9999)
            
        Returns:
            Formatted command string: #000P1500T1000!
        """
        # Validate and format servo ID (3 digits, zero-padded)
        if servo_id < 0 or servo_id > 254:
            raise ValueError(f"Servo ID must be 0-254, got {servo_id}")
        servo_id_str = f"{servo_id:03d}"
        
        # Validate and format PWM (4 digits, zero-padded)
        if pwm < 500 or pwm > 2500:
            raise ValueError(f"PWM must be 500-2500, got {pwm}")
        pwm_str = f"{pwm:04d}"
        
        # Validate and format TIME (4 digits, zero-padded)
        if time_ms < 0 or time_ms > 9999:
            raise ValueError(f"TIME must be 0-9999, got {time_ms}")
        time_str = f"{time_ms:04d}"
        
        return f"#{servo_id_str}P{pwm_str}T{time_str}!"
    
    def create_multi_servo_command(self, servo_commands: List[str]) -> str:
        """
        Create multi-servo command with {} wrapper.
        
        Args:
            servo_commands: List of individual servo command strings
            
        Returns:
            Formatted multi-servo command: {#000P1500T1000!#001P2500T0000!...}
        """
        if len(servo_commands) == 0:
            return ""
        elif len(servo_commands) == 1:
            return servo_commands[0]
        else:
            # Multiple commands need {} wrapper
            return "{" + "".join(servo_commands) + "}"
    
    def coordinate_to_command(self, row: int, col: int) -> Optional[str]:
        """
        Convert board coordinates to ESP32 servo command.
        
        Args:
            row: Row index (0-2)
            col: Column index (0-2)
            
        Returns:
            Formatted command string or None if position not configured
        """
        if row < 0 or row > 2 or col < 0 or col > 2:
            raise ValueError(f"Coordinates must be (0-2, 0-2), got ({row}, {col})")
        
        pos_key = f"({row},{col})"
        
        if pos_key not in self.config.get("positions", {}):
            print(f"Warning: Position {pos_key} not found in configuration")
            return None
        
        position_config = self.config["positions"][pos_key]
        servo_commands = []
        
        for servo_config in position_config.get("servos", []):
            servo_id = servo_config.get("id")
            pwm = servo_config.get("pwm")
            time_ms = servo_config.get("time")
            
            if servo_id is None or pwm is None or time_ms is None:
                print(f"Warning: Incomplete servo configuration for {pos_key}")
                continue
            
            try:
                cmd = self.format_servo_command(servo_id, pwm, time_ms)
                servo_commands.append(cmd)
            except ValueError as e:
                print(f"Error formatting servo command: {e}")
                continue
        
        if len(servo_commands) == 0:
            return None
        
        return self.create_multi_servo_command(servo_commands)
    
    def send_command(self, command: str) -> bool:
        """
        Send command to ESP32 via serial port.
        
        Args:
            command: Command string to send
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.serial_connection or not self.serial_connection.is_open:
            print("Serial connection not established. Cannot send command.")
            return False
        
        try:
            # Send command with newline terminator
            command_bytes = (command + '\n').encode('utf-8')
            self.serial_connection.write(command_bytes)
            self.serial_connection.flush()
            print(f"Command sent: {command}")
            return True
        except Exception as e:
            print(f"Error sending command: {e}")
            return False
    
    def move_to_position(self, row: int, col: int) -> bool:
        """
        Move robot arm to specified board position.
        
        Args:
            row: Row index (0-2)
            col: Column index (0-2)
            
        Returns:
            True if command sent successfully, False otherwise
        """
        command = self.coordinate_to_command(row, col)
        if command is None:
            return False
        
        return self.send_command(command)
    
    def update_position_config(self, row: int, col: int, servos: List[Dict]):
        """
        Update configuration for a specific position.
        
        Args:
            row: Row index (0-2)
            col: Column index (0-2)
            servos: List of servo config dicts, each with 'id', 'pwm', 'time'
        """
        pos_key = f"({row},{col})"
        
        if "positions" not in self.config:
            self.config["positions"] = {}
        
        self.config["positions"][pos_key] = {"servos": servos}
        self._save_config(self.config)
        print(f"Updated configuration for position {pos_key}")
    
    def list_available_ports(self):
        """List available serial ports (helper function)"""
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        print("Available serial ports:")
        for port in ports:
            print(f"  - {port.device}: {port.description}")
        return [port.device for port in ports]
