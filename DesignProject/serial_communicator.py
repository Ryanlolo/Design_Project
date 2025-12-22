"""
Serial communication module for ESP32.
Handles UART communication to send servo commands.
"""
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("Warning: pyserial not installed. ESP32 serial communication will be disabled.")
    print("Install it with: pip install pyserial")

import time


class SerialCommunicator:
    """
    Handles serial communication with ESP32.
    """
    
    def __init__(self, port=None, baudrate=115200, timeout=1):
        """
        Initialize serial communicator.
        
        Args:
            port: Serial port name (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
                 If None, will attempt to auto-detect
            baudrate: Baud rate (default 115200)
            timeout: Read timeout in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_connection = None
        self.is_connected = False
    
    def list_available_ports(self):
        """
        List all available serial ports.
        
        Returns:
            List of available port names
        """
        if not SERIAL_AVAILABLE:
            print("pyserial not available, cannot list ports")
            return []
        
        ports = serial.tools.list_ports.comports()
        port_list = []
        for port in ports:
            port_list.append({
                'device': port.device,
                'description': port.description,
                'hwid': port.hwid
            })
        return port_list
    
    def print_available_ports(self):
        """Print all available serial ports"""
        ports = self.list_available_ports()
        if ports:
            print("\n=== Available Serial Ports ===")
            for i, port in enumerate(ports):
                print(f"{i+1}. {port['device']}: {port['description']}")
            print("=" * 35)
        else:
            print("No serial ports found")
        return ports
    
    def connect(self, port=None):
        """
        Connect to ESP32 via serial port.
        
        Args:
            port: Serial port name (if None, uses self.port)
            
        Returns:
            True if connected successfully, False otherwise
        """
        if not SERIAL_AVAILABLE:
            print("Error: pyserial not available. Install it with: pip install pyserial")
            return False
        
        if port is not None:
            self.port = port
        
        if self.port is None:
            print("Error: No port specified")
            print("Available ports:")
            self.print_available_ports()
            return False
        
        try:
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            
            # Wait for connection to stabilize
            time.sleep(0.1)
            
            self.is_connected = True
            print(f"Connected to {self.port} at {self.baudrate} baud")
            return True
            
        except serial.SerialException as e:
            print(f"Error connecting to {self.port}: {e}")
            self.is_connected = False
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """Disconnect from serial port"""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            self.is_connected = False
            print("Disconnected from serial port")
    
    def send_command(self, command):
        """
        Send command to ESP32.
        
        Args:
            command: Command string to send
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not SERIAL_AVAILABLE:
            print("Error: pyserial not available. Cannot send command.")
            return False
        
        if not self.is_connected or self.serial_connection is None:
            print("Error: Not connected to ESP32")
            return False
        
        if not self.serial_connection.is_open:
            print("Error: Serial port is not open")
            return False
        
        try:
            # Convert string to bytes and send
            command_bytes = command.encode('utf-8')
            
            # Debug: Print command details
            print(f"[Python] Sending command to ESP32: {command}")
            print(f"[Python] Command length: {len(command)} bytes")
            print(f"[Python] Command bytes (hex): {command_bytes.hex()}")
            
            bytes_written = self.serial_connection.write(command_bytes)
            
            # Flush to ensure data is sent immediately
            self.serial_connection.flush()
            
            print(f"[Python] Sent {bytes_written} bytes to ESP32")
            print(f"[Python] Command should be received by ESP32 now")
            return True
            
        except serial.SerialException as e:
            print(f"Error sending command: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error sending command: {e}")
            return False
    
    def read_response(self, max_bytes=1024):
        """
        Read response from ESP32.
        
        Args:
            max_bytes: Maximum bytes to read
            
        Returns:
            Response string or None if error/timeout
        """
        if not self.is_connected or self.serial_connection is None:
            return None
        
        try:
            if self.serial_connection.in_waiting > 0:
                response = self.serial_connection.read(max_bytes)
                return response.decode('utf-8', errors='ignore')
            return None
        except Exception as e:
            print(f"Error reading response: {e}")
            return None
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()
        