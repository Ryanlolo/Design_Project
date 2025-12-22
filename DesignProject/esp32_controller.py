"""
ESP32 controller module for generating and sending servo commands.
Converts board positions to ESP32 servo control commands.
"""
from servo_config import ServoConfig


class ESP32Controller:
    """
    Generates ESP32 servo control commands in the format:
    Single servo: #000P1500T1000!
    Multiple servos: {#000P1500T1000!#001P2500T0000!#002P1500T1000!}
    """
    
    def __init__(self, config_file='servo_config.json'):
        """
        Initialize ESP32 controller.
        
        Args:
            config_file: Path to servo configuration file
        """
        self.servo_config = ServoConfig(config_file)
    
    def format_servo_command(self, servo_id, pwm, time):
        """
        Format a single servo command.
        
        Args:
            servo_id: Servo ID (0-254)
            pwm: PWM value (500-2500)
            time: Time in ms (0-9999)
            
        Returns:
            Formatted command string: #000P1500T1000!
        """
        # Format servo ID (3 digits, zero-padded)
        id_str = f"{servo_id:03d}"
        
        # Format PWM (4 digits, zero-padded)
        pwm_str = f"{pwm:04d}"
        
        # Validate PWM range
        if pwm < 500 or pwm > 2500:
            print(f"Warning: PWM {pwm} is outside recommended range (500-2500)")
        
        # Format TIME (4 digits, zero-padded)
        time_str = f"{time:04d}"
        
        # Validate TIME range
        if time < 0 or time > 9999:
            print(f"Warning: TIME {time} is outside recommended range (0-9999)")
        
        # Build command: #IDPWMVTIME!
        command = f"#{id_str}P{pwm_str}T{time_str}!"
        
        return command
    
    def generate_prepare_position_command(self):
        """
        Generate ESP32 command for prepare position.
        
        Returns:
            Formatted command string for all 4 servos
        """
        servo_configs = self.servo_config.get_prepare_position_config()
        
        if servo_configs is None:
            print("Error: No configuration found for prepare_position")
            return None
        
        # Generate commands for all 4 servos
        commands = []
        for servo in servo_configs:
            cmd = self.format_servo_command(
                servo['id'],
                servo['pwm'],
                servo['time']
            )
            commands.append(cmd)
        
        # If multiple servos, wrap in braces
        if len(commands) > 1:
            return "{" + "".join(commands) + "}"
        elif len(commands) == 1:
            return commands[0]
        else:
            return None
    
    def generate_position_command(self, row, col):
        """
        Generate ESP32 command for a board position.
        
        Args:
            row: Row index (0-2)
            col: Column index (0-2)
            
        Returns:
            Formatted command string for all 4 servos
        """
        servo_configs = self.servo_config.get_servo_config(row, col)
        
        if servo_configs is None:
            print(f"Error: No configuration found for position ({row}, {col})")
            return None
        
        # Generate commands for all 4 servos
        commands = []
        for servo in servo_configs:
            cmd = self.format_servo_command(
                servo['id'],
                servo['pwm'],
                servo['time']
            )
            commands.append(cmd)
        
        # If multiple servos, wrap in braces
        if len(commands) > 1:
            return "{" + "".join(commands) + "}"
        elif len(commands) == 1:
            return commands[0]
        else:
            return None
    
    def generate_custom_command(self, servo_commands):
        """
        Generate ESP32 command from custom servo configurations.
        
        Args:
            servo_commands: List of dicts, each with 'id', 'pwm', 'time'
                          Example: [{'id': 0, 'pwm': 1500, 'time': 1000}, ...]
        
        Returns:
            Formatted command string
        """
        if not servo_commands:
            return None
        
        commands = []
        for servo in servo_commands:
            cmd = self.format_servo_command(
                servo['id'],
                servo['pwm'],
                servo['time']
            )
            commands.append(cmd)
        
        # If multiple servos, wrap in braces
        if len(commands) > 1:
            return "{" + "".join(commands) + "}"
        else:
            return commands[0]
    
    def get_position_info(self, row, col):
        """
        Get servo configuration info for a position (for display/debugging).
        
        Args:
            row: Row index (0-2)
            col: Column index (0-2)
            
        Returns:
            Dictionary with position info and generated command
        """
        servo_configs = self.servo_config.get_servo_config(row, col)
        
        if servo_configs is None:
            return None
        
        command = self.generate_position_command(row, col)
        
        return {
            "position": (row, col),
            "servos": servo_configs,
            "command": command
        }
        