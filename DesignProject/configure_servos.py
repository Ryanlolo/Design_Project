"""
Configuration tool for setting servo PWM and TIME values for each board position.
Use this tool to configure the 4 servos for each of the 9 board positions.
"""
from servo_config import ServoConfig


def print_menu():
    """Print main menu"""
    print("\n" + "=" * 50)
    print("Servo Configuration Tool")
    print("=" * 50)
    print("1. View current configuration")
    print("2. Set servo values for a position")
    print("3. Set all servos for a position at once")
    print("4. Set PWM value for a specific servo at a position")
    print("5. Set TIME value for a specific servo at a position")
    print("6. Export configuration to file")
    print("7. Exit")
    print("=" * 50)


def set_position_servos(config, row, col):
    """Set all 4 servos for a position"""
    print(f"\nConfiguring position ({row}, {col})")
    print("Enter values for 4 servos:")
    
    servos = []
    for i in range(4):
        print(f"\nServo {i}:")
        try:
            pwm = int(input(f"  PWM (500-2500, default 1500): ") or "1500")
            time = int(input(f"  TIME in ms (0-9999, default 1000): ") or "1000")
            servos.append({"id": i, "pwm": pwm, "time": time})
        except ValueError:
            print("Invalid input, using defaults")
            servos.append({"id": i, "pwm": 1500, "time": 1000})
    
    if config.set_all_servos_for_position(row, col, servos):
        print(f"Successfully configured position ({row}, {col})")
    else:
        print(f"Error configuring position ({row}, {col})")


def set_single_servo_pwm(config):
    """Set PWM value for a specific servo"""
    try:
        row = int(input("Enter row (0-2): "))
        col = int(input("Enter col (0-2): "))
        servo_id = int(input("Enter servo ID (0-3): "))
        pwm = int(input("Enter PWM value (500-2500): "))
        
        # Get current time value to preserve it
        servos = config.get_servo_config(row, col)
        if servos:
            current_time = servos[servo_id]['time']
            if config.set_servo_config(row, col, servo_id, pwm, current_time):
                print(f"Successfully set PWM for servo {servo_id} at position ({row}, {col})")
            else:
                print("Error setting PWM value")
        else:
            # Position doesn't exist, create with default time
            if config.set_servo_config(row, col, servo_id, pwm, 1000):
                print(f"Successfully set PWM for servo {servo_id} at position ({row}, {col})")
            else:
                print("Error setting PWM value")
    except ValueError:
        print("Invalid input")


def set_single_servo_time(config):
    """Set TIME value for a specific servo"""
    try:
        row = int(input("Enter row (0-2): "))
        col = int(input("Enter col (0-2): "))
        servo_id = int(input("Enter servo ID (0-3): "))
        time = int(input("Enter TIME value in ms (0-9999): "))
        
        # Get current PWM value
        servos = config.get_servo_config(row, col)
        if servos:
            current_pwm = servos[servo_id]['pwm']
            if config.set_servo_config(row, col, servo_id, current_pwm, time):
                print(f"Successfully set TIME for servo {servo_id} at position ({row}, {col})")
            else:
                print("Error setting TIME value")
        else:
            print("Error: Position not found")
    except ValueError:
        print("Invalid input")
    except (IndexError, KeyError):
        print("Error: Invalid servo ID or position")


def main():
    """Main configuration tool"""
    config = ServoConfig()
    
    print("Servo Configuration Tool")
    print("This tool helps you configure PWM and TIME values for each board position.")
    print("You can use your controller to find the correct values, then enter them here.")
    
    while True:
        print_menu()
        choice = input("\nSelect an option (1-7): ").strip()
        
        if choice == '1':
            config.print_config()
        
        elif choice == '2':
            try:
                row = int(input("Enter row (0-2): "))
                col = int(input("Enter col (0-2): "))
                if 0 <= row <= 2 and 0 <= col <= 2:
                    set_position_servos(config, row, col)
                else:
                    print("Invalid row or col value")
            except ValueError:
                print("Invalid input")
        
        elif choice == '3':
            try:
                row = int(input("Enter row (0-2): "))
                col = int(input("Enter col (0-2): "))
                if 0 <= row <= 2 and 0 <= col <= 2:
                    set_position_servos(config, row, col)
                else:
                    print("Invalid row or col value")
            except ValueError:
                print("Invalid input")
        
        elif choice == '4':
            set_single_servo_pwm(config)
        
        elif choice == '5':
            set_single_servo_time(config)
        
        elif choice == '6':
            config._save_config()
            print("Configuration saved!")
        
        elif choice == '7':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice, please try again")


if __name__ == "__main__":
    main()
