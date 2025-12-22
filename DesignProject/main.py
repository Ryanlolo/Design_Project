import os
# Suppress OpenCV warnings (MSMF backend warnings are harmless when using RealSense)
# Must be set before importing cv2
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

import cv2
import numpy as np
import warnings
import time
from camera_processor import CameraProcessor
from board_detector import BoardDetector
from piece_classifier import PieceClassifier
from game_ai import TicTacToeAI
from esp32_controller import ESP32Controller
from serial_communicator import SerialCommunicator

def main():
    # Suppress Python warnings from OpenCV
    warnings.filterwarnings('ignore', category=UserWarning)
    # Configuration: Set which color the AI plays
    # Options: 'red' or 'blue'
    # Note: Red always goes first in Tic-Tac-Toe
    AI_COLOR = 'blue'  # Change to 'red' if you want AI to play red
    
    # ESP32 Configuration
    SERIAL_PORT = None  # Set to specific port (e.g., 'COM3' on Windows) or None for auto-detect
    SERIAL_BAUDRATE = 128000  # Match ESP32 USB_SERIAL_BAUDRATE (changed to 128000)
    AUTO_SEND_DELAY = 10  # Seconds to wait before auto-sending command
    PREPARE_TO_TARGET_DELAY = 5  # Seconds to wait between prepare position and target position
    
    #Begin
    camera = CameraProcessor()
    detector = BoardDetector()
    classifier = PieceClassifier('models/piece_classifier.h5')
    game_ai = TicTacToeAI(ai_piece=AI_COLOR)
    
    # Initialize ESP32 controller and serial communication
    esp32_controller = ESP32Controller()
    serial_comm = SerialCommunicator(port=SERIAL_PORT, baudrate=SERIAL_BAUDRATE)
    
    # Try to connect to ESP32
    if SERIAL_PORT is None:
        print("\n=== ESP32 Connection ===")
        ports = serial_comm.print_available_ports()
        if ports:
            try:
                port_choice = input("Enter port number to connect (or press Enter to skip): ").strip()
                if port_choice:
                    selected_port = ports[int(port_choice) - 1]['device']
                    serial_comm.connect(selected_port)
                else:
                    print("Skipping ESP32 connection")
            except (ValueError, IndexError):
                print("Invalid port selection, skipping ESP32 connection")
        else:
            print("No serial ports available, ESP32 features disabled")
    else:
        serial_comm.connect(SERIAL_PORT)
    
    print("\nAI system starting...")
    print(f"AI is playing as: {AI_COLOR.upper()}")
    print(f"Player is playing as: {'RED' if AI_COLOR == 'blue' else 'BLUE'}")
    print("Red goes first in Tic-Tac-Toe")
    print("\nControls:")
    print("  'q' - quit")
    print("  'r' - reset game board")
    print("  's' - send command to ESP32 (when AI suggests a move)")
    print(f"  Auto-send after {AUTO_SEND_DELAY} seconds when AI suggests a move")
    
    # Track AI move state
    last_ai_move = None
    ai_move_time = None
    command_sent = False
    prepare_sent = False
    prepare_sent_time = None
    
    while True:
        # Capture camera
        frame = camera.get_frame()
        if frame is None:
            break
        
        try:
            # Detect game board
            board_roi = detector.detect_board(frame)
            if board_roi is not None:
                # Analyze board status
                board_state = classifier.analyze_board(board_roi)
                
                # Show board status
                display_frame = detector.draw_board_state(frame, board_state)
                
                # AI decision
                if game_ai.is_ai_turn(board_state):
                    ai_move = game_ai.get_best_move(board_state)
                    if ai_move:
                        display_frame = detector.draw_ai_suggestion(display_frame, ai_move)
                        
                        # Check if this is a new AI move
                        if ai_move != last_ai_move:
                            print(f"AI recommended location: {ai_move}")
                            last_ai_move = ai_move
                            ai_move_time = time.time()
                            command_sent = False
                            prepare_sent = False
                            prepare_sent_time = None
                        
                        # Check if it's time to auto-send command
                        if not command_sent and ai_move_time is not None:
                            elapsed_time = time.time() - ai_move_time
                            remaining_time = AUTO_SEND_DELAY - elapsed_time
                            
                            if remaining_time > 0:
                                # Display countdown
                                countdown_text = f"Auto-send in: {int(remaining_time)}s (Press 's' to send now)"
                                cv2.putText(display_frame, countdown_text, (20, display_frame.shape[0] - 30),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            else:
                                # Two-step process: prepare_position -> wait 5s -> target_position
                                if serial_comm.is_connected:
                                    # Step 1: Send prepare position command
                                    if not prepare_sent:
                                        prepare_cmd = esp32_controller.generate_prepare_position_command()
                                        if prepare_cmd:
                                            serial_comm.send_command(prepare_cmd)
                                            print(f"Auto-sent prepare position command: {prepare_cmd}")
                                            prepare_sent = True
                                            prepare_sent_time = time.time()
                                        else:
                                            print("Error: Could not generate prepare position command")
                                    # Step 2: Wait 5 seconds, then send target position command
                                    elif prepare_sent_time is not None:
                                        time_since_prepare = time.time() - prepare_sent_time
                                        if time_since_prepare >= PREPARE_TO_TARGET_DELAY:
                                            target_cmd = esp32_controller.generate_position_command(ai_move[0], ai_move[1])
                                            if target_cmd:
                                                serial_comm.send_command(target_cmd)
                                                print(f"Auto-sent target position command: {target_cmd}")
                                                command_sent = True
                                            else:
                                                print(f"Error: Could not generate command for position {ai_move}")
                                else:
                                    print("ESP32 not connected, cannot send command")
                                    command_sent = True  # Mark as sent to avoid retry
                
                cv2.imshow('Tic Tac Toe AI', display_frame)
            else:
                cv2.imshow('Tic Tac Toe AI', frame)
                #print("No chessboard detected, please make sure the chessboard is in the screen")
        
        except Exception as e:
            print(f"handling errors: {e}")
            cv2.imshow('Tic Tac Toe AI', frame)
        
        # keyboard control
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("Reset board state")
            last_ai_move = None
            ai_move_time = None
            command_sent = False
            prepare_sent = False
            prepare_sent_time = None
        elif key == ord('s'):
            # Manual send command (two-step process)
            if last_ai_move is not None and not command_sent:
                if serial_comm.is_connected:
                    # Step 1: Send prepare position command
                    if not prepare_sent:
                        prepare_cmd = esp32_controller.generate_prepare_position_command()
                        if prepare_cmd:
                            serial_comm.send_command(prepare_cmd)
                            print(f"Manually sent prepare position command: {prepare_cmd}")
                            prepare_sent = True
                            prepare_sent_time = time.time()
                        else:
                            print("Error: Could not generate prepare position command")
                    # Step 2: Wait 5 seconds, then send target position command
                    elif prepare_sent_time is not None:
                        time_since_prepare = time.time() - prepare_sent_time
                        if time_since_prepare >= PREPARE_TO_TARGET_DELAY:
                            target_cmd = esp32_controller.generate_position_command(last_ai_move[0], last_ai_move[1])
                            if target_cmd:
                                serial_comm.send_command(target_cmd)
                                print(f"Manually sent target position command: {target_cmd}")
                                command_sent = True
                            else:
                                print(f"Error: Could not generate command for position {last_ai_move}")
                        else:
                            remaining = PREPARE_TO_TARGET_DELAY - time_since_prepare
                            print(f"Waiting {remaining:.1f}s before sending target position...")
                else:
                    print("ESP32 not connected, cannot send command")
            elif command_sent:
                print("Command already sent for this move")
            else:
                print("No AI move available to send")
    
    # Cleanup
    camera.release()
    serial_comm.disconnect()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()