import os
# Suppress OpenCV warnings (MSMF backend warnings are harmless when using RealSense)
# Must be set before importing cv2
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

import cv2
import numpy as np
import warnings
from camera_processor import CameraProcessor
from board_detector import BoardDetector
from piece_classifier import PieceClassifier
from game_ai import TicTacToeAI

def main():
    # Suppress Python warnings from OpenCV
    warnings.filterwarnings('ignore', category=UserWarning)
    # Configuration: Set which color the AI plays
    # Options: 'red' or 'blue'
    # Note: Red always goes first in Tic-Tac-Toe
    AI_COLOR = 'blue'  # Change to 'red' if you want AI to play red
    
    #Begin
    camera = CameraProcessor()
    detector = BoardDetector()
    classifier = PieceClassifier('models/piece_classifier.h5')
    game_ai = TicTacToeAI(ai_piece=AI_COLOR)
    
    print("AI system starting...")
    print(f"AI is playing as: {AI_COLOR.upper()}")
    print(f"Player is playing as: {'RED' if AI_COLOR == 'blue' else 'BLUE'}")
    print("Red goes first in Tic-Tac-Toe")
    print("press'q' quit,press'r' reset game board")
    
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
                        print(f"AI recommended location: {ai_move}")
                
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
    
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()