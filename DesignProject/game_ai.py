class TicTacToeAI:
    def __init__(self, ai_piece='blue'):
        self.ai_piece = ai_piece
        self.player_piece = 'red' if ai_piece == 'blue' else 'blue'
    
    def is_ai_turn(self, board_state):
        # Count the number of pieces to determine whose turn it is
        # Returns True if it's AI's turn, False if player's turn
        red_count = sum(row.count('red') for row in board_state)
        blue_count = sum(row.count('blue') for row in board_state)
        
        # Red always goes first in Tic-Tac-Toe
        if self.ai_piece == 'blue':
            # AI is blue, player is red
            # It's AI's turn when red has MORE pieces than blue
            return red_count > blue_count
        else:
            # AI is red, player is blue
            # It's AI's turn when red has EQUAL or fewer pieces than blue
            return red_count <= blue_count
    
    def get_best_move(self, board_state):
        best_score = float('-inf')
        best_move = None
        
        for i in range(3):
            for j in range(3):
                if board_state[i][j] == 'empty':
                    # try this location
                    board_state[i][j] = self.ai_piece
                    score = self._minimax(board_state, 0, False)
                    board_state[i][j] = 'empty'
                    
                    if score > best_score:
                        best_score = score
                        best_move = (i, j)
        
        return best_move
    
    def _minimax(self, board, depth, is_maximizing):
        winner = self._check_winner(board)
        
        if winner == self.ai_piece:
            return 10 - depth
        elif winner == self.player_piece:
            return depth - 10
        elif self._is_board_full(board):
            return 0
        
        if is_maximizing:
            best_score = float('-inf')
            for i in range(3):
                for j in range(3):
                    if board[i][j] == 'empty':
                        board[i][j] = self.ai_piece
                        score = self._minimax(board, depth + 1, False)
                        board[i][j] = 'empty'
                        best_score = max(score, best_score)
            return best_score
        else:
            best_score = float('inf')
            for i in range(3):
                for j in range(3):
                    if board[i][j] == 'empty':
                        board[i][j] = self.player_piece
                        score = self._minimax(board, depth + 1, True)
                        board[i][j] = 'empty'
                        best_score = min(score, best_score)
            return best_score
    
    def _check_winner(self, board):
        # Check line
        for i in range(3):
            if board[i][0] == board[i][1] == board[i][2] != 'empty':
                return board[i][0]
        
        # Check column
        for j in range(3):
            if board[0][j] == board[1][j] == board[2][j] != 'empty':
                return board[0][j]
        
        # Check the diagonal
        if board[0][0] == board[1][1] == board[2][2] != 'empty':
            return board[0][0]
        if board[0][2] == board[1][1] == board[2][0] != 'empty':
            return board[0][2]
        
        return None
    
    def _is_board_full(self, board):
        for row in board:
            if 'empty' in row:
                return False
        return True