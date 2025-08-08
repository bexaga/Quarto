import random
import pandas as pd
from enum import Enum
from itertools import product
import sys

import openai_player

DEBUG = "--debug" in sys.argv

consolidated_log = []

def save_consolidated_log():
    with open("consolidated_log.csv", "w", encoding="utf-8") as file:
        for line in consolidated_log:
            file.write(line+"\n")

def update_consolidated_log(line):
    consolidated_log.append(line)

# --- Game Setup ---

# 1. Enumerations for Properties and Game Types
class Shape(Enum):
    CIRCLE = 0
    SQUARE = 1

class Height(Enum):
    TALL = 0
    SMALL = 1

class Color(Enum):
    DARK = 0
    LIGHT = 1

class Finish(Enum):
    FULL = 0
    HOLLOW = 1

class GameplayType(Enum):
    RANDOM = 0
    DETERMINISTIC = 1
    HEURISTIC = 2
    MONOMANIAC = 3
    OPENAI_PLAYER = 4

# 2. Piece Class Representation
class Piece:
    def __init__(self, shape, height, color, finish):
        self.shape = shape
        self.height = height
        self.color = color
        self.finish = finish
        self.placed = False

    def __repr__(self):
        my_repr = (
            "circular" if self.shape.value == 0 else "square",
            "tall" if self.height.value == 0 else "short",
            "black" if self.color.value == 0 else "white",
            "full" if self.finish.value == 0 else "hollow",

        )
        return f"Piece{my_repr}"

    def as_boolean_str(self):
        return f"({self.shape.value}, {self.height.value}, {self.color.value}, {self.finish.value})"

    def get_attributes(self):
        return (self.shape, self.height, self.color, self.finish)

    def __hash__(self):
        return hash((self.shape, self.height, self.color, self.finish))

# 3. Board Class Representation
class Board:
    def __init__(self):
        self.grid = [[None for _ in range(4)] for _ in range(4)]
        self.game = None

    def place_piece(self, x, y, piece):
        if self.grid[x][y] is None:
            self.grid[x][y] = piece
            piece.placed = True
            return True
        return False

    def get_empty_cells(self):
        return [(i, j) for i in range(4) for j in range(4) if self.grid[i][j] is None]

    def get_available_pieces(self):
        return self.game.available_pieces

    def check_winner(self):
        """Returns ``True`` if the board contains a winner, else returns ``False``"""
        for i in range(4):
            # Check rows
            if self._check_line([self.grid[i][j] for j in range(4)]):
                return True
            # Check columns
            if self._check_line([self.grid[j][i] for j in range(4)]):
                return True
        # Check diagonals
        if self._check_line([self.grid[i][i] for i in range(4)]) or self._check_line([self.grid[i][3 - i] for i in range(4)]):
            return True
        return False

    def _check_line(self, line):
        "Checks if a ``line`` of 4 pieces causes a player to win. Expects ``line`` to be a ``list`` with 4 ``Piece`` objects."
        if None in line:
            return False
        
        attributes = [piece.get_attributes() for piece in line]
        for i in range(4):  # Each attribute position (shape, height, color, finish)
            if all(attr[i] == attributes[0][i] for attr in attributes):
                return True
        return False

    def print_board(self):
        for row in self.grid:
            print(row)
        print()
        
    def get_board_as_matrix(self):
        return "[\n" + ",\n".join(f"row {num}: " + str(row) for num, row in enumerate(self.grid)) + "\n]"

class Player:
    """Interfaces with it's AI for piece movement and placement, each player has a 
    unique gameplay_type defining their strategy"""
    def __init__(self, gameplay_type, name):
        self.gameplay_type = gameplay_type
        self.name = name

    def select_piece(self, available_pieces, board, game_state, last_opponent_piece=None, game=None):
        if self.gameplay_type == GameplayType.RANDOM:
            return select_random_piece(available_pieces, board)
        elif self.gameplay_type == GameplayType.DETERMINISTIC:
            return select_deterministic_piece(available_pieces, board)
        elif self.gameplay_type == GameplayType.HEURISTIC:
            return select_heuristic_piece(available_pieces, board)
        elif self.gameplay_type == GameplayType.MONOMANIAC:
            return select_monomaniac_piece(available_pieces, board, last_opponent_piece, ) # Pass last_opponent_piece here
        elif self.gameplay_type == GameplayType.OPENAI_PLAYER:
            return select_AI_piece(game)
        
    def place_piece(self, board: Board, piece: Piece, game=None):
        if self.gameplay_type == GameplayType.RANDOM:
            return select_random_move(board)
        elif self.gameplay_type == GameplayType.HEURISTIC:
            return select_heuristic_move(board, piece)
        elif self.gameplay_type == GameplayType.DETERMINISTIC:
            return select_deterministic_move(board, piece)
        elif self.gameplay_type == GameplayType.MONOMANIAC:
            return select_monomaniac_move(board, piece)
        elif self.gameplay_type == GameplayType.OPENAI_PLAYER:
            return select_AI_move(game)
        return None

# --- Game Simulation ---

class QuartoGame:
    def __init__(self, player1: Player, player2: Player):
        self.board = Board()
        self.board.game = self
        self.player1 = player1
        self.player2 = player2
        self.available_pieces = [Piece(shape, height, color, finish) for shape, height, color, finish in product(Shape, Height, Color, Finish)]
        self.log = []
        self.board_log = []
        self.current_player = player1
        self.other_player = player2
        self.game_state = f"" # or f"Game started betwen {self.player1} and {self.player2}!"

    def append_log(self, action_type=None, piece_selected = None, outcome="In-play"):
        """
            Appends a move to be printed when the game is logged.
            action_types are {
            0: "Choose Piece",
            1: "Place Piece",
            2: "Game Ended"
        }
        """
        
        self.game_state = self.get_game_state(action_type, piece_selected, outcome)
        self.log.append(self.game_state)
        update_consolidated_log(self.game_state)

    def get_game_state(self, action_type, piece_selected, outcome = "In-play") -> str:
        """Return a string depicting the current game state (based on the last move)
        action_types are {
            0: "Choose Piece",
            1: "Place Piece",
            2: "Game Ended"
        }"""
        action_types = {
            0: "Choose Piece",
            1: "Place Piece",
            2: "Game Ended"
        }
        return  f"""
    Turn Number: {16 - len(self.available_pieces)},
    Action Type: {action_types[action_type]},
    Chooser: {self.current_player.gameplay_type.name},
    Placer: {self.other_player.gameplay_type.name},
    Piece Chosen: {piece_selected},
    Board State: {self.board.get_board_as_matrix()},
    Game Outcome: {outcome}"""

    def log_board(self, comment=""):
        with open("quarto_boards.log", "a", encoding="utf-8") as f:
            if comment:
                self.board_log.append(comment)
            else:
                grid_as_matrix = self.board.get_board_as_matrix()
                self.board_log.append(grid_as_matrix)

    def play_game(self):
        last_opponent_piece = None  # Initialize last_opponent_piece
        self.log.append(f"Starting new game between {self.player1.name} and {self.player2.name}")
        while self.available_pieces:

            # If every available piece results in a win for the opponent then current player loses
            # (If currrent player cannot pick a piece without losing then he loses)
            if all(self.creates_win_for_opponent(piece) for piece in self.available_pieces):
                # The current player cannot choose a piece without immediately losing.
                # ``selected_piece`` and ``position`` are undefined in this branch, so
                # log the outcome without referencing them.
                self.append_log(
                    action_type=2,
                    piece_selected=None,
                    outcome=(
                        f"{self.other_player.name} won the game cause {self.current_player.name} "
                        "can't pick a piece without losing!"
                    ),
                )
                self.log.append((self.other_player.name, "WIN", None, None))
                self.log_board(comment=f"{self.other_player.name} won the game!")

                return self.other_player  # Return the winner


            position="Not placed yet"
            selected_piece = self.current_player.select_piece(self.available_pieces, self.board, self.game_state, last_opponent_piece, self)
            self.available_pieces.remove(selected_piece)

            self.append_log(action_type = 0, piece_selected = selected_piece)
            self.log.append((self.current_player.name, "Select", selected_piece, position))
            self.log_board()

            last_opponent_piece = selected_piece
            # Choose a position and place the piece
            position = self.other_player.place_piece(self.board, last_opponent_piece, game=self)
            x,y = position
            self.board.grid[x][y] = selected_piece

            self.append_log(action_type = 1, piece_selected = selected_piece)
            self.log.append((self.other_player.name, "Place", selected_piece, position))
            self.log_board()


            if self.board.check_winner():  # Check if other_player placed the winning piece
                self.append_log(action_type = 2, piece_selected = selected_piece, outcome=f"{self.other_player.name} won the game cause he placed the winning piece!")
                self.log.append((self.other_player.name, "WIN", selected_piece, position))
                self.log_board(comment=f"{self.other_player.name} won the game!")
                return self.other_player

            # Swap current and other player
            self.current_player, self.other_player = self.other_player, self.current_player

        # If no winner, it's a tie
        self.append_log(action_type = 2, piece_selected = selected_piece)
        self.log.append(("-", "Tie", selected_piece, position))  # No more pieces available, tie
        self.log_board("The game ended in a tie.")
        return "tie"

    
    def creates_win_for_opponent(self, piece):
        # Simulate if placing the piece leads to a guaranteed win for the opponent
        for position in self.board.get_empty_cells():
            self.board.place_piece(position[0], position[1], piece)
            if self.board.check_winner():
                self.board.grid[position[0]][position[1]] = None  # Remove the piece
                return True
            self.board.grid[position[0]][position[1]] = None
        return False


# --- Player Logic ---

def select_random_piece(available_pieces, board):
    safe_pieces = [piece for piece in available_pieces if not would_give_win(piece, board)]
    return random.choice(safe_pieces) if safe_pieces else random.choice(available_pieces)

def select_heuristic_piece(available_pieces, board):
    # Block opponent's winning move first
    for piece in available_pieces:
        if would_give_win(piece, board):
            return piece  # Avoid giving the opponent a winning piece

    # Prioritize double threats if no immediate danger
    max_threat_count = -1
    best_piece = None
    for piece in available_pieces:
        threat_count = count_double_threats(piece, board)
        if threat_count > max_threat_count:
            max_threat_count = threat_count
            best_piece = piece

    return best_piece if best_piece else random.choice(available_pieces)


def select_deterministic_piece(available_pieces, board):
    non_winning_pieces = [piece for piece in available_pieces if not would_give_win(piece, board)]
    if non_winning_pieces:
        # Choose the most diverse piece from those that don't give a win
        return max(non_winning_pieces, key=lambda p: calculate_diversity_score(p, board))
    # If all pieces are winning pieces, fallback to diversity
    return max(available_pieces, key=lambda p: calculate_diversity_score(p, board))


def select_monomaniac_piece(available_pieces, board, last_opponent_piece=None, target_dimension=None):
    if target_dimension is None:
        target_dimension = random.randint(0, 3)

    # Avoid giving the opponent a winning piece
    for piece in available_pieces:
        if would_give_win(piece, board):
            continue  # Skip this piece if it enables an opponent's win

    # Focus on the target dimension
    if last_opponent_piece is not None:
        matching_pieces = [
            piece for piece in available_pieces
            if piece.get_attributes()[target_dimension] == last_opponent_piece.get_attributes()[target_dimension]
        ]
    else:
        matching_pieces = []

    # Fallback to random selection if no matching pieces found
    return random.choice(matching_pieces) if matching_pieces else random.choice(available_pieces)

def select_AI_piece(quarto_game: QuartoGame) -> Piece:
    piece_selected = openai_player.select_AI_piece(quarto_game)
    for piece in quarto_game.available_pieces:
        if piece.as_boolean_str() == str(piece_selected):
            return piece
        
    raise AssertionError("GPT returned a piece that's not valid and it wasn't correctly handled.")
    

def would_give_win(piece, board):
    for i in range(4):
        for j in range(4):
            if board.grid[i][j] is None:
                board.grid[i][j] = piece  # Temporarily place the piece
                if board.check_winner():
                    board.grid[i][j] = None  # Reset the cell
                    return True
                board.grid[i][j] = None  # Reset the cell
    return False


def select_random_move(board):
    empty_cells = board.get_empty_cells()
    return random.choice(empty_cells) if empty_cells else None

def select_heuristic_move(board, piece):
    best_move = None
    max_threat_count = -1

    empty_cells = board.get_empty_cells()

    # First, check for a winning move
    for i, j in empty_cells:
        board.grid[i][j] = piece
        if board.check_winner():
            board.grid[i][j] = None
            return (i, j)  # Make the winning move
        board.grid[i][j] = None

    # Second, block opponent's winning move
    for opponent_piece in board.get_available_pieces():
        for i, j in empty_cells:
            board.grid[i][j] = opponent_piece
            if board.check_winner():
                board.grid[i][j] = None
                return (i, j)  # Block opponent's win
            board.grid[i][j] = None

    # Fallback to creating double threats
    for i, j in empty_cells:
        board.grid[i][j] = piece
        threat_count = count_double_threats(piece, board)
        if threat_count > max_threat_count:
            max_threat_count = threat_count
            best_move = (i, j)
        board.grid[i][j] = None  # Reset the board cell after evaluation

    return best_move if best_move else select_random_move(board)

# Helper to count double threats
def count_double_threats(piece, board):
    double_threat_count = 0
    for i in range(4):
        for j in range(4):
            if board.grid[i][j] is None:
                board.grid[i][j] = piece
                threat_lines = [
                    board.grid[i],                        # Row
                    [row[j] for row in board.grid]        # Column
                ]
                if i == j:
                    threat_lines.append([board.grid[k][k] for k in range(4)])
                if i + j == 3:
                    threat_lines.append([board.grid[k][3 - k] for k in range(4)])

                for line in threat_lines:
                    similar_pieces = sum(1 for p in line if p == piece)
                    empty_spaces = line.count(None)
                    if similar_pieces == 2 and empty_spaces == 2:
                        double_threat_count += 1
                board.grid[i][j] = None  # Reset the cell after checking

    return double_threat_count

def select_deterministic_move(board, piece):
    empty_cells = board.get_empty_cells()

    # Check for winning move
    for i, j in empty_cells:
        board.grid[i][j] = piece
        if board.check_winner():
            board.grid[i][j] = None
            return (i, j)  # Play winning move
        board.grid[i][j] = None

    # Check for blocking move
    for opponent_piece in board.get_available_pieces():
        for i, j in empty_cells:
            board.grid[i][j] = opponent_piece
            if board.check_winner():
                board.grid[i][j] = None
                return (i, j)  # Block opponent win
            board.grid[i][j] = None

    # Fallback to diversity-based move
    return max(empty_cells, key=lambda cell: calculate_diversity_score(piece, board))

def calculate_diversity_score(piece, board):
    diversity_score = 0
    for i in range(4):
        for j in range(4):
            if board.grid[i][j] is not None:
                placed_piece = board.grid[i][j]
                differing_attributes = sum(
                    1 for attr_piece, attr_placed in zip(piece.get_attributes(), placed_piece.get_attributes())
                    if attr_piece != attr_placed
                )
                diversity_score += differing_attributes
    return diversity_score

def select_monomaniac_move(board, piece, preferred_dimension=0):
    preferred_moves = []
    adjacent_moves_with_common_dimension = []

    # Check for a winning move
    empty_cells = board.get_empty_cells()
    for i, j in empty_cells:
        board.grid[i][j] = piece
        if board.check_winner():
            board.grid[i][j] = None
            return (i, j)  # Make the winning move
        board.grid[i][j] = None

    # Block opponent's winning move
    for opponent_piece in board.get_available_pieces():
        for i, j in empty_cells:
            board.grid[i][j] = opponent_piece
            if board.check_winner():
                board.grid[i][j] = None
                return (i, j)  # Block opponent's win
            board.grid[i][j] = None

    # Preferred moves along the specified dimension
    for i in range(4):
        if preferred_dimension == "diagonal":
            cell = (i, i)
        elif preferred_dimension == "anti-diagonal":
            cell = (i, 3 - i)
        elif isinstance(preferred_dimension, int) and preferred_dimension < 4:
            cell = (preferred_dimension, i)
        elif isinstance(preferred_dimension, int) and preferred_dimension >= 4:
            cell = (i, preferred_dimension - 4)
        else:
            raise ValueError("Invalid preferred_dimension value. Must be 'diagonal', 'anti-diagonal', or an integer.")

        if board.grid[cell[0]][cell[1]] is None:
            preferred_moves.append(cell)

    # Evaluate adjacency for potential moves
    for i, j in empty_cells:
        adjacent_cells = get_adjacent_cells(i, j)
        for x, y in adjacent_cells:
            if board.grid[x][y] is not None and count_common_dimensions(piece, board.grid[x][y]) > 0:
                adjacent_moves_with_common_dimension.append((i, j))
                break

    # Prioritize adjacency or preferred moves
    if adjacent_moves_with_common_dimension:
        return random.choice(adjacent_moves_with_common_dimension)
    elif preferred_moves:
        return random.choice(preferred_moves)
    else:
        return select_random_move(board)


def get_adjacent_cells(i, j):
    adjacent_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    adjacent_cells = [(i + di, j + dj) for di, dj in adjacent_offsets if 0 <= i + di < 4 and 0 <= j + dj < 4]
    return adjacent_cells

def calculate_common_dimensions(piece, board, adjacent_cells):
    return sum(1 for x, y in adjacent_cells if board.grid[x][y] and count_common_dimensions(piece, board.grid[x][y]) > 0)

def adjust_diversity_score(diversity_score, has_neighbors, common_dimensions):
    if has_neighbors:
        return diversity_score - common_dimensions
    else:
        return diversity_score + 5

def count_common_dimensions(piece1, piece2):
    return sum(1 for dim in range(4) if piece1.get_attributes()[dim] == piece2.get_attributes()[dim])

def select_AI_move(quarto_game: QuartoGame) -> tuple[int]:
    position = openai_player.select_AI_move(quarto_game)
    return position


# --- Simulation of Multiple Games ---

def simulate_games(num_games_per_combination):
    gameplay_types = [GameplayType.RANDOM, GameplayType.DETERMINISTIC, GameplayType.HEURISTIC, GameplayType.MONOMANIAC, GameplayType.OPENAI_PLAYER]
    results = {}
    log_data = []
    board_states_log = []
    
    for p1_type, p2_type in product(gameplay_types, repeat=2):
        p1_wins, p2_wins, ties = 0, 0, 0
        for game_counter in range(1, num_games_per_combination+1):
            print(f"playing game {game_counter} for {p1_type} vs {p2_type}")
            player1 = Player(p1_type, f"Player 1 ({p1_type})".replace("GameplayType.", ""))
            player2 = Player(p2_type, f"Player 2 ({p2_type})".replace("GameplayType.", ""))
            game = QuartoGame(player1, player2)
            result = game.play_game()

            log_data.extend(game.log)
            board_states_log.extend(game.board_log)
            update_consolidated_log(game.game_state)
            save_consolidated_log()

            if result == player1:
                p1_wins += 1
            elif result == player2:
                p2_wins += 1
            elif result == "tie":
                ties += 1
            else: quit()


        results[(p1_type, p2_type)] = (p1_wins, p2_wins, ties)
        

    # log_df = pd.DataFrame(log_data, columns=["Player1", "Player2", "Selected Piece", "Position/Winner"])
    # log_df["Selected Piece"] = log_df["Selected Piece"].astype(str)
    # log_df.to_csv('quarto_game_log.csv', index=False)

    with open("quarto_boards.log", "w", encoding="utf-8") as file:
        file.write("\n".join(board_states_log))

    # Persist any remaining consolidated log entries to disk.
    save_consolidated_log()

    for key, value in results.items():
        print(f"{key[0].name} vs {key[1].name}: Player 1 wins {value[0]}, Player 2 wins {value[1]}, Ties {value[2]}")

def main():
    with open("quarto_boards.log", "w", encoding="utf-8") as f:
        f.write("Board states across simulations\n")
    num_games = 1
    simulate_games(num_games)
    


if __name__ == "__main__":
    main()