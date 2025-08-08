import os
import types
import sys

# Ensure the project root is on sys.path so ``main`` can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Stub out openai_player before importing main to avoid external dependencies
openai_stub = types.ModuleType("openai_player")
openai_stub.select_AI_piece = lambda game: None
openai_stub.select_AI_move = lambda game: (0, 0)
sys.modules["openai_player"] = openai_stub

# Stub pandas since ``main`` imports it but tests don't rely on it
pandas_stub = types.ModuleType("pandas")
sys.modules["pandas"] = pandas_stub

import main


def test_game_ends_when_no_safe_piece():
    player1 = main.Player(main.GameplayType.RANDOM, "Player 1")
    player2 = main.Player(main.GameplayType.RANDOM, "Player 2")
    game = main.QuartoGame(player1, player2)
    # Force scenario where any available piece would give a win to the opponent
    game.creates_win_for_opponent = lambda piece: True
    result = game.play_game()
    assert result == player2
