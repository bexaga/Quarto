import json
from typing import List
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import sys

import config

load_dotenv(".env")
DEBUG = "--debug" in sys.argv
client = OpenAI()

move_prediction_prompt = """You are playing Quarto. Each turn, you will receive the current game state. Your task is to either:

1. Choose a position [row, column] (row and column are integers 0–3) to place a given piece.
2. Choose a piece for your opponent to place.

Pieces are described as ['circular'/'square', 'tall'/'short', 'black'/'white', 'full'/'hollow']. For example, ['square', 'tall', 'black', 'hollow'] represents a square tall black hollow piece."""

smartmove_prediction_prompt = "You are playing the game of Quarto. You will be provided the current game state resulting from the last turn played and you will have to either choose a position [row, column], where row and column are integers between 0 and 3, or choose a piece for your opponent to place. Pieces are represented by their attributes in the following format: ('circular'/'square', 'tall'/'short', 'black'/'white', 'full'/'hollow') ['square', 'tall', 'black', 'hollow'] is a `square tall black hollow` piece."

class Prediction():
    prompt = move_prediction_prompt
class SmartPrediction():
    prompt = smartmove_prediction_prompt



class MovePrediction(BaseModel, Prediction):
    position: list[int] 

class PiecePrediction(BaseModel, Prediction):
    piece: list[str]
    
# TODO
class SmartMovePrediction(BaseModel, SmartPrediction):
    reasoning: str
    position: List[int]

class SmartMovePrediction(BaseModel, SmartPrediction):
    reasoning: str
    piece: List[str]

def select_AI_piece(game_obj, model = PiecePrediction) -> tuple[int]:
    user_prompt = f"""
    Current Game State (last move) - Board is represented row by row:
        {game_obj.game_state}
    You just placed a piece.

    Task: 
        - Select a new piece for your opponent to place that maximizes YOUR chances of winning. 
        - Ensure the piece is not already on the board and that you are not giving them a winning piece. 
        - Use the format [shape, height, color, finish].
    """
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": model.prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=model,
    )

    event = completion.choices[0].message.content
    parsed_event = json.loads(event)
    if DEBUG:
        print(user_prompt)
        print(event)
        print(completion.__dict__)
    piece_as_str = tuple(parsed_event["piece"])

    piece_invalid = False
    piece = piece_as_str
    try:
        # Ensure piece is valid
        assert "circular" in piece or "circle" in piece or "square" in piece, f"Piece must be circular/circle, or square, you chose {piece[0]}"
        assert "tall" in piece or "short" in piece, f"Piece must be tall or short, you chose {piece[1]}"
        assert "black" in piece or "white" in piece, f"Piece must be black or white, you chose {piece[2]}"
        assert "full" in piece or "hollow" in piece, f"Piece must be full or hollow, you chose {piece[3]}"
    except AssertionError as e:
        error_message = str(e)
        piece_invalid = True

    if not piece_invalid:
        piece = (
            0 if piece[0] in ["circular", "circle"] else 1,
            0 if piece[1] == "tall" else 1,
            0 if piece[2] == "black" else 1,
            0 if piece[3] == "full" else 1,
        )
        # Validate GPT output
        try:
            assert str(piece) in [available_piece.as_boolean_str() for available_piece in game_obj.available_pieces], f"AI chose {piece_as_str} but the available pieces are {[piece for piece in game_obj.available_pieces]}"
        except AssertionError as e:
            error_message = str(e)
            piece_invalid = True

    illegal_moves = 1
    while piece_invalid and illegal_moves < config.MAX_AI_ILLEGAL_MOVES:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": model.prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": json.dumps(parsed_event)},
                {"role": "user", "content": error_message}
            ],
            response_format=model,
        )

        event = completion.choices[0].message.content
        parsed_event = json.loads(event)
        if DEBUG:
            print(user_prompt)
            print(error_message)
            print(event)
            print(completion.__dict__)
        piece_as_str = tuple(parsed_event["piece"])

        piece = piece_as_str
        try:
            # Ensure piece is valid
            assert "circular" in piece or "circle" in piece or "square" in piece, f"Piece must be circular/circle, or square, you chose {piece[0]}"
            assert "tall" in piece or "short" in piece, f"Piece must be tall or short, you chose {piece[1]}"
            assert "black" in piece or "white" in piece, f"Piece must be black or white, you chose {piece[2]}"
            assert "full" in piece or "hollow" in piece, f"Piece must be full or hollow, you chose {piece[3]}"
            piece_invalid = False
        except AssertionError as e:
            error_message = str(e)
            piece_invalid = True
            illegal_moves += 1

            
        if not piece_invalid:
            piece = (
                0 if piece[0] in ["circular", "circle"] else 1,
                0 if piece[1] == "tall" else 1,
                0 if piece[2] == "black" else 1,
                0 if piece[3] == "full" else 1,
            )
            # Validate GPT output
            try:
                print(f"ERROR LOG: looking for {piece_as_str} in {[available_piece.as_boolean_str() for available_piece in game_obj.available_pieces]}")
                assert str(piece) in [available_piece.as_boolean_str() for available_piece in game_obj.available_pieces], f"AI chose {piece_as_str} but the available pieces are {[piece for piece in game_obj.available_pieces]}"
            except AssertionError as e:
                error_message = str(e)
                piece_invalid = True
                illegal_moves += 1
    return piece

def select_AI_move(game_obj, model = MovePrediction) -> tuple[int]:
    """
        Return a tuple in format (row, column) of where AI would place the selected piece.
    """

    user_prompt = f"""
    Current Game State (last move):
        {game_obj.game_state}

    Task:
    - Select the best valid **unoccupied position** on the board to place the piece: Piece('circular', 'short', 'black', 'full').
    - The position must be in the format [row, column].
    - A valid move places the piece in an empty (None) cell.
    """
    # """
    # Goal:
    #     - Maximize your chances of winning by choosing a position that forms or blocks a winning line.
    # """
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": model.prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=model,
    )
    
    # Get GPT output
    event = completion.choices[0].message.content
    parsed_event = json.loads(event)
    if DEBUG:
        print(user_prompt)
        print(event)
        print(completion.__dict__)
    position = tuple(parsed_event["position"])
        

    # Validate GPT output, ask for a valid position until it returns one, stops at config.MAX_AI_ILLEGAL_MOVES
    for cell in game_obj.board.get_empty_cells():
        if cell == position:
            return cell

        output_invalid = True
    illegal_moves = 1
    while output_invalid and illegal_moves < config.MAX_AI_ILLEGAL_MOVES:
        log_gpt = f"GPT chose {position} but the available positions are {game_obj.board.get_empty_cells()}"
        print(log_gpt)
        
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": model.prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": json.dumps(parsed_event)},
                {"role": "user", "content": log_gpt}
            ],
            response_format=model,
        )
        # Get new GPT output
        event = completion.choices[0].message.content
        parsed_event = json.loads(event)

        if DEBUG:
            print(user_prompt)
            print(log_gpt)
            print(event)
            print(completion.__dict__)
        position = tuple(parsed_event["position"])

        # Validate GPT output, ask for a valid position until it returns one, stops at config.MAX_AI_ILLEGAL_MOVES
        for cell in game_obj.board.get_empty_cells():
            if cell == position:
                return cell
        # If we haven't returned, output is still invalid
        output_invalid = True
        illegal_moves += 1
    raise Exception(f"GPT made {config.MAX_AI_ILLEGAL_MOVES} illegal moves in a row.")