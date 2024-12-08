# Quarto Game Simulation

This repository hosts a simulation framework for the game **Quarto**, exploring the gameplay of various strategies, including **random play**, **deterministic strategies**, **heuristics**, and **GPT-driven gameplay**. The framework aims to test how different strategies perform against each other, with particular attention to the capabilities of GPT models as a player.

---

## Features

- **Multiple Gameplay Strategies**:
  All gameplay strategies implement a check to avoid obivous mistakes of selecting a piece that would give an immediate win to the opponent. We develop a set of 'safe pieces' by (1) selecting each of available piece and (2) positioning the selected available piece in open slots and (3) confirm that the selected available piece would not give an immediate win 
  - **Random**: Completely random moves.
  - **Deterministic**: Predictable logic-based decisions.
  - **Heuristic**: Decision-making based on game state analysis.
  - **Monomaniac**: Focuses on a single dimension during gameplay.
  - **GPT-driven**: OpenAI-powered player implementing adaptive gameplay.
  
- **Game Mechanics**:
  - Fully simulated board and gameplay mechanics for Quarto.
  - Automatic validation of moves to ensure rule compliance.
  - Winner determination based on Quarto's unique alignment rules.

- **Logging**:
  - Detailed game logs for each move, including board state, selected pieces, and placements.
  - Board state snapshots for post-game analysis.
  - Consolidated logs across multiple game simulations.

---

## Getting Started

### Prerequisites
- Python 3.8+
- Required libraries:
  - `openai`
  - `pydantic`
  - `python-dotenv`

Install dependencies:
```bash
pip install openai pydantic python-dotenv
Setup
Clone this repository:

bash
Copier le code
git clone https://github.com/yourusername/quarto-simulation.git
cd quarto-simulation
Set up your OpenAI API key:

Create a .env file in the root directory.
Add your API key:
makefile
Copier le code
OPENAI_API_KEY=your_api_key_here
Run the simulation:

bash
Copier le code
python main.py
Configuration
You can enable debugging by passing the --debug flag when running the script:

bash
Copier le code
python main.py --debug
File Overview
Core Files
main.py: Main driver for the simulation. Manages game initialization, logging, and result aggregation.
openai_player.py: Contains GPT-powered decision-making logic for selecting pieces and moves.
quarto_game.py: Implements the game mechanics, board state management, and winner determination.
Logs
quarto_boards.log: Records snapshots of board states after each move.
consolidated_log.csv: Logs all moves, board states, and outcomes for analysis.
How It Works
Simulation:

The simulation pits all gameplay strategies against each other in pairwise matches.
For each game:
Players alternate selecting pieces and placing them on the board.
Moves are validated to ensure compliance with game rules.
Results:

Each match is logged, detailing player actions and final outcomes.
Results are aggregated across multiple games to evaluate strategy performance.
GPT Player:

Utilizes OpenAI's GPT model to simulate intelligent gameplay.
Prompts are structured to guide the model in choosing optimal moves and pieces.
Running Simulations
To simulate multiple games between all strategy combinations, update the num_games variable in the main() function of main.py:

python
Copier le code
num_games = 10
Then run:

bash
Copier le code
python main.py
Results will be printed to the console and logged in the output files.

Contributing
Contributions are welcome! To add new features or gameplay types:

Fork the repository.
Create a new branch for your feature.
Submit a pull request with a clear description of your changes.
License
This project is licensed under the MIT License. See the LICENSE file for details.
