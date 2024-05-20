import random
from board import Board

def inputPlayer(letter:str) -> list: #pode ser removido, basta ver o pq de ser usado no min.py
    #retorna lista com a ordem do jogador
    if letter == 'X':
        return ['X', 'O']
    else:
        return ['O', 'X']

def askForFirstPlayer() -> list:
    while True:
        piece = input("Queres ser X, O or R? ")
        if piece.upper() == "X":
            return ['X', 'O']
        elif piece.upper() == "O":
            return ['O', 'X']
        elif piece.upper() == "R":
            m = random.randint(0, 1)
            return ['X', 'O'] if m == 0 else ['O', 'X']

def askForAlgorithm() -> int:
    while True:
        try:
            algorithm = int(input("Qual algoritmo queres? 1 - P vs P, 2 - A*, 3 - MC, 4 - Minimax : "))
        except ValueError:
            print("Tente inteiros. \n")
            continue

        if algorithm in range(1, 5, 1):
            return algorithm
        else:
            print('Tente um número de 1 a 4. \n')
        
def playAgain() -> bool:
    while True:
        play = input(("Queres jogar de novo? S ou N. : "))
        if play.upper() == 'S':
            return True
        elif play.upper() == 'N':
            return False
        else:
            print('Tente S ou N. \n')

def testMove(board:Board , col:int) -> int:
    #-1 se não é possível
    for i in range(6):
        if board.getPos(5-i, col) == "-":
            return 5 - i
    return -1

def possibleMoves(board:Board) -> list:
    acc = []
    for col in range(7):
        line = testMove(board, col)
        if line != -1:
            acc.append((line, col))
    return acc

def askForNextMove(board:Board) -> tuple:
    while True:
        try:
            col = int(input("Em qual coluna? "))
        except ValueError:
            print("Tente inteiros.")
            continue
        
        if col > 6 or col < 0:
            print("Fora das margens.")
        else:
            line = testMove(board, col)
            
            if line == -1:
                print("Coluna cheia.")
            else:
                board.setPos(line, col)
                return (line, col)

def winnerAi(board:Board , order:list) -> bool: 
    win = board.finished()
    if isinstance(win, str):
        if win == 'Tie':
            print('Empate.')
        elif win == order[0]:
            print('Ganhaste.')
        else:
            print('A IA ganhou.')
        return True
    return False
