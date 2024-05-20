import pandas as pd, pickle
from decisionTreeClassifier import DecisionTreeClassifier
from board import Board
from preProcess import PreprocessData
from statistic import Statistics
from board import Board
from connect4 import *

data = pd.read_csv('datasets\connect4.csv')


indice = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 
 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 
 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 
 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 
 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 
 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 
 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'Class']

data.columns = indice

connect_df = data

'''
     a b c d e f g 
    6
    5
    4
    3
    2
    1
'''

def to_board(row: list):
    b = Board()
    dic = {'x': 'X', 'o': 'O', 'b': '-'}
    for i, player in enumerate(row):
        col = i // 6
        line = 5 - (i % 6)
        b.setPos(line, col, dic.get(player))
    l = [pos for sublist in b.board for pos in sublist]
    if l.count('O') >= l.count('X'): b.player = 'O'
    else: b.player = 'X'
    return b
        
row = to_board(connect_df.iloc[1].tolist()[:-1])
print(row)

#models
with open('variables/models.pkl', 'rb') as f:
    p, dt = pickle.load(f)

# dt.print_tree()

# stats = Statistics()
# stats.evaluate_once(tree= dt, process= p)

def dt_decision(dt: DecisionTreeClassifier, board: Board) -> Board:
    indexes = dt.original_dataset.columns[:-1]
    
    #lista de todos os possiveis proximos moves ja em forma de lista
    next_moves = []
    for line, col in possibleMoves(board= board):
        copy = board.boardCopy()
        copy.setPos(line, col)
        next_moves.append(copy.to_csv_by_columns())
        
    #df com os proximos moves
    X = pd.DataFrame(data= next_moves, columns= indexes)
    predicts = dt.predict(X)
    
    #saber qual o index dos moves com wins ou draws
    win_indexes = []
    draw_indexes = []
    for idx, result in enumerate(predicts):
        if result == 'win':
            win_indexes.append(idx)
        elif result == 'draw':
            draw_indexes.append(idx)
    print(predicts)
    print(win_indexes)
    print(draw_indexes)
    
    #preferencia win > draw > loss
    if len(win_indexes) > 0:
        idx = random.choice(win_indexes)
    elif len(draw_indexes) > 0:
        idx = random.choice(draw_indexes)
    else: idx = 0
    return to_board( next_moves[idx] )

def game(board: Board, order: list):
    print(board)
    while True:
        print('Tua vez.')
        askForNextMove(board)
        print(board)
        
        if winnerAi(board, order):
            return 
        
        #logica da arvore (?)
        board = dt_decision(dt, board)
        # print('A IA pôs uma peça na coluna ' + str(col) + '.')
        print(board)
        
        if winnerAi(board, order):
            return

def main():
    play = True
    while play:
        board = Board('O')
        board.resetBoard()
        game(board= board, order= ['O', 'X'])
        play = playAgain()

if __name__ == '__main__':
    main()