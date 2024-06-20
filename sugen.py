import time
from sudoku import Sudoku

def generate_sudoku():
    puzzle = Sudoku(3).solve()
    return puzzle

lis = [10, 25, 50, 75, 100]

start_time = time.time()
for i in lis:
    for _ in range(i):
        puzzle = generate_sudoku()
    end_time = time.time()
    print(f"Generated {i} Sudoku in {end_time - start_time} seconds.")