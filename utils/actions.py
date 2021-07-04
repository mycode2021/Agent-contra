SIMPLE_MOVEMENT = [
    [],
    ['UP', 'B'],
    ['DOWN', 'B'],
    ['LEFT', 'B'],
    ['RIGHT', 'B'],
    ['A', 'B']
]

COMPLEX_MOVEMENT = [
    [],
    ['UP', 'B'],
    ['DOWN', 'B'],
    ['LEFT', 'B'],
    ['LEFT', 'UP', 'B'],
    ['LEFT', 'DOWN', 'B'],
    ['LEFT', 'A', 'B'],
    ['RIGHT', 'B'],
    ['RIGHT', 'UP', 'B'],
    ['RIGHT', 'DOWN', 'B'],
    ['RIGHT', 'A', 'B'],
    ['A', 'B']
]

Actions = {
    "simple": SIMPLE_MOVEMENT,
    "complex": COMPLEX_MOVEMENT
}
