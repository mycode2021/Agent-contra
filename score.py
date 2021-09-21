import re, sys


with open(sys.argv[1]) as f:
    readlines = f.read()
    print("Max score: %.2f"%max(
        [-10]+list(
            map(float, re.findall("Score: (.*?),", readlines))
        )
    ))
    print("Max step: %d"%max(
        [0]+list(
            map(int, re.findall("Step: (.*?),", readlines))
        )
    ))
    print("Finish: %d"%sum(
        [0]+list(
            map(bool, re.findall("Evaluation: (success?).", readlines))
        )
    ))
