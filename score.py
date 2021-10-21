import re, sys


with open(sys.argv[1]) as f:
    readlines = f.read()
<<<<<<< HEAD
    print("Scroll: %d"%max(
        [0]+list(
            map(float, re.findall("Xscroll: (.*?),", readlines))
=======
    print("Max score: %.2f"%max(
        [-10]+list(
            map(float, re.findall("Score: (.*?),", readlines))
>>>>>>> origin/master
        )
    ))
    print("Finish: %d"%sum(
        [0]+list(
<<<<<<< HEAD
            map(bool, re.findall("Finish: (True?).", readlines))
=======
            map(bool, re.findall("Evaluation: (success?).", readlines))
>>>>>>> origin/master
        )
    ))
