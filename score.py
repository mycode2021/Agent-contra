import re, sys


with open(sys.argv[1]) as f:
    readlines = f.read()
    print("Scroll: %d"%max(
        [0]+list(
            map(float, re.findall("Xscroll: (.*?),", readlines))
        )
    ))
    print("Finish: %d"%sum(
        [0]+list(
            map(bool, re.findall("Finish: (True?).", readlines))
        )
    ))
