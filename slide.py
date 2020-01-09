import sys
from odf.opendocument import load 

infile = sys.argv[1]

doc = load(infile)

for key in doc.element_dict:
    for x in key:
        print(x)
    for elm in doc.element_dict[key]:
        print(elm)

    print("-------------------")
