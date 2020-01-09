import sys
from odf.opendocument import load 

infile = sys.argv[1]

doc = load(infile)
print(doc.body())
