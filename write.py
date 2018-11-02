import sys

l = [i for i in range(10)]

f = open(sys.argv[1], 'w')
for item in l:
  f.write(str(item) + "\n")
f.close()
