import mnist_loader
t = mnist_loader.load_data()
#printing number of test cases
print len(t[2][0])
for x in t[2][0]:
	for p in x:
		print p,
	print ""

for x in t[2][1]:
	print "0 "*(x) + "1 " + "0 "*(9-x)