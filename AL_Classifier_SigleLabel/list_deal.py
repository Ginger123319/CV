import numpy

l1 = ["air", "water", "sunshine"]
l1 = numpy.array(l1)
l1 = numpy.append(l1, "army")
l1.sort()
print(l1)
l2 = [True, False, True, True]
print(l1[l2])
