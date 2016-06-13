

import apriori as ap

dataSet = ap.loadDataSet()
#print dataSet
C1 = ap.createC1(dataSet)
#print C1
D = map(set, dataSet)
#print D
L1, suppData0 = ap.scanD(D, C1, 0.5)
#print suppData0
L, S = ap.apriori(D, 0.5)
#print L

print L

List = ap.generateRules(L, S, minConf=0.4)
print List


