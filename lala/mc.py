from sklearn.metrics import confusion_matrix

f = open("val.txt","r")

true = []
predict = []

for i in range(0,10000):
  l = f.readline()
  pos = l.find("Label:")
  t = l[pos+7:pos+8]
  p = l[pos+16:pos+17]
  true.append(t)
  predict.append(p)

pos = f.readline()

print(confusion_matrix(true,predict))
print(pos)
