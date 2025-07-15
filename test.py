# import random
# a = [[1.2,[34,2.4]],[1.23,3.2],[4.55,4.6],[[2.3,2.5],1.4],[1,1,1],[1,2,3],[4,5,6],[6,[7,7],9],[2],[3]]
# b = a[:6]
# random.shuffle(b)
# a[:6] = b
# print(a)

# import matplotlib.pyplot as plt
# plt.figure()
# a = [1,2,3,4,5,6,7,8]
# b = [2,3,4,5,6,7,8,9]
# x = range(len(a))
# plt.xlabel("as")
# plt.title("error of training and validation")
# plt.plot(x,a,"r",label = "train")
# plt.plot(x,b,"b",label = "valid")
# plt.legend()
# plt.show()

# a = range(10)
# for i in a:
#     print(i)

# import os
# path = 'trajData/Geolife_out'
# total_len = len(os.listdir(path))
# print(total_len)

# def a(b):
#     b += 1
#     return b

# def aa(b, bb):
#     if bb == 'a':
#         f = a()
#     print(f(b))

# aa(2,'a')
# a = [1,3,4,5.79,6.2]
# with open('example.txt', 'a') as f:
#     for i in a:
#         f.write("\naa"+str(i))

import matplotlib.pyplot as plt
fig1 = plt.figure()

a = [48.74, 48.79, 47.96, 49.14, 46.50, 
     49.95, 47.96, 48.03, 48.09, 51.33, 
     46.26, 46.41, 45.98, 48.09, 46.99, 
     50.98, 47.09, 46.06, 47.23, 47.12, 
     50.64, 47.66, 48.83, 46.68, 47.57, 
     45.18, 47.38, 48.44, 49.54, 46.65, 
     45.66, 47.70, 46.68, 48.52, 46.58, 
     47.98, 44.78, 47.07, 45.52, 45.13, 
     45.22, 47.65, 45.19, 46.02, 45.78, 
     46.61, 46.30, 46.63, 47.22, 52.61]
x = range(len(a))
plt.plot(x,a)
b = [2,3,4,5,6,7,8,9]
xx = range(len(b))
plt.legend()

fig2 = plt.figure()
with open("online-rlts/errors_records.txt") as f:
    lines = f.readlines()
datas = lines[5].strip().split()
print(datas)
for i in range(len(datas)):
    datas[i] = float(datas[i])
print(datas)
xx = range(len(datas))
plt.plot(xx,datas)
plt.legend()
plt.show()