import matplotlib.pyplot as plt
import xlrd

# 数据散点图分析
iris = xlrd.open_workbook('iris.xlsx')
table = iris.sheets()[0]
sepal_length_setosa = table.col_values(0)[1:50]
sepal_width_setosa = table.col_values(1)[1:50]
sepal_length_versicolor = table.col_values(0)[51:100]
sepal_width_versicolor = table.col_values(1)[51:100]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('sepal length and width scatter')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
ax1.scatter(sepal_length_setosa, sepal_width_setosa, c='r', marker='o')
ax1.scatter(sepal_length_versicolor, sepal_width_versicolor, c='G', marker='o')
plt.show()
