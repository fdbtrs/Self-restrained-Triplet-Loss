import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
from csv import reader

from matplotlib.font_manager import FontProperties
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}

matplotlib.rc('font', **font)

positive=[]
negative=[]
fig = plt.figure()
det_plot = fig.add_subplot(111)

det_plot.set_ylabel('Distance',fontsize=14,fontweight="bold")
det_plot.set_xlabel('Iteration',fontsize=14,fontweight="bold")

det_plot.grid(True)
det_plot.legend( prop=FontProperties(size=20,weight="bold"))
font0 = FontProperties()
font0.set_size(20)
font0.set_weight("bold")

with open('../logs/Triplet.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    header = next(csv_reader)
    # Check file as empty
    if header != None:
        # Iterate over each row after the header in the csv
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            #print(row)
            positive.append(float(row[2]))
            negative.append(float(row[3]))
det_plot.plot(positive,linewidth=2, label="Triplet loss-d1")
det_plot.plot(negative,linewidth=2,label="Triplet loss-d2")
positive=[]
negative=[]
d3=[]
with open('../logs/SRT.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    header = next(csv_reader)
    # Check file as empty
    if header != None:
        # Iterate over each row after the header in the csv
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            #print(row)
            positive.append(float(row[2]))
            negative.append(float(row[3]))
            d3.append(float(row[3]))
det_plot.plot(negative,linewidth=2,linestyle=":",label="d3")
det_plot.plot(positive,linewidth=2, label="SRT loss-d1")
det_plot.plot(negative,linewidth=2,linestyle="--",label="SRT loss-d2")

plt.legend()
plt.savefig("triplet_training.eps")
plt.show()

