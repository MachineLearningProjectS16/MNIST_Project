import matplotlib.pyplot as plt
#I had to manually do the plotting with results because of an error with
#matplotlib on AWS GPU

def line_plot(x,y,y1,x_lable,y_lable,title):
    plt.plot(x,y)
    plt.plot(x,y1)
    plt.xlabel(x_lable)
    plt.ylabel(y_lable)
    plt.axis([min(x),max(x),min(y),max(y)])
    plt.title(title)
    plt.legend(['Hidden units=256,dop out=0.5', 'Hidden units=800,drop\
        out=0.2'], loc='lower right')
    plt.show()

x_arr=[1,5,10,20,30,40,45]
y_arr=[97.19,99.02,99.32,99.50,99.55,99.60,99.55]
y_arr2=[98.11,99.28,99.47,99.51,99.54,99.59,99.53 ]
line_plot(x_arr,y_arr,y_arr2,"Num Iterations","Test Accuracy","Test Accuracy vs\
        Num Iterations CNN Ensembles")

def line_plot_2(x,y,x_lable,y_lable,title):
    plt.plot(x,y)
    plt.xlabel(x_lable)
    plt.ylabel(y_lable)
    plt.axis([min(x),max(x),min(y),max(y)])
    plt.title(title)
    plt.show()

x_arr=[10,20,40,50,70,100]
y_arr=[98.04,98.50,98.85,98.86,98.97,98.97]
line_plot_2(x_arr,y_arr,"Num iter","Test Accuracy","Test Accuracy vs Num\
Iter NN")

def line_plot_3(x,y,y1,x_lable,y_lable,title):
    plt.plot(x,y)
    plt.plot(x,y1)
    plt.xlabel(x_lable)
    plt.ylabel(y_lable)
    plt.axis([min(x),max(x),min(min(y1),min(y)),max(max(y1),max(y))])
    plt.title(title)
    plt.legend(['Ensemble CNN=5 with synth data', 'Without synth data'], loc='lower right')
    plt.show()

x_arr=[1,10,20,30,40,50,60]
y_arr=[97.19,99.02,99.32,99.50,99.55,99.60,99.55]
y_arr2=[92.33,98.17,98.9,99.2,99.6,99.34,99.39 ]
line_plot_3(x_arr,y_arr,y_arr2,"Num Iterations","Test Accuracy","Test Accuracy vs\
        Num Iterations CNN Ensembles")

