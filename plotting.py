import matplotlib.pyplot as plt
from csv_reader import csv_reader
from more_itertools import sort_together
def plot_year_actor(df):
    cl = df.get('Lead')
    year = df.get('Year')
    year_countlabel_m = {}
    year_countlabel_f = {}
    for index in range(len(cl)):
        l = cl[index]
        y = year[index]
        
        if l == "Male":
            if str(y) in year_countlabel_m:
                
                year_countlabel_m[str(y)]= year_countlabel_m[str(y)]+ 1
                
            else:
                
                year_countlabel_m[str(y)]= 1
        else:
            if str(y) in year_countlabel_f:
                
                year_countlabel_f[str(y)]= year_countlabel_f[str(y)]+ 1
                
            else:
                
                year_countlabel_f[str(y)]= 1

    year_m = year_countlabel_m.keys()
    amount_m = list(year_countlabel_m.values())
    year_f = year_countlabel_f.keys()
    amount_f = list(year_countlabel_f.values())
    year_m = [int(year) for year in year_m]

    year_f = [int(year) for year in year_f]
    return year_m,year_f,amount_m,amount_f
def index_of(val, in_list):
    try:
        return in_list.index(val)
    except ValueError:
        return -1     
def get_diffs(year_m,year_f,amount_m,amount_f):
    years = []
    diffs = []
    for index in range(len(year_m)):
        year = year_m[index]
        
        i = index_of(year,year_f)
        if i>=0:
            years.append(year)
            diffs.append((amount_f[i]/amount_m[index])*100)
    return years,diffs



if __name__ == "__main__":
    year_m,year_f,amount_m,amount_f = plot_year_actor(csv_reader("data/train.csv"))
    res = sort_together([year_m, amount_m])
    year_m = res[0]
    amount_m = res[1]
    res = sort_together([year_f, amount_f])
    year_f = res[0]
    amount_f = res[1]
    year_diff,amount_diff = get_diffs(year_m,year_f,amount_m,amount_f)
    plt.plot(year_diff,amount_diff, label = "Diff%")
    plt.plot(year_m,amount_m, label = "Male")
    plt.plot(year_f,amount_f, label = "Female")
    plt.xlabel("year")
    plt.xlabel("Leads")
    plt.legend()
    plt.show()