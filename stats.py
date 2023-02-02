import matplotlib.pyplot as plt
from csv_reader import csv_reader
from more_itertools import sort_together
def year_lead(df):
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
            diffs.append((amount_f[i]/(amount_m[index]+amount_f[i]))*100)
    return years,diffs

def plot_year_leads():
    year_m,year_f,amount_m,amount_f = year_lead(csv_reader("data/train.csv"))
    res = sort_together([year_m, amount_m])
    year_m = res[0]
    amount_m = res[1]
    year_f,amount_f = sort_together([year_f, amount_f])
     
    year_diff,amount_diff = get_diffs(year_m,year_f,amount_m,amount_f)
    plt.plot(year_diff,amount_diff, label = "Female%_of_roles")
    plt.plot(year_m,amount_m, label = "Male")
    plt.plot(year_f,amount_f, label = "Female")
    plt.xlabel("year")
    plt.xlabel("Leads")
    plt.legend()
    plt.show()

def calculate_money_lead():
    df = csv_reader("data/train.csv")
    money_male = []
    money_female = []
    df = df[['Gross', 'Lead']]
    df = list(df.values)
    for d in df:
        if d[1] == "Female":
            money_female.append(d[0])
        else:
            money_male.append(d[0])
    return sum(money_male)/len(money_male),sum(money_female)/len(money_female)
def calculate_percentage_gender():
    df = csv_reader("data/train.csv")
    gender = df.get('Lead')
    male = 0
    female = 0
    for g in gender:
        if g == "Female":
            female +=1
        else:
            male +=1
    return (male/(female+male))*100,(female/(female+male))*100

if __name__ == "__main__":
    
    perc_male,perc_female = calculate_percentage_gender()
    print(perc_male,"are male and",perc_female,"are female")
    mean_male,mean_female = calculate_money_lead()
    print(mean_male,"is the average gross with male",mean_female,"is the average with female")
    print("this means a film with a male lead makes",((mean_male/(mean_female))*100)-100,"percent more than a movie with a female lead")
    plot_year_leads()