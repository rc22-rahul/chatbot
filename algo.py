

import pandas as pd
data=pd.read_csv("data.csv")
data

courseconv={"Automobile":1,"Bio-Medical":2,"Chemical":3,"Civil":4,"Computer":5,"EC":6,"Electrical":7,
"Environment":8,"IC":9,"IT":10,"Mechanical":11,"Plastic":12,"Rubber":13,"Textile":14}
input1=input("Enter category : ")
print('Enter course name from this list :',courseconv.keys())
input2=input("Enter Course:")

try:
    input3=courseconv[input2]
except:
    print('Enter Course name properly!')

inputrank=int(input("Enter merit rank :"))

mean=data.groupby('Course')[input1].mean()

meanrank=mean[input3]
if(inputrank<=meanrank):
    print("You may be eligible for",input2,"department")
else:
    print("You might not be eligible for",input2, "department")