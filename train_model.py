import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv("Heart.csv")

X = df.iloc[:,: -1]
le = LabelEncoder()

y = le.fit_transform(df["Thal"])

X_train, X_test, y_trian, y_test = train_test_split(X,y,train_size=0.25,random_state=0)

rdcmodel = RandomForestClassifier(n_estimators=200,random_state=42)


rdcmodel.fit(X_train,y_trian)



# print(rdcmodel.predict([[5.1,3.5,1.4,0.2]]))

with open("rdcmodel.pkl","wb") as f:
    pickle.dump(rdcmodel,f)

with open("leencoder.pkl","wb") as f:
    pickle.dump(le,f)

# with open("drcmodel.pkl","wb") as f:
#     pickle.dump(drcmodel,f)

# with open("lrmodel.pkl","wb") as f:
#     pickle.dump(lrmodel,f)

print("Model trained and saved as rdcmodel.pkl")