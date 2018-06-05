# import package
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV , KFold
from sklearn.svm import SVR

# read train data and test data
df=pd.read_csv("training_data.csv")
df_test=pd.read_csv("test_data.csv")

#make text become array
text=df.text.values
new_text=df_test.text.values
y=df.stars.values

#Let the review_id , business_id and user_id become 0~1
id_train=(df.iloc[:,:3]-df.iloc[:,:3].min())/(df.iloc[:,:3].max()-df.iloc[:,:3].min())
id_test=(df_test.iloc[:,:3]-df_test.iloc[:,:3].min())/(df_test.iloc[:,:3].max()-df_test.iloc[:,:3].min())

#use the tfidfvectorizer package and encode the text
vectorizer=TfidfVectorizer(max_df=0.8,min_df=3).fit(text)
text=vectorizer.transform(text)
fea=vectorizer.transform(new_text)

#Let the correcting id and encoding text become array from
textid=pd.DataFrame(id_train)
textname=pd.DataFrame(text.toarray())
text=pd.concat([textid,textname],axis=1)
text=text.values
newid,newname=pd.DataFrame(id_test),pd.DataFrame(fea.toarray())
fea=pd.concat([newid,newname],axis=1)
fea=fea.values

#Select the best parameters of the module SVM
param_grid={"C":[0.1,1,10],"gamma":[0.1,1,10]}
cv=KFold(shuffle=True)
grid=GridSearchCV(SVR(),param_grid=param_grid,cv=cv,verbose=3)

#Let the GridSearchCV fit the text and predict the stars
grid.fit(text,y)
pred=grid.predict(fea)

#Decide the stars num >5 and <1 become 5 and 1
for i in range(len(pred)):
	if pred[i]>5 :
		pred[i]=5
	elif pred[i]<1 :
		pred[i]=1

#Write the predict data into csv
cs=pd.DataFrame({"aa":df_test.iloc[:,0],'bb':pred})
cs.to_csv("pred.csv",index=False,header=False)


