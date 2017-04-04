'''
Created on 7 mar. 2017

@author: fara
@todo: Try to improve the algorithm with stacking, NN or CNN
'''

from mlxtend.classifier import StackingClassifier, EnsembleVoteClassifier
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, \
    voting_classifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics.regression import mean_squared_error, \
    explained_variance_score
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import pandas as pd
import re as re


# Utility classes
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

def isMaiden(name):
    title_search = re.search(' \(.', name)
    # If the title exists, extract and return it.
    if title_search:        
        return 1
    else: return 0



#https://www.kaggle.com/sinakhorami/titanic/titanic-best-working-classifier
def main():
    
    # Import train csv into train_df
    train_df=pd.read_csv("./train.csv")
    test_df=pd.read_csv("./test.csv")
    
    # Merge both dataset
    full_data = [train_df, test_df]    
    #print(test_df.info())
        
    # FamiliSize = SibSp and Parch
    for dataset in full_data:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    #print (train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
    
    # Create a feature isAlone (1 =True) (0=False)
    for dataset in full_data:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    #print (train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
    
        
    # Embarked
    for dataset in full_data:
        # Filling with 'S' because it is the most used (644 times in train_df)
        # print (train_df[['PassengerId','Embarked']].groupby(['Embarked']).count())
        dataset['Embarked'] = dataset['Embarked'].fillna('S')
    #print (train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
    
    #Fare calculation     
    #print (test_df['Fare'].index[test_df['Fare'].apply(np.isnan)])            
    for dataset in full_data:
        #Calculate the missing Fares based on Pclass and Embarked features        
        dataset['Fare'] = dataset.groupby(['Pclass','Embarked'])['Fare'].transform(lambda x: x.fillna(x.median()))        
        
    # Name Title mapping
    for dataset in full_data:
        dataset['Title'] = dataset['Name'].apply(get_title)
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
         'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')  
    
    #Maiden Name
    for dataset in full_data:
        dataset['Maiden'] = dataset['Name'].apply(isMaiden)
        
    # For Age calculation    
    for dataset in full_data:  
        #Calculate age based on the title        
        dataset['Age'] = dataset.groupby('Title')["Age"].transform(lambda x: x.fillna(x.mean()))                   
        dataset['Age'] = dataset['Age'].astype(int)
    
    
    # Data cleanup
    for dataset in full_data:
        # Mapping Sex
        dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
        
        # Mapping titles
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)
        
        # Mapping Embarked
        dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
        
        # Mapping Fare        
        fare_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
        dataset['Fare'] = fare_scaler.fit_transform(dataset[['Fare']]).round(2)
        #dataset['Fare'] = dataset['Fare'].astype(int)
        
        # Mapping Age
        '''
        dataset.loc[ dataset['Age'] <= 20, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 28), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 28) & (dataset['Age'] <= 38), 'Age'] = 2        
        dataset.loc[ dataset['Age'] > 38, 'Age'] = 3
        '''
        '''
        dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
        '''
        
        # Preprocessing: Cabin 
        dataset['Cabin'].fillna('0', inplace=True)  
                        
        dataset.loc[dataset.Cabin.str[0] == 'A', 'Cabin'] = 1
        dataset.loc[dataset.Cabin.str[0] == 'B', 'Cabin'] = 2
        dataset.loc[dataset.Cabin.str[0] == 'C', 'Cabin'] = 3
        dataset.loc[dataset.Cabin.str[0] == 'D', 'Cabin'] = 4
        dataset.loc[dataset.Cabin.str[0] == 'E', 'Cabin'] = 5
        dataset.loc[dataset.Cabin.str[0] == 'F', 'Cabin'] = 6
        dataset.loc[dataset.Cabin.str[0] == 'G', 'Cabin'] = 7
        dataset.loc[dataset.Cabin.str[0] == 'T', 'Cabin'] = 8
        
        dataset['Cabin'] = dataset['Cabin'].astype(int)
    
    
    # Save the column to be use to submit the results
    testPassengerId = test_df['PassengerId']
    
    # Feature Selection
    drop_elements = ['PassengerId', 'Name', 'Ticket', 'SibSp',  'Parch','Title']
    train_df = train_df.drop(drop_elements, axis = 1)    
    test_df  = test_df.drop(drop_elements, axis = 1)
    
   
    
    
    print(train_df.head())
    
    # Get numpy arrays
    train = train_df.values
    test  = test_df.values
    
    # To be able to reproduce
    seed=0 
    
        
    # Remove first column from train (Survived)
    X = train[0::, 1::]
    # Let just Survived column
    y = train[0::, 0]
    
    #X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.1,random_state=seed)
    
    X_train = X
    y_train = y
    
    
    # Define Classifiers
    classifiers = {
        'RandomForestClassifier' : RandomForestClassifier(random_state=seed),       
        'DecisionTreeClassifier' : DecisionTreeClassifier(),
        'AdaBoostClassifier' : AdaBoostClassifier(),
        'GradientBoostingClassifier' : GradientBoostingClassifier(random_state=seed),         
        'ExtraTreesClassifier': ExtraTreesClassifier(random_state=seed)                   
        }
    
    # Ensemble Classifier    
    eclf = EnsembleVoteClassifier(clfs=classifiers.values())
    
    #Define params based on best result  
    params={'randomforestclassifier__n_estimators': [500],
            'randomforestclassifier__max_depth' : [50],
            'gradientboostingclassifier__n_estimators' : [50],
            'gradientboostingclassifier__max_depth' :  [2],
            'adaboostclassifier__n_estimators' : [100],
            'adaboostclassifier__learning_rate' : [0.1],      
            'extratreesclassifier__n_estimators': [200],
            'extratreesclassifier__max_depth' : [2]}
    
    # GridSearcj
    grid=GridSearchCV(estimator=eclf, param_grid=params, cv=5,n_jobs=-1)
    
    #Fit the train dataset
    grid.fit(X_train, y_train)
    
    for params, mean_score, scores in grid.grid_scores_:
        print("%0.3f(+/-%0.03f) for%r"%(mean_score, scores.std()/2, params))
    print("Best parameters: ",grid.best_params_)
   
    #Predict the result in the test dataset
    Y_pred = grid.predict(test).astype(int)
    
    
    submission = pd.DataFrame({
        "PassengerId": testPassengerId,
        "Survived": Y_pred
    })
    submission.to_csv('./titanic_response.csv', index=False)
    
    print("Finished")
    
if __name__ == '__main__':
    # Setting some printing parameters
    pd.set_option('max_columns',20)
    pd.set_option('max_rows',1000)
    pd.set_option('display.expand_frame_repr', False)
    main()