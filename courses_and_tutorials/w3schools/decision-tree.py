import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = pandas.read_csv('./comedy_data.csv')
print('\nComedy Data:\n', df)
print(df)

print('\nAll data has to be numerical to create a decision tree.\n')
d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)

print('\nComedy Data:\n', df)
print(df)

print('\nX are the feature columns, y is the target column.\n')

features = ['Age', 'Experience', 'Rank', 'Nationality']
X = df[features]
y = df['Go']

print('\nX:\n', X)
print('\ny:\n', y)

print('\nCreate a decision tree classifier.\n')
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X.values, y.values)

tree.plot_tree(dtree, feature_names=features)

print('\nPredict if a 40 year old American with 10 years of experience and a rank of 7 will make me go to the show.\n')
print(dtree.predict([[40, 10, 7, 1]]))
print('\nPredict if a 40 year old American with 10 years of experience and a rank of 6 will make me go to the show.\n')
print(dtree.predict([[40, 10, 6, 1]]))