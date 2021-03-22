# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sc
import numpy as np
import random


# %%
LOL_data = pd.read_csv("high_diamond_ranked_10min.csv")
del LOL_data["gameId"]
del LOL_data["redFirstBlood"]  # for the model we wont need gameId column


# %%
LOL_data


# %%
def basic_info(df):
    return {"shape": df.shape,
            "columns": list(df.columns),
            "Nas": df.isna().sum(),
            "info": df.info(),
            "describe": df.describe()}


# %%
# basic_info(LOL_data)
# we have no missing values
# we can also see here some basics descriptive statistics and info about the features of the dataset.¿


# %%
print("Duplicates", len(LOL_data[LOL_data.duplicated()]))

# %% [markdown]
# There is nothing much to do about the clean process so let's proceed to do some EDA.
# %% [markdown]
# ## EDA (Exploratory Data Analysis)

# %%
sns.set_style("whitegrid")
plt.style.use("ggplot")


# %%
ax = sns.catplot(x="blueWins", data=LOL_data, palette=["r", "b"], kind="count")
plt.ylabel("Total victories")
plt.xlabel("Red_Team = 0              Blue_Team = 1")
plt.title("Total Victories per Team")
plt.show()

LOL_data.value_counts("blueWins")

# %% [markdown]
# Red team won 19 games more than blue team, this just for the approx 10k ranked games (SOLO QUEUE) from a high ELO (DIAMOND I to MASTER)

# %%
fig, ax = plt.subplots(1, 2, figsize=(15, 4))
sns.histplot(x="blueWardsPlaced", data=LOL_data, ax=ax[0])
sns.histplot(x="redWardsPlaced", data=LOL_data, ax=ax[1], color="r")
plt.show()

# its nor normal to place more than 15 wards within the first ten min. and more than that is really strange we have some ouliers here
# lets deal with it


# %%
def outliers(x):
    if x > 15:
        x = random.randint(1, 15)
    return x


LOL_data["blueWardsPlaced"] = LOL_data["blueWardsPlaced"].apply(outliers)
LOL_data["redWardsPlaced"] = LOL_data["redWardsPlaced"].apply(outliers)


# %%
fig, ax = plt.subplots(1, 2, figsize=(15, 4))
sns.histplot(x="blueWardsPlaced", data=LOL_data, ax=ax[0])
sns.histplot(x="redWardsPlaced", data=LOL_data, ax=ax[1], color="r")
plt.show()

# %% [markdown]
# The following code is to create some visualizations, if you want to see them uncomment and run. Are some pairplot figures. comparing some features of the datasets, and its correleation and distribution.

# %%
"""
x_vars = ["blueKills","blueTotalGold","blueTotalExperience","redKills","redTotalGold","redTotalExperience"]
y_vars = ["blueKills","blueTotalGold","blueTotalExperience","redKills","redTotalGold","redTotalExperience"]
sns.pairplot(data=LOL_data,x_vars=x_vars,y_vars=y_vars,hue="blueWins",corner=True,markers=["o","s"],diag_kind="hist")
plt.show()
"""

# %% [markdown]
# As a league of legends player, it might not surprise the graph above, where the more gold, kills, and experience a team has per game, the more are the chances to win.
#
# kill the enemy is just one way to get experience, killing minions, wards, and the monsters are some other, I will review more features.

# %%
"""
x_vars = ["blueWardsPlaced","blueAssists","blueTotalMinionsKilled","redWardsPlaced","redAssists","redTotalMinionsKilled"]
y_vars = ["blueWardsPlaced","blueAssists","blueTotalMinionsKilled","redWardsPlaced","redAssists","redTotalMinionsKilled"]
sns.pairplot(data=LOL_data,x_vars=x_vars,y_vars=y_vars,hue="blueWins",corner=True,markers=["o","s"])
plt.show()
"""

# %% [markdown]
# ## Modelling
# %% [markdown]
# #### A Basic feature engineering added
# In league of legends we can add a feature calles KDA which is represented as KDA = (Kills + Assists) / Deaths
# I have decided to add this to each team within the dataset.

# %%
LOL_data["blue_KDA"] = round(
    (LOL_data["blueKills"] + LOL_data["blueAssists"] / LOL_data["blueDeaths"]), 2)
# 2 blue teams had no KDA within the first 10 minutes
LOL_data["blue_KDA"] = LOL_data.fillna(0)


LOL_data["red_KDA"] = round(
    (LOL_data["redKills"] + LOL_data["redAssists"] / LOL_data["redDeaths"]), 2)
LOL_data["red_KDA"] = LOL_data.fillna(0)

LOL_data.shape
# LOL_data.info()


# %%
LOL_data.describe()


# %%
y = LOL_data["blueWins"]

X = LOL_data.drop("blueWins", axis=1)

# %% [markdown]
# #### Feature selection
# I would like to use some Featureselection using random forest

# %%


# %%
SEED = 42


# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y)


# %%
rf = RandomForestClassifier(
    n_estimators=400, min_samples_leaf=0.12, random_state=SEED)
rf.fit(X_train, y_train)


# %%
#selected_features = X_train.columns[rf.get_support()]
# print(selected_features)


# %%
plt.figure(figsize=(12, 7))
pd.Series(rf.feature_importances_, index=X.columns).sort_values().plot(
    kind="barh", color="gray")
plt.show()


# %%
feature_importances = pd.DataFrame(rf.feature_importances_, index=X.columns,  columns=[
                                   'importance']).sort_values('importance', ascending=False)
most_important_features = list(feature_importances.index[:24])
most_important_features


# %%
X_train = X_train[most_important_features]
X_test = X_test[most_important_features]
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

# %% [markdown]
# ## KNN
# lets start for the basics

# %%


# %%
k_range = range(1, 30)
scores = list()
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_train, y_train))
plt.plot(k_range, scores, marker="o")
plt.ylabel("acurracy")
plt.xlabel("N#_neighbors")
plt.show()


# %%
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print(accuracy_score(y_test, y_pred_knn))

# we can see that we have an overfitting problem, the data is having an accuracy score of 100% with a number of neighbors = 1, but when we test it on new data it doesn't have a good performance

# %% [markdown]
# ## Desicion tree
#

# %%


# %%
d_tree = DecisionTreeClassifier(max_depth=6, random_state=SEED)
d_tree.fit(X_train, y_train)
y_pred_dtree = d_tree.predict(X_test)
print(d_tree.score(X_train, y_train))
print(accuracy_score(y_test, y_pred_dtree))


# %%


# %%
print(confusion_matrix(y_test, y_pred_dtree))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_dtree).ravel()
print("acurracy :", (tp+tn)/(tp+tn+fp+fn))
print("precision :", tp/(tp+fn))

# %% [markdown]
# tp = true positive
# fp = false positive
# fn = false negative
# tn = true negative

# %%
predictions = pd.DataFrame(y_pred_dtree, columns=["BlueWins"])
predictions


# %%
ax = sns.catplot(x="BlueWins", data=predictions,
                 palette=["r", "b"], kind="count")
plt.ylabel("Total victories")
plt.xlabel("Red_Team = 0              Blue_Team = 1")
plt.title("Total Victories per Team")
plt.show()

predictions.value_counts("BlueWins")


# %%
