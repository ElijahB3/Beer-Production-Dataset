#Tidy Tuesday prject July 27 2025
import numpy as np  # pyright: ignore[reportMissingImports]
import pandas as pd  
import matplotlib.pyplot as plt 

from sklearn.linear_model import LogisticRegression #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay #type: ignore
from sklearn.metrics import classification_report#type:ignore
brewing_materials = pd.read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2020/2020-03-31/brewing_materials.csv')
beer_taxed = pd.read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2020/2020-03-31/beer_taxed.csv')
brewer_size = pd.read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2020/2020-03-31/brewer_size.csv')
beer_states = pd.read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2020/2020-03-31/beer_states.csv')


# Lets just start by trying to plot as much as we can too check for outliers
#We will attempt to apply binning, normalization, and scaling to the data and scrubbing here

# Minimum we should have atleast 4 subplots for each section of the data set, try to find some questions and answer whilst applying binning
# norm, scaling etc.

def show_scatter_plot(df, x_col, y_col, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_col], df[y_col], alpha=0.5)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
   

def show_plot(df, x_col, y_col, title):
    plt.figure(figsize=(10, 6))
    plt.plot(df[x_col], df[y_col], marker='o', linestyle='-', alpha=0.5)
    plt.Color('red')
    plt.grid(True)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    

# show_plot(beer_states, 'year', 'barrels', 'Beer Production by Year')
# show_scatter_plot(beer_states, 'year', 'barrels', 'Beer Production by Year')
#These plots arent very usefull...
#Drawing these plots have not been great to lets find the best correlations from each set

# print(beer_taxed.corr(numeric_only=True)) 
# print(brewing_materials.corr(numeric_only=True))
# print(brewer_size.corr(numeric_only=True))   
# print(beer_states.corr(numeric_only=True))

#Showing the correlations open up to plot those with the positive ones, so

#show_scatter_plot(beer_taxed, 'month_prior_year', 'month_current', 'Beer taxed this month last year vs this year')
#High correlation of beer taxe between years I'd like to dive into this

#print(beer_taxed.count()) 

#I want a bar plot of 'type' 
#plt.show()

#Lets target the beer in kegs as they have the smallest count
#We plot all positive correlations of the beer kegs and try and come to a conclucsion
beer_kegs = beer_taxed[beer_taxed['type'] == 'In kegs']

# print(beer_kegs.head(50))
# print(beer_kegs.corr(numeric_only=True))

fig, axs = plt.subplots(2, 3, figsize=(18, 16))
fig.suptitle("Beer Kegs Analysis")

axs[0,0].bar(beer_kegs['month'], beer_kegs['ytd_prior_year'], alpha=0.5)
axs[0,0].set_title('Beer Kegs Barrels Made in 2018')

axs[1,0].bar(beer_kegs['month'], beer_kegs['ytd_current'], alpha=0.5)
axs[1,0].set_title('Beer Kegs Barrels Made in 2019')

axs[0,1].scatter(beer_kegs['month_current'], beer_kegs['month_prior_year'], alpha=0.5,color='red')
axs[0,1].set_title('Beer Kegs Made From 2018 to 2019')

axs[1,1].bar(beer_kegs['month'].value_counts().index,beer_kegs['month'].value_counts().values, alpha=0.5,color ='red')
axs[1,1].set_title('Total Amount of Kegs Made per Month')

axs[0,2].pie(beer_taxed['type'].value_counts(), 
             labels=beer_taxed['type'].value_counts().index, 
             autopct='%1.1f%%',
             textprops={'fontsize': 6})
axs[0,2].set_title("Distribution of Beer Types Produced")

axs[1,2].pie(beer_kegs['tax_status'].value_counts(),
             labels=beer_kegs['tax_status'].value_counts().index,
             autopct='%1.1f%%',
            )
axs[1,2].set_title=("Beer Kegs Tax Status")
#Lets predict the amount of tax beer keg barrels based off the kegs made in 2018 to 2019
copy_kegs = beer_kegs.copy()
x = beer_kegs['month_current']
y = beer_kegs['month_prior_year']

m,b = np.polyfit(x,y,1)
copy_kegs['kegs_month_pred'] = (m*x) + b

axs[0,1].plot(beer_kegs['month_current'], copy_kegs['kegs_month_pred'],alpha=0.5,color='black')


ans_kegs = """The regression analysis indicates a positive relationship between beer keg production in the prior year and the current year.
Monthly keg production shows an upward trend, suggesting that keg output has been increasing over time.

Because beer taxes are applied based on production and removal volumes, increased keg production may be associated with higher
taxable amounts in future periods. Although kegs currently represent a smaller share of total beer production, their consistent
growth suggests a rising contribution to overall production and taxation. """

#plt.show()
#Ill leave kegs for now, which is a sub category of beer taxed, id like to do a full analysis of that part but lets move onto brewing materials

# print(brewing_materials.head(30))
# print(brewing_materials.corr(numeric_only=True))

fig2, axs2 = plt.subplots(3,3,figsize= (12,10))
fig2.suptitle("Brewing Materials Set")

copy_brew = brewing_materials.copy()

#High Correlation between the years so well draw a line of regression
#Should predict the amount of 
mask = np.isfinite(copy_brew['ytd_current']) & np.isfinite(copy_brew['ytd_prior_year'])
scrubbed_brew = copy_brew[mask]

x_mat= scrubbed_brew['ytd_current']
y_mat= scrubbed_brew['ytd_prior_year']


m,b = np.polyfit(x_mat,y_mat,1)
scrubbed_brew['ytd_pred'] = (m *x_mat) + b

axs2[0,0].scatter(scrubbed_brew['ytd_current'],scrubbed_brew['ytd_prior_year'], alpha=0.5,color='forestgreen')
axs2[0,0].plot(scrubbed_brew['ytd_current'],scrubbed_brew['ytd_pred'],alpha=0.5,color='deeppink')
axs2[0,0].set_title("Prediction of Beer to Be Made in the next Year",fontsize=6)

axs2[0,1].scatter(copy_brew['ytd_current'],copy_brew['month_current'],alpha=0.5,color='m')
axs2[0,1].set_title("Current number of barrels for this year",fontsize=6)

axs2[0,2].pie(copy_brew['material_type'].value_counts(),
            labels = copy_brew['material_type'].value_counts().index,
            autopct="%1.1f%%",
            textprops={'fontsize': 6})
axs2[0,2].set_title("Grain Product Ratios",fontsize=6)

grain_set = scrubbed_brew[copy_brew['material_type'].isin(["Grain Products", "Total Grain products"])]
non_grain_set = scrubbed_brew[copy_brew['material_type'].isin(["Non-Grain Products", "Total Non-Grain products"])]

#Weve got both grain and non grain as sets seperated

# print(grain_set.corr(numeric_only = True))
# print(non_grain_set.corr(numeric_only = True))

#Is there a factor to tell which is grain and non grain? Nope not here, but lets try and predict the demand of them
x_grain = grain_set['ytd_prior_year']
y_grain = grain_set['ytd_current']

m,b = np.polyfit(x_grain,y_grain,1)
grain_set['ytd_pred'] = (m * x_grain) + b

axs2[1,0].scatter(grain_set['ytd_prior_year'], grain_set['ytd_current'],alpha=0.5,color = 'y')
axs2[1,0].plot(grain_set['ytd_prior_year'],grain_set['ytd_pred'],alpha=0.5,color = 'b')
axs2[1,0].set_title("Prediciton of Grain Barrels",fontsize=6)

axs2[1,1].bar(grain_set['month'].value_counts().index,grain_set['month'].value_counts().values,color="C5")
axs2[1,1].set_title("Total Grain Beer Produced over 2019",fontsize = 6) 

axs2[1,2].bar(grain_set['year'].value_counts().index,grain_set['year'].value_counts().values,color = 'C4')
axs2[1,2].set_title("Grain Beer produced from 2008-2014",fontsize = 6)

#Non Grain Prodcuts
x_nongrain = non_grain_set['ytd_prior_year']
y_nongrain = non_grain_set['ytd_current']

m,b = np.polyfit(x_nongrain,y_nongrain,1)
non_grain_set['ytd_pred'] = (m * x_nongrain) + b

axs2[2,0].scatter(non_grain_set['ytd_prior_year'], non_grain_set['ytd_current'],alpha=0.5,color = 'C6')
axs2[2,0].plot(non_grain_set['ytd_prior_year'],non_grain_set['ytd_pred'],alpha=0.5,color = 'c')
axs2[2,0].set_title("Prediciton of Non-Grain Barrels",fontsize=6)

axs2[2,1].bar(non_grain_set['month'].value_counts().index,non_grain_set['month'].value_counts().values,color="C2")
axs2[2,1].set_title("Total Non-Grain Beer Produced over 2019",fontsize = 6) 

axs2[2,2].bar(non_grain_set['year'].value_counts().index,non_grain_set['year'].value_counts().values,color = 'C1')
axs2[2,2].set_title("Non-Grain Beer produced from 2008-2014",fontsize = 6)

ans_grain_vs_non = '''The non grain products is increasing through out the years, its likely we'll see more production in those barrels as well
However, there are more demand for grain based beers since our range is larger than non grain. Although both have similar prediction of production there will be higher
demand for grain-based beer. The production of grain-based beer is higher.'''

#Well attempt to predict if a beer is grain or not based off 3 features
#1. ytd_current
#2. year
#3. month

#The if condition
copy_brew['is_grain'] = copy_brew['material_type'].apply(lambda x: 1 if x =="Grain Products" else 0)

#Now our feature to for our model to tell the difference
X = copy_brew[['year','ytd_prior_year','ytd_current']].dropna()
y = copy_brew.loc[X.index, 'is_grain']

#Split the data up
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#Fitting to the set now
model = LogisticRegression()
model.fit(X_train,y_train)

#Now predict
score = model.score(X_test,y_test)
print(f"Accuracy: {score:.2f}")

#Lets print this
y_pred = model.predict(X_test)
print("Predictions:", y_pred)
print("Actual:", y_test.values)

#Confusion matrix to see the true positives and false negative etc...
cm = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()
print(classification_report(y_test,y_pred))
#plt.show()

#Well create a set for the beer states next

fig3, axs3 = plt.subplots(2,2, figsize= (10,10))
fig3.suptitle("Beer States Set")
axs3[0,0].pie(beer_states['type'].value_counts(),
             labels = beer_states['type'].value_counts().index,
             autopct="%1.1f%%",
             textprops = {'fontsize': 8})

axs3[0,1].pie(beer_states['state'].value_counts(),
              labels=beer_states['state'].value_counts().index,
              autopct="%1.1f%%",
              textprops = {"fontsize":6})
axs3[0,1].set_title("Distribution of Produced Beer from Each State")

axs3[1,0].bar(beer_states['year'],beer_states['barrels'])
axs3[1,0].set_title("Barrels produced from 2008-2019")

total_made = beer_states[beer_states['state'] == 'total']

axs3[1,1].bar(total_made['year'],total_made['barrels'])
axs3[1,1].set_title("Total Beer Made by Year")

ans_beer_states = '''After visualizing the production of beer of each state in the USA, It seems every state is producing the same amount. The amount of beer produced is slightly going on a negative trend,
as the bar plot shows a slight decrease over the past decade. There could be many factors of this, since many other alcoholic options exists now and the popularity of beer
with the younger generation could be a factor in this. In 2018 and 2019 we saw a very similar amount made in both years hence I would predict that well either be staying
the same or continuing on a slgith downward trend of beer production for the next decade. '''

# print(brewer_size.head(10))
# print(brewer_size.corr(numeric_only=True))

#high correlation between taxable_removals, and the total shipped
plt.close()

copy_size = brewer_size.copy()
mask2 = np.isfinite(copy_size['taxable_removals']) & np.isfinite(copy_size["total_shipped"]) & np.isfinite(copy_size['total_barrels'])
copy_size = copy_size[mask2]

fig4, axs4 = plt.subplots(2,2, figsize = (10,10))
fig4.suptitle("Brewing Size Set")

x_tax = copy_size['taxable_removals']
y_total = copy_size['total_shipped']

m,b = np.polyfit(x_tax,y_total,1)
copy_size['shipped_pred'] = (m*x_tax) + b

axs4[0,0].scatter(copy_size['taxable_removals'],copy_size['total_shipped'])
axs4[0,0].plot(copy_size['taxable_removals'],copy_size['shipped_pred'],color="g")
axs4[0,0].set_title("Prediction of Barrels to be Shipped and Consumed under Taxation", fontsize = 6)
#Doesnt produce a line... UNTRUE! We scrubbed the data then we got a line. Good instinct

ans_size = '''I believe the line is saying that barrels of beer being shipped off from duty free warehouse, where the tax is applied ONLY when Its taken out for domestic use/sales
So if More beer is made to have a removable tax, then they ship more beer to be under the same warehouse where the owners dont pay a tax on storing them
and pays it only when its shipped off to be sold/consumption. So the line predict beer that has not been taxed
'''


axs4[0,1].pie(copy_size['brewer_size'].value_counts(),
              labels=copy_size['brewer_size'].value_counts().index,
              autopct = "%1.1f%%",
              textprops = {'fontsize':5})

#Line for taxable barrels and the total amount produced

x_nontax = copy_size['taxable_removals']
y_barrels = copy_size['total_barrels']

m,b = np.polyfit(x_nontax,y_barrels,1)
copy_size['pred_barrels'] = (m * x_nontax) + b

axs4[1,0].scatter(copy_size['taxable_removals'],copy_size['total_barrels'])
axs4[1,0].plot(copy_size['taxable_removals'],copy_size['pred_barrels'],color="r")
axs4[1,0].set_title("Prediction of Barrels to be Made")

axs4[1,1].bar(copy_size['year'],copy_size['n_of_brewers'])
axs4[1,1].set_title("Number of Brewers from 2010-2019")

print(ans_grain_vs_non)
print(ans_beer_states)
print(ans_kegs)
print(ans_size)

ans_con = '''After visualizing the data set, there is a clear increase in the popularity of beer and its production
althought this set is almost 5 years old, its safe to say that our data still aligns with the data we have today as beer produciton is on a continual increase.
We can assume the costs increase to produce beer and how the investment will bring in more funding since'''

plt.show()