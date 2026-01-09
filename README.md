# Beer-Production-Dataset
My second project in regards to practicing Data Science Techniques on a data set about beer! Which is wildly popular amongst my group of friends....
I used sklearn, numpy, pandas and matplotlip to visualize, run tests and interpret my data.

Beer Production Analysis – What did we find??

## Overview

This project analyzes U.S. beer production data from the **TidyTuesday (March 31, 2020)** dataset. The goal was to explore trends in beer production, understand differences between grain and non-grain products, examine taxation and brewer size effects, and apply basic predictive modeling to support data-driven conclusions.

The analysis combines **exploratory data analysis (EDA)**, **visualization**, and **machine learning** to interpret patterns in beer production over time.

---

## Major Conclusions

### 1. Beer Keg Production Trends

* Beer produced in **kegs**, while currently the least common type, shows a **consistent upward trend**.
* Strong correlation exists between beer taxed in the prior year and the current year.
* Linear regression suggests that **beer keg production and associated taxation are likely to continue increasing** in future years.
* As keg production rises, tax revenue related to kegs is expected to grow accordingly.

**Interpretation:** Although kegs represent a smaller share of production, their growth rate indicates increasing demand, particularly from modern distribution channels such as bars and breweries.

---

### 2. Grain vs Non-Grain Brewing Materials

* Both **grain-based** and **non-grain-based** beer production show upward trends.
* Grain-based beers dominate overall production volume and exhibit a wider range of values.
* Non-grain products are increasing steadily, but still remain a smaller portion of total output.
* Regression analysis indicates similar growth behavior for both categories, with grain-based beer maintaining higher demand.

**Conclusion:** While innovation in non-grain products is growing, **grain-based beer remains the primary driver of production** and demand.

---

### 3. Predicting Grain-Based Beer Production

* A **logistic regression model** was trained to classify whether beer production was grain-based using:

  * Year
  * Year-to-date production (current year)
  * Year-to-date production (prior year)
* The model achieved reasonable accuracy, demonstrating that **historical production features contain predictive signal**.

**Interpretation:** Even simple models can effectively distinguish production types, making this a useful baseline approach for classification tasks.

---

### 4. Beer Production by State

* Visualization across states shows relatively **uniform production patterns** when aggregated.
* Total beer production from 2008–2019 exhibits a **slight downward trend** in recent years.
* Production levels in 2018 and 2019 were nearly identical, suggesting stabilization.

**Possible Factors:**

* Increased competition from alternative alcoholic beverages
* Changing preferences among younger consumers

**Prediction:** Beer production is likely to **remain stable or continue a mild decline** in the near future.

---

### 5. Brewer Size and Taxation Effects

* Strong correlation exists between **taxable removals** and **total barrels shipped**.
* Regression analysis suggests beer is often produced and stored before being taxed at the point of removal.
* Brewer size distribution shows that production is concentrated among certain size categories, while the number of brewers has increased over time.

**Interpretation:** Taxation policies influence storage and shipping strategies, with breweries optimizing when beer becomes taxable.

---

## Final Takeaway

Overall, the analysis highlights a **continued popularity of beer**, with evolving production patterns across product types and materials. Despite slight declines in total output, specific categories—such as kegs and non-grain products—are growing. These insights suggest increasing costs, continued investment in brewing infrastructure, and evolving consumer demand.

Although the dataset is several years old, the observed trends remain consistent with modern beer production patterns and provide a solid foundation for exploratory and predictive data science work.

---

## Skills Demonstrated

* Exploratory Data Analysis (EDA)
* Data visualization
* Feature engineering
* Linear and logistic regression
* Model evaluation and interpretation
* Translating data insights into real-world conclusions
