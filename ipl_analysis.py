# ============================================================
# IPL DATA ANALYSIS PROJECT (ADVANCED VERSION)
# ============================================================

# ===================== 1. IMPORT LIBRARIES =====================
# Purpose: Import all required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.proportion import proportions_ztest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sns.set_style("darkgrid")
plt.rcParams['figure.facecolor'] = '#f5f5f5'

# ===================== 2. LOAD DATA =====================
# Purpose: Load dataset and understand structure

df = pd.read_excel("python_dataset.xlsx")

print(df.head())
print(df.info())
print(df.isnull().sum())

# ============================================================
# 3. TOTAL RUNS BY TEAM (BAR GRAPH)
# Purpose: Identify best performing teams
# ============================================================

team_runs = df.groupby('Batting Team')['Total Runs'].sum().sort_values(ascending=False)

team_runs.plot(kind='bar', color='skyblue')
plt.title("Total Runs by Team")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# ============================================================
# 4. RUNS PER OVER (LINE GRAPH)
# Purpose: Analyze scoring trend over overs
# ============================================================

over_runs = df.groupby('over')['Total Runs'].sum()

over_runs.plot(marker='o', color='green')
plt.title("Runs per Over")
plt.xlabel("Over")
plt.ylabel("Runs")
plt.tight_layout()  
plt.show()

# ============================================================
# 5. HISTOGRAM (RUN DISTRIBUTION)
# Purpose: Understand frequency of runs
# ============================================================

plt.hist(df['Total Runs'], bins=20, color='purple')
plt.title("Run Distribution")
plt.show()

# ============================================================
# 6. PIE CHART (TOP 5 TEAMS)
# Purpose: Show top team contribution
# ============================================================

top5 = team_runs.head(5)

plt.pie(top5, labels=top5.index, autopct='%1.1f%%',
        colors=['red','blue','green','orange','purple'])
plt.title("Top 5 Teams")
plt.tight_layout() 
plt.show()

# ============================================================
# 7. HEATMAP (CORRELATION)
# Purpose: Identify relationships
# ============================================================

numeric_df = df.select_dtypes(include=['int64','float64'])

sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()  
plt.show()

# ============================================================
# 8. BOXPLOT (RUN DISTRIBUTION)
# Purpose: Detect outliers in runs
# ============================================================

sns.boxplot(x=df['Total Runs'], color='orange')
plt.title("Runs Distribution (Boxplot)")
plt.tight_layout()  
plt.show()

# ============================================================
# 9. DISMISSAL TYPE ANALYSIS
# Purpose: Analyze wicket types
# ============================================================

dismissals = df['Dismissal Type'].value_counts()

dismissals.plot(kind='bar', color='purple')
plt.title("Dismissal Types")
plt.xticks(rotation=45)
plt.tight_layout()  
plt.show()

# ============================================================
# 10. EXTRAS BY TEAM
# Purpose: Compare extra runs given by teams
# ============================================================

extras = df.groupby('Bowling Team')['Extras'].sum()

extras.plot(kind='bar', color='orange')
plt.title("Extras by Team")
plt.xticks(rotation=90)
plt.tight_layout()  
plt.show()

# ============================================================
# 11. MI vs CSK COMPARISON
# Purpose: Compare two top teams
# ============================================================

teams = ['Mumbai Indians', 'Chennai Super Kings']
mi_csk = df[df['Batting Team'].isin(teams)]

compare = mi_csk.groupby('Batting Team')['Total Runs'].sum()

compare.plot(kind='bar', color=['blue','yellow'])
plt.title("MI vs CSK Comparison")
plt.tight_layout() 
plt.show()

# ============================================================
# 12. TEAM PERFORMANCE OVER YEARS (TOP 5 TEAMS ONLY)
# Purpose: To compare performance of top 5 teams over seasons
# ============================================================

# Get top 5 teams based on total runs
top_teams = df.groupby('Batting Team')['Total Runs'].sum().sort_values(ascending=False).head(5).index

# Filter dataset
df_top = df[df['Batting Team'].isin(top_teams)]

# Extract year
df_top['Year'] = pd.to_datetime(df_top['Date']).dt.year

# Aggregate runs by team and year
team_year = df_top.groupby(['Year', 'Batting Team'])['Total Runs'].sum().reset_index()

plt.figure(figsize=(12,6))

sns.lineplot(data=team_year, x='Year', y='Total Runs', hue='Batting Team', marker='o')

plt.title("Top 5 Teams Performance Over Years", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Total Runs")
plt.legend(title="Teams")
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# 13. STRIKE RATE & ECONOMY
# Purpose: Evaluate performance
# ============================================================

batsman = df.groupby('Batting Team').agg({
    'Total Runs':'sum',
    'Ball Number':'count'
}).reset_index()

batsman['Strike Rate'] = (batsman['Total Runs'] / batsman['Ball Number']) * 100
print(batsman)

bowler = df.groupby('Bowling Team').agg({
    'Total Runs':'sum',
    'Ball Number':'count'
}).reset_index()

bowler['Overs'] = bowler['Ball Number'] / 6
bowler['Economy'] = bowler['Total Runs'] / bowler['Overs']
print(bowler)

# ============================================================
# 14. Z-TEST
# Purpose: Statistical significance
# ============================================================

success = len(df[df['Total Runs'] > 0])
total = len(df)

z_stat, p_value = proportions_ztest(success, total, value=0.5)
print("Z-Test p-value:", p_value)

# ============================================================
# 15. MACHINE LEARNING
# Purpose: Predict run category
# ============================================================

data = df[['Batting Team','Bowling Team','Total Runs']].dropna()

data['Run Category'] = data['Total Runs'].apply(lambda x: 1 if x >= 4 else 0)
le = LabelEncoder()
for col in ['Batting Team','Bowling Team']:
    data[col] = le.fit_transform(data[col])

X = data[['Batting Team','Bowling Team']]
y = data['Run Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(class_weight='balanced')
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

# ============================================================
# 16. CONFUSION MATRIX
# Purpose: Evaluate model
# ============================================================

y_pred = model.predict(X_test)
from sklearn.metrics import classification_report
print("\nClassification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()