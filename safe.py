import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score


# Load the dataset
df = pd.read_csv('safetytips.csv')

# Ensure there are no NaN values in 'place' and 'season'
df['place'] = df['Place'].fillna('').astype(str)
df['season'] = df['Season'].fillna('').astype(str)
df['tips'] = df['Tips'].fillna('')

# Combine 'place' and 'season' as input features
df['combined'] = df['place'] + ' ' + df['season']

# Split data into features and target
X = df['combined']
y = df['tips']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a pipeline for vectorization and classification
pipeline = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression()
)

# Train the model
pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = pipeline.predict(X_test)

# Calculate accuracy
f1 = f1_score(y_test, y_pred, average='weighted')
#print(f'F1 Score: {f1:.2f}')


# Function to get safety tips for a given state
def get_safety_tips(state, df, pipeline):
    seasons = df['Season'].unique()
    tips_output = {}
    for season in seasons:
        input_combined = f"{state} {season}"
        predicted_tips = pipeline.predict([input_combined])
        tips_output[season] = predicted_tips[0]
    return tips_output

# Test the model with a new input
user_input = input("enter destination:")
tips_for_state = get_safety_tips(user_input, df, pipeline)

# Display the tips for each season
for season, tips in tips_for_state.items():
    print(f"\n{season} Tips:")
    print(tips)
