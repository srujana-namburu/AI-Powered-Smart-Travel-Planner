import pandas as pd
from sklearn.neighbors import NearestNeighbors
from fastapi import FastAPI, Query
from sklearn.preprocessing import MinMaxScaler

# Load dataset (replace with your dataset file path)
# Example dataset columns: Hotel_Name, Location, Hotel_Price, Hotel_Rating, DistanceFromCenter
hotels_df = pd.read_csv("hotels_india.csv")

# Preprocessing
hotels_df['Hotel_Price'] = pd.to_numeric(hotels_df['Hotel_Price'], errors='coerce')
hotels_df['Hotel_Rating'] = pd.to_numeric(hotels_df['Hotel_Rating'], errors='coerce')

# Fill missing values
hotels_df.fillna({
    'Hotel_Price': hotels_df['Hotel_Price'].mean(),
    'Hotel_Rating': hotels_df['Hotel_Rating'].mean()
}, inplace=True)


# Define weights for features
FEATURE_WEIGHTS = {'Hotel_Price': 0.5, 'Hotel_Rating': 0.4, 'DistanceFromCenter': 0.1}
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(hotels_df[['Hotel_Price','Hotel_Rating']])
normalized_df = pd.DataFrame(normalized_features,columns=['Normalized_Price','Normalized_Rating'])


# Compute a score for each hotel
hotels_df['Score'] = (
    FEATURE_WEIGHTS['Hotel_Price'] * (1 - normalized_df['Normalized_Price']) +  # Lower Hotel_Price is better
    FEATURE_WEIGHTS['Hotel_Rating'] * normalized_df['Normalized_Rating'])  # Closer is better
hotels_df['Score'] = hotels_df['Score'].fillna(0)
# API Setup with FastAPI
app = FastAPI()

@app.get("/recommend-hotels")
def recommend_hotels(location: str = Query(...), top_n: int = Query(5)):
    # Filter hotels by location
    filtered_hotels = hotels_df[hotels_df['City'].str.contains(location, case=False)]
    
    if filtered_hotels.empty:
        return {"message": f"No hotels found for location: {location}"}
    
    # Sort by score
    recommendations = filtered_hotels.sort_values(by='Score', ascending=False).head(top_n)
    
    # Return results
    return recommendations[['Hotel_Name', 'Hotel_Price', 'Hotel_Rating','Score']].to_dict(orient='records')

# Run API (use `uvicorn script_name:app --reload` to test locally)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8017)