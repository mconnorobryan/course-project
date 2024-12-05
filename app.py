import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Function to fetch and process data
def fetch_sector_data(sector_etfs, start_date, end_date):
    sector_data = {}
    for sector, ticker in sector_etfs.items():
        data = yf.download(ticker, start=start_date, end=end_date)
        data['Sector'] = sector
        sector_data[sector] = data
    combined_df = pd.concat(sector_data.values(), keys=sector_data.keys())
    combined_df.reset_index(inplace=True)
    combined_df.rename(columns={'index': 'Date'}, inplace=True)
    return combined_df

# Streamlit UI
st.title("Sector Performance Analysis")
st.sidebar.header("User Input")

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-01-01"))

sector_etfs = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Utilities": "XLU",
    "Energy": "XLE",  # Energy Select Sector SPDR Fund
    "Transportation": "IYT",  # iShares Transportation Average ETF
    "Real Estate": "XLRE",  # Real Estate Select Sector SPDR Fund
    "Retail": "XRT"  # SPDR S&P Retail ETF
}  # Close this dictionary properly


# Sidebar inputs
selected_sectors = st.sidebar.multiselect(
    "Select sectors to analyze:",
    options=list(sector_etfs.keys()),  # Dynamically generate list from the keys
    default=["Technology", "Healthcare"]  # Default selections
)

# Filter selected sectors
filtered_etfs = {sector: sector_etfs[sector] for sector in selected_sectors}

# Fetch data based on user input
if filtered_etfs:
    st.write(f"Analyzing data for: {', '.join(selected_sectors)}")
    combined_df = fetch_sector_data(filtered_etfs, start_date, end_date)

    # Calculate metrics
    combined_df['Daily Return'] = combined_df['Adj Close'].pct_change()
    combined_df['Cumulative Return'] = (1 + combined_df['Daily Return']).cumprod()
    combined_df['Rolling Volatility'] = combined_df['Daily Return'].rolling(window=30).std()
    combined_df.dropna(inplace=True)

    # Visualizations
    st.subheader("Cumulative Returns")
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=combined_df, x='Date', y='Cumulative Return', hue='Sector')
    plt.title('Cumulative Return of Selected Sectors')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    st.pyplot(plt)

    st.subheader("Correlation Heatmap")
    sector_corr = combined_df.pivot_table(values='Daily Return', index='Date', columns='Sector').corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(sector_corr, annot=True, cmap='coolwarm')
    plt.title('Sector Return Correlation Heatmap')
    st.pyplot(plt)

    # Machine Learning
    combined_df['Lagged Return'] = combined_df['Daily Return'].shift(1)
    combined_df.dropna(inplace=True)
    X = combined_df[['Lagged Return']]
    y = combined_df['Daily Return']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Machine Learning Model Performance")
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R-squared: {r2}")

else:
    st.warning("Please select at least one sector to analyze.")

# Display historical data
st.subheader("Historical Data")

# Ensure there is data to display
if not combined_df.empty:
    # Sort data by date in descending order
    combined_df = combined_df.sort_values(by="Date", ascending=False)

    # Display the filtered data
    st.dataframe(combined_df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])

    # Add a download button
    csv = combined_df.to_csv(index=False)
    st.download_button(
        label="Download Historical Data as CSV",
        data=csv,
        file_name="historical_data.csv",
        mime="text/csv",
    )
else:
    st.write("No data available for the selected sectors and date range.")

import plotly.graph_ocbjects as go

# Add a section for visualizing data
st.subheader("Interactive Sector Performance Chart")

# Add a chart type toggle
chart_type = st.radio(
    "Select Chart Type:",
    options=["Line Chart", "Candlestick Chart"],
    index=0,  # Default is Line Chart
)

# Display the selected chart
if chart_type == "Line Chart":
    st.line_chart(
        combined_df.pivot_table(
            index="Date",
            columns="Sector",
            values="Adj Close"
        )
    )
elif chart_type == "Candlestick Chart":
    # Ensure only one sector is selected for candlestick charts
    if len(selected_sectors) == 1:
        selected_sector = selected_sectors[0]
        sector_data = combined_df[combined_df["Sector"] == selected_sector]

        # Create the candlestick chart
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=sector_data["Date"],
                    open=sector_data["Open"],
                    high=sector_data["High"],
                    low=sector_data["Low"],
                    close=sector_data["Close"],
                )
            ]
        )
        fig.update_layout(
            title=f"{selected_sector} Candlestick Chart",
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
        )

        # Display the chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please select only one sector to view the candlestick chart.")
