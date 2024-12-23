import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load the cleaned dataset
full_data = pd.read_csv('cleaned_full_data.csv')

# Set app title
st.title("Order Predictions and Sales Dashboard")

# Set sidebar titles
st.sidebar.title("Choose Dashboard Function")
dashboard_choice = st.sidebar.radio("Select a Dashboard Function:", [
    "Prediction Options",
    "Sales Analysis"
])

if dashboard_choice == "Prediction Options":
    # Sidebar for prediction type selection
    prediction_type = st.sidebar.selectbox("Choose Prediction Type", [
        "Sales Frequency Prediction for Top 5 Categories",
        "Sales Prediction for Most Popular Category in Each State",
        "Sales Prediction for Top 5 Categories"
    ])

    # Number of future months to predict
    future_months = 12

    if prediction_type == "Sales Frequency Prediction for Top 5 Categories":
        # Calculate order frequency for each product category
        order_frequency_by_category = (
            full_data.groupby('product_category_name_english')['order_id']
            .count()
            .reset_index()
        )

        # Identify the top 5 categories by order frequency
        top_categories = order_frequency_by_category.nlargest(5, 'order_id')['product_category_name_english'].tolist()

        for category_name in top_categories:
            # Filter data for the specific category
            category_data = full_data[full_data['product_category_name_english'] == category_name]
            category_monthly_sales = category_data.groupby(['order_year', 'order_month'])['order_id'].count().reset_index()
            
            # Create a new Date column
            category_monthly_sales['date'] = pd.to_datetime(
                category_monthly_sales['order_year'].astype(str) + '-' + 
                category_monthly_sales['order_month'].astype(str).str.zfill(2) + '-01'
            )
            
            # Set date as index and configure monthly frequency
            category_monthly_sales.set_index('date', inplace=True)
            train_data = category_monthly_sales[category_monthly_sales.index < '2018-01-01']
            test_data = category_monthly_sales[category_monthly_sales.index >= '2018-01-01']

            try:
                model = ARIMA(train_data['order_id'], order=(1, 1, 1))
                model_fit = model.fit()

                # Test and future predictions
                y_pred_test = model_fit.forecast(steps=len(test_data))
                future_index = pd.date_range(start=category_monthly_sales.index[-1] + pd.DateOffset(months=1), periods=future_months, freq='ME')
                future_predictions = model_fit.forecast(steps=future_months)

                # Plotting with Plotly
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=train_data.index, y=train_data['order_id'], mode='lines+markers', name='Training Frequency', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=test_data.index, y=test_data['order_id'], mode='lines+markers', name='Actual Frequency (Test)', line=dict(color='red')))
                fig.add_trace(go.Scatter(x=test_data.index, y=y_pred_test, mode='lines+markers', name='Predicted Frequency (Test)', line=dict(dash='dash', color='orange')))
                fig.add_trace(go.Scatter(x=future_index, y=future_predictions, mode='lines+markers', name='Future Predicted Frequency', line=dict(dash='dot', color='green')))

            except Exception as e:
                st.write(f"Error fitting ARIMA model for {category_name}: {e}")
                continue

            fig.update_layout(
                title=f'Order Prediction for {category_name.capitalize()} Category',
                xaxis_title='Date',
                yaxis_title='Transaction Count',
                hovermode="x unified"
            )
            st.plotly_chart(fig)

    elif prediction_type == "Sales Prediction for Most Popular Category in Each State":
        # Group by state and product category, summing the sales
        state_category_sales = (
            full_data.groupby(['customer_state', 'product_category_name_english'])['price']
            .sum()
            .reset_index()
        )

        # Most popular category in each state
        most_popular_categories = state_category_sales.loc[
            state_category_sales.groupby('customer_state')['price'].idxmax()
        ]

        # Time Series Analysis for Each Most Popular Category by State
        for _, row in most_popular_categories.iterrows():
            state = row['customer_state']
            category_name = row['product_category_name_english']
            category_data = full_data[(full_data['customer_state'] == state) & 
                                    (full_data['product_category_name_english'] == category_name)]
            category_monthly_sales = category_data.groupby(['order_year', 'order_month'])['price'].sum().reset_index()
            
            category_monthly_sales['date'] = pd.to_datetime(
                category_monthly_sales['order_year'].astype(str) + '-' + 
                category_monthly_sales['order_month'].astype(str).str.zfill(2) + '-01'
            )

            train_data = category_monthly_sales[category_monthly_sales['date'] < '2018-01-01'].copy()
            test_data = category_monthly_sales[category_monthly_sales['date'] >= '2018-01-01'].copy()

            train_data['time_index'] = range(len(train_data))
            test_data['time_index'] = range(len(train_data), len(train_data) + len(test_data))
            X_train = train_data[['time_index']]
            y_train = train_data['price']
            X_test = test_data[['time_index']]
            y_test = test_data['price']

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)

            # Future Predictions
            future_time_index = range(len(train_data) + len(test_data), len(train_data) + len(test_data) + future_months)
            future_dates = pd.date_range(start=category_monthly_sales['date'].max() + pd.DateOffset(months=1), periods=future_months, freq='MS')
            future_pred = model.predict(pd.DataFrame({'time_index': future_time_index}))

            # Plotting
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train_data['date'], y=y_train, mode='lines+markers', name='Training Sales', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=test_data['date'], y=y_test, mode='lines+markers', name='Actual Sales (Test)', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=test_data['date'], y=y_pred_test, mode='lines+markers', name='Predicted Sales (Test)', line=dict(dash='dash', color='orange')))
            fig.add_trace(go.Scatter(x=future_dates, y=future_pred, mode='lines+markers', name='Future Predictions', line=dict(dash='dash', color='green')))

            fig.update_layout(
                title=f'Sales Prediction for {category_name.capitalize()} in {state}',
                xaxis_title='Date',
                yaxis_title='Sales ($)',
                hovermode="x unified"
            )
            st.plotly_chart(fig)

    elif prediction_type == "Sales Prediction for Top 5 Categories":
        # Calculate total sales for each product category
        total_sales_by_category = (
            full_data.groupby('product_category_name_english')['price']
            .sum()
            .reset_index()
        )

        # Identify the top 5 categories by total sales
        top_categories = total_sales_by_category.nlargest(5, 'price')['product_category_name_english'].tolist()

        for category_name in top_categories:
            # Filter data for the specific category
            category_data = full_data[full_data['product_category_name_english'] == category_name]

            # Group by year and month, summing up the sales
            category_monthly_sales = category_data.groupby(['order_year', 'order_month'])['price'].sum().reset_index()

            # Create a new Date column for easy time-based indexing
            category_monthly_sales['date'] = pd.to_datetime(
                category_monthly_sales['order_year'].astype(str) + '-' + 
                category_monthly_sales['order_month'].astype(str).str.zfill(2) + '-01'
            )

            # Sort values by date
            category_monthly_sales = category_monthly_sales.sort_values(by='date').reset_index(drop=True)

            # Split data: 2016-2017 for training, 2018 for testing
            train_data = category_monthly_sales[category_monthly_sales['date'] < '2018-01-01'].copy()
            test_data = category_monthly_sales[category_monthly_sales['date'] >= '2018-01-01'].copy()

            # Set up features and target variable
            train_data['time_index'] = range(len(train_data))
            test_data['time_index'] = range(len(train_data), len(train_data) + len(test_data))
            X_train = train_data[['time_index']]
            y_train = train_data['price']
            
            # Create and train the model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make predictions for the test set
            y_pred_test = model.predict(test_data[['time_index']])

            # Future Predictions
            last_index = len(category_monthly_sales)  # Last index in the dataset
            future_time_index = range(last_index, last_index + future_months)
            future_dates = pd.date_range(start=category_monthly_sales['date'].max() + pd.DateOffset(months=1), periods=future_months, freq='MS')
            future_sales = model.predict(pd.DataFrame({'time_index': future_time_index}))

            # Plotting
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train_data['date'], y=y_train, mode='lines+markers', name='Training Sales', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=test_data['date'], y=test_data['price'], mode='lines+markers', name='Actual Sales (Test)', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=test_data['date'], y=y_pred_test, mode='lines+markers', name='Predicted Sales (Test)', line=dict(dash='dash', color='orange')))
            fig.add_trace(go.Scatter(x=future_dates, y=future_sales, mode='lines+markers', name='Future Sales Prediction', line=dict(dash='dot', color='green')))

            fig.update_layout(
                title=f'Sales Prediction for {category_name.capitalize()} Category',
                xaxis_title='Date',
                yaxis_title='Sales ($)',
                hovermode="x unified"
            )
            st.plotly_chart(fig)

elif dashboard_choice == "Sales Analysis":
    st.sidebar.title("Sales Analysis Options")

    # Sidebar for sales visualization selection
    sales_analysis_type = st.sidebar.radio("Choose Analysis Type:", [
        "Monthly Sales by Category",
        "Yearly Sales by Category"
    ])

    # Sidebar to choose whether to display top 5 products by sales
    top_5_option = st.sidebar.checkbox("Show Top 5 Products by Sales", value=False)

    # Sidebar to select chart type
    chart_type = st.sidebar.radio("Select Chart Type:", ["Line Chart", "Bar Chart"])

    # Filter the data based on the top 5 option
    if top_5_option:
        # Group by 'product_category_name_english' and sum the prices for all categories
        top_5_categories = full_data.groupby('product_category_name_english')['price'].sum().reset_index()

        # Sort by price in descending order and get the top 5 categories
        top_5_categories = top_5_categories.nlargest(5, 'price')

        # Sidebar to select category from the top 5
        sorted_categories = sorted(top_5_categories['product_category_name_english'].tolist())
        selected_category = st.sidebar.selectbox("Select Product Category", sorted_categories)

        # Filter data for the selected category
        category_data = full_data[full_data['product_category_name_english'] == selected_category]
    else:
        # If the top 5 option is not checked, display all categories
        categories = full_data['product_category_name_english'].dropna().unique().tolist()

        # Sidebar to select category from all categories
        sorted_categories = sorted(categories)
        selected_category = st.sidebar.selectbox("Select Product Category", sorted_categories)


        # Filter data for the selected category
        category_data = full_data[full_data['product_category_name_english'] == selected_category]

    # Perform sales analysis based on the selected type
    if sales_analysis_type == "Monthly Sales by Category":
        # Group by year and month, summing up the sales
        monthly_sales = category_data.groupby(['order_year', 'order_month'], as_index=False)['price'].sum()

        # Ensure order_month is formatted correctly as a two-digit string
        monthly_sales['order_month'] = monthly_sales['order_month'].astype(str).str.zfill(2)

        # Create a new Date column for plotting
        monthly_sales['date'] = pd.to_datetime(
            monthly_sales['order_year'].astype(str) + '-' + monthly_sales['order_month'] + '-01'
        )

        # Plotting based on chart type
        if chart_type == "Line Chart":
            fig = px.line(
                monthly_sales,
                x='date',
                y='price',
                title=f"Monthly Sales for {selected_category.capitalize()}",
                labels={'price': 'Sales ($)', 'date': 'Date'}
            )
        elif chart_type == "Bar Chart":
            fig = px.bar(
                monthly_sales,
                x='date',
                y='price',
                title=f"Monthly Sales for {selected_category.capitalize()}",
                labels={'price': 'Sales ($)', 'date': 'Date'},
                text_auto=True
            )
        st.plotly_chart(fig)

    elif sales_analysis_type == "Yearly Sales by Category":
        # Group by year, summing up the sales
        yearly_sales = category_data.groupby('order_year', as_index=False)['price'].sum()

        # Plotting based on chart type
        if chart_type == "Line Chart":
            fig = px.line(
                yearly_sales,
                x='order_year',
                y='price',
                title=f"Yearly Sales for {selected_category.capitalize()}",
                labels={'price': 'Sales ($)', 'order_year': 'Year'}
            )
        elif chart_type == "Bar Chart":
            fig = px.bar(
                yearly_sales,
                x='order_year',
                y='price',
                title=f"Yearly Sales for {selected_category.capitalize()}",
                labels={'price': 'Sales ($)', 'order_year': 'Year'},
                text_auto=True
            )
        st.plotly_chart(fig)
