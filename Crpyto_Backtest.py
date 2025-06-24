#So here we are going to do a cryptocurrency mean reversion strategy backtesting
#as we know the crpytocurrency markets, a way to profit from the wild oscillations of the prices is mean reversion, where the prices tend to revert to an average level over time
#for implementing this mean reverting strategy, there is the Ornstein-Uhlenbeck mean reversion process which means that we can go on and apply a mathematical framework for implementing the mean reversion
#so then we have to perform backtests to see if the mean rev strategy is performing well or not, so that we should retrieve high-frequency data, but then backtesting iteration by iteration can be slow so we have vectorized bactesting
#so essentially we are going to have trading signals from the OU model, and then we are going to backtest the trading strategy based on these so that we can see after how it performed

#the ornstein uhlenbeck process and parameter estimation
#here this process is going to have a tendency to revert to the average value overtime so that we can determine signals and then trade on them based on this 
#the model is decrived by a stochastic differential equation which is


# The Ornstein-Uhlenbeck process is defined by the following stochastic differential equation (SDE):
#     dX(t) = θ(μ - X(t))dt + σ dW(t)


#here 
# X(t): this is the value of the process, the cryptocurrency price at time t
# μ: this is the long-term mean level of the crypto price, this is the value towards which the value will revert or converge towards on the long term
# θ: this is the speed of the reversion, which means that a higher value for this variable will mean that the process reverts more quickly to the mean after the deviation, this can be thought of as the strength of the reversion
# σ: this is the volatility or the magnitude of the random fluctuations, where this represents the noise or the randomness inherent in the price movements, so 
# dW(t): this is going to represent the brownian motion term, where this is the random shock at each movement, so that with a certain volatility we are going to have a random movement every time, and then it will als
# in summary we are going to have the where the theta is assumed to be greater than zero, , so that as the time will approach infinity, then the deterministic part of the price will tend to the mean level

#all the explanations and the concepts to understand what is happening with the equation and the drift term is there in the word document so that we can understand it

# essentially we are going to use this mean reverting strategy

#now importing everything and then retrieving the data

#core idea: 
#the exchange is going to be ccxt.binance()
#frequency = hourly
#limit the number of candles to fetch from the database
# ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
# df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
# df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
# df.set_index('timestamp', inplace=True)
# price_series = df['close'] # We'd typically use this series
# log_price_series = np.log(df['close']) # Or the log prices

#stationary testing
#the model can be inappropriate, because it can be non mean reverting, so that we assume that it is not just a random process, which is going to be the case if theta will be zero because then the only term that remains is the brownian motion term
#if the OU process generates a stationary time series, which happens when the theta is greater than zero, then the mean and variance will not change overtime 
# so then if we assume that theta is not zero and there is actually a reversion, but it does not happen, then there will not be a mean reversion and then the values we get will not be meaningful

#TESTING STATIONARITY, which means that this is a test to check whether the mean and the variance is constant over time so that will it return to the same mean value or is it going to change overtime so that we have a changing autocorrelation structure: 
##### Augmented Dickey Fuller test: this is the most common test for stationarity, which means that 
# null hypothesis: series has a unit root: non-stationary
# alternative hypothesis: series is stationary
# we assume that it will return to an average but if it goes off indefenitely the crypto price, then the assumptions are violated and the parameter estimates are useless

# if the p value is below 5%, then we reject the null so that it is non stationary so that it will be stationary and we can use the mean reversion method
#if the p value is high, then the series will be high and it is likely nonstationary, so that OU cannot be applied and mean reversion neither

# The Ornstein-Uhlenbeck (OU) process is defined by the following stochastic differential equation (SDE):
#     dX(t) = θ(μ - X(t))dt + σ dW(t)

#we will estimate the OU parameters 
#for small time changes we can estimate delta x(t) as  
# ΔX = X(t) - X(t - Δt)
# This represents the discrete difference of the process X over a time interval Δt
#written down in the word doc
#delta t will mean the time intervals between the retrieved data 

#these OLS estimates will provide us with good estimates, we could have used the MLE method but it is often more complex

#data acquisition where using log prices is better for stationarity and more accurate

import streamlit as st
import ccxt
import pandas as pd 
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt

# --- Data Acquisition ---

time_inp = st.selectbox("Please input the frequency with which you would like to retrieve the data so delta t (options: 1h, 4h, 1d)", options=['1h', '4h', '1d'], index=0)
exch_inp = st.sidebar.text_input("Please input the specified exchange from which you would like to fetch the data", value="binance")
symb_inp = st.sidebar.text_input("Please input the symbol of the crypto you would like to see from the specified exchange", value='BTC/USDT')
limit_inp = st.sidebar.number_input("Please give us the number of candles you would like to dowload with the timesteps", value=2000)

use_live_data = st.checkbox("Use latest data", value=True)
if use_live_data:
    end_date = datetime.datetime.now(datetime.timezone.utc)
else:
    end_date = datetime.datetime(2025, 4, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)

########################################################################################################################################################################################################################
def download_crypto_data(symbol= symb_inp, timeframe = time_inp, limit=limit_inp, exchange_id = exch_inp):
    """
    Downloads historical OHLCV data for a cryptocurrency pair from a specified exchange.

    Args:
        symbol (str): The trading pair symbol (e.g., 'BTC/USDT').
        timeframe (str): The timeframe for the data (e.g., '1h', '4h', '1d').
        limit (int): The number of data points (candles) to fetch.
        exchange_id (str): The ID of the exchange (e.g., 'binance', 'coinbasepro').

    Returns:
        pandas.DataFrame: DataFrame containing OHLCV data indexed by timestamp,
                          or None if fetching fails.
    """
    st.write(f"Attempting to download {limit} {timeframe} candles for {symbol} from {exchange_id}...")
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class()
        # Ensure the time is consistent, fetch up to the current time minus one interval
        # Calculate the 'since' timestamp to get data up to roughly April 1, 2025
        # Note: ccxt fetches backwards from the current time by default if 'since' is not provided
        # To get data *ending* around a specific past date, we need to make multiple requests or use specific exchange features
        # For simplicity, we fetch the latest 'limit' candles available now.
        # If specific end date is critical, more complex logic using 'since' and loops is needed.
        # We'll fetch the most recent `limit` candles ending before the current time.
        # Fetching up to a specific date like April 1, 2025 requires knowing the timestamp for that date.
        end_timestamp = int(end_date.timestamp() * 1000)

        # Fetch data ending at the specified timestamp
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit, params={'endTime': end_timestamp})

        if not ohlcv:
            st.write("No data returned from the exchange.")
            return None

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True) # Use UTC
        df.set_index('timestamp', inplace=True)

        # Ensure data doesn't exceed the target end date (sometimes exchanges return slightly more)
        df = df[df.index < end_date]

        # Remove potential duplicates and sort
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)

        st.write(f"Successfully downloaded {len(df)} data points ending {df.index.max()}.")
        return df

    except ccxt.ExchangeError as e:
        st.write(f"Exchange Error: {e}")
        return None
    except ccxt.NetworkError as e:
        st.write(f"Network Error: {e}")
        return None
    except Exception as e:
        st.write(f"An unexpected error occurred: {e}")
        return None

#so all in all this function that we wrote will connect to a specified exchange and fill fetch the last few candles for the symbol and timeframe

########################################################################################################################################################################################################################
#stationary testing
st.subheader("Now we are testing whether the crypto price series actually exhibits the stationarity behavior and if the mean reversion strategy can be applied")

# --- Stationarity Testing ---

def test_stationarity(timeseries, series_name="Time Series"):
    """
    Performs the Augmented Dickey-Fuller (ADF) test for stationarity.

    Args:
        timeseries (pd.Series): The time series data to test.
        series_name (str): Name of the series for printing results.
    """
    st.write(f'\nResults of Augmented Dickey-Fuller Test for {series_name}:')
    # Handle potential NaN values by dropping them
    timeseries_clean = timeseries.dropna()
    if timeseries_clean.empty:
        st.write("Time series is empty after dropping NaN values. Cannot perform ADF test.")
        return

    dftest = adfuller(timeseries_clean, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value

    st.write(dfoutput)
    # Interpretation
    if dftest[1] <= 0.05:
        st.write(f"Conclusion: p-value ({dftest[1]:.4f}) is less than or equal to 0.05. Reject the null hypothesis.")
        st.write(f"The '{series_name}' series is likely stationary.")
    else:
        st.write(f"Conclusion: p-value ({dftest[1]:.4f}) is greater than 0.05. Fail to reject the null hypothesis.")
        st.write(f"The '{series_name}' series is likely non-stationary.")

#in summary this test is going to take a pandas series and run the ADF test 
#then it will print the key results , the test statistics, the p value etc. 
#if the p value will be smaller than 0.05 then we reject nonstationary so that it is likely to be stationary and the data might be suitable

#now we are going for the parameter estimation
#so then if the data looks reasonably stationary, then we can go on and extract the values that we actually estimated with the OLS method where we are going to use the discretized form of the OU equation
# Discretized Ornstein-Uhlenbeck process:
# X(t) ≈ (1 - θ * Δt) * X(t - Δt) + θ * μ * Δt + error

########################################################################################################################################################################################################################
# --- OU Parameter Estimation (OLS) ---

def estimate_ou_parameters_ols(price_series, dt=1):
    """
    Estimates Ornstein-Uhlenbeck parameters (theta, mu, sigma) using OLS
    on the discretized equation: dX = theta * (mu - X) * dt + sigma * dW
    which leads to the regression: X_{t+1} = alpha + beta * X_t + epsilon_t

    Args:
        price_series (pd.Series): Time series of prices (or log prices).
        dt (float): Time step between observations (e.g., 1 for discrete steps,
                    or 1/(24*365) for hourly data in annualized terms).
                    Here we use dt=1, interpreting parameters per time step (e.g., per hour).

    Returns:
        tuple: Estimated (theta, mu, sigma), or (None, None, None) if estimation fails.
    """
    # Ensure no NaNs
    price_series = price_series.dropna()
    if len(price_series) < 2:
        st.write("Error: Need at least two data points for OLS estimation.")
        return None, None, None

    # Prepare lagged and differenced series for OLS
    X_t = price_series.shift(1).dropna() # Drop the first NaN created by shift
    # Ensure X_t1 aligns with X_t after dropping NaNs
    X_t1 = price_series[X_t.index] # Use the index from X_t to align X_{t+1}

    # Add a constant for the intercept term in OLS
    X_t_with_const = sm.add_constant(X_t)

    # Perform OLS regression: X_{t+1} ~ const + X_t
    try:
        model = sm.OLS(X_t1, X_t_with_const)
        results = model.fit()

        # Extract regression coefficients
        intercept, slope = results.params
        residuals = results.resid

        # Calculate OU parameters from OLS results
        # slope = (1 - theta * dt) => theta = (1 - slope) / dt
        theta = (1 - slope) / dt

        # intercept = theta * mu * dt => mu = intercept / (theta * dt)
        # Handle potential division by zero if theta is close to zero
        if theta * dt == 0:
             st.write("Warning: Estimated theta*dt is zero, cannot calculate mu reliably.")
             mu = np.nan # Or handle appropriately, maybe mean of series?
        else:
            mu = intercept / (theta * dt)

        # sigma is related to the standard deviation of residuals
        # std(residuals) = std(sigma * dW_t) = sigma * sqrt(dt)
        # => sigma = std(residuals) / sqrt(dt)
        # Ensure dt is positive before taking sqrt
        if dt <= 0:
            st.write("Error: dt must be positive to calculate sigma.")
            return None, None, None

        sigma = np.std(residuals) / np.sqrt(dt)

        # Adjust sigma for degrees of freedom (optional, usually minor effect for large N)
        # N = len(residuals)
        # k = X_t_with_const.shape[1] # Number of regressors (including constant)
        # sigma_adj = np.sqrt(np.sum(residuals**2) / (N - k)) / np.sqrt(dt)
        # sigma = sigma_adj # Use adjusted sigma if preferred


        st.write(f"\nOLS Regression Results for OU Parameter Estimation:")
        # print(results.summary()) # Optionally print full summary
        st.write(f"Intercept (alpha): {intercept:.6f}")
        st.write(f"Slope (beta):      {slope:.6f}")
        st.write(f"Residual Std Dev:  {np.std(residuals):.6f}")
        st.write(f"\nEstimated OU Parameters (dt = {dt}):")
        st.write(f"Theta (mean reversion speed): {theta:.6f}")
        st.write(f"Mu (long-term mean):          {mu:.6f}")
        st.write(f"Sigma (volatility):           {sigma:.6f}")

        return theta, mu, sigma

    except Exception as e:
        st.write(f"An error occurred during OLS estimation: {e}")
        return None, None, None
    
# so essentially what will happen is that this function is going to take our price series and the time step dt and then setting dt equal to 1, so that the estimated parameters will be interpreted per time step. 
#Then the function creates the lagged series and the current series, after which we regress the current one on the lagged, and then the slope of the regression and the intercept and calculate the coefficients
########################################################################################################################################################################################################################
# --- Using the Functions ---
st.title("Ornstein-Uhlenbeck Parameter Estimation")

# --- Run Button ---
if st.button("Run OU Analysis"):
    crypto_data_df = download_crypto_data(symbol=symb_inp, timeframe=time_inp, limit=limit_inp, exchange_id=exch_inp)

    if crypto_data_df is not None and not crypto_data_df.empty:
        st.success("Data downloaded successfully.")

        # Use log prices
        log_price_series = np.log(crypto_data_df['close'])

        # --- 1. Stationarity Testing ---
        test_stationarity(log_price_series, series_name=f"Log {symb_inp} Prices")
        test_stationarity(crypto_data_df['close'], series_name=f"{symb_inp} Prices")

        # --- 2. Estimate OU Parameters ---
        st.subheader("Estimating OU Parameters (per hour)")
        theta_ols, mu_ols, sigma_ols = estimate_ou_parameters_ols(log_price_series, dt=1)

        if theta_ols is not None:
            st.write(f"**Theta (mean-reversion speed)**: {theta_ols:.4f}")
            st.write(f"**Mu (long-term log price level)**: {mu_ols:.4f}")
            st.write(f"**Sigma (volatility)**: {sigma_ols:.4f}")

            if theta_ols > 0:
                half_life = np.log(2) / theta_ols
                st.write(f"**Characteristic time (1/theta):** {1/theta_ols:.2f} hours")
                st.write(f"**Half-life of mean reversion:** {half_life:.2f} hours")
            else:
                st.warning("Estimated theta is non-positive, suggesting no mean reversion.")

            # Store for later use (optional)
            ou_params = {'theta': theta_ols, 'mu': mu_ols, 'sigma': sigma_ols}
            st.session_state['log_price_series'] = log_price_series
            st.session_state['ou_params'] = ou_params

        else:
            st.error("OU parameter estimation failed.")
    else:
        st.error("Data download failed or returned empty.")
########################################################################################################################################################################################################################
#so now we can move onto generating a strategy signal for the mean reversion if the test above has passed and if it means that we can actually proceed with the implementation of the values because there is some mean reversion

#calculating the OU z-score
#so the main point is that if the theta term is positive meaning that there is some mean reversion, meaning that as we increase time, the distribution of the prices will converge to a normal distribution, so if the distribution is going to have a normal distribution, then I will be able to standardize this distribution by subtracting the mean and then seeing how many standard deviations that is from the mean value

#so essentially we are going to look at the standardized value of measuring how far away we are at a current point in time from the mean value to which we are going to converge to
#if there is a great deviation then it means that we can trade on it and short or long based on a negative or positive z score

# Assume 'log_price_series' and 'ou_params' are available from Section 2 execution
# If running sections independently, ensure these are loaded or recalculated.
# Example placeholder if previous section's variables aren't in the environment:
# log_price_series = pd.Series(...) # Load or create example data
# ou_params = {'theta': 0.008, 'mu': 11.35, 'sigma': 0.0065}


def calculate_ou_zscore(log_price_series, ou_params):
    """
    Calculates the OU-based z-score using the stationary distribution's std dev.

    Args:
        log_price_series (pd.Series): Time series of log prices.
        ou_params (dict): Dictionary containing 'theta', 'mu', 'sigma'.

    Returns:
        pd.Series: The calculated OU z-score, or None if calculation fails.
    """
    theta = ou_params.get('theta')
    mu = ou_params.get('mu')
    sigma = ou_params.get('sigma')

    if theta is None or mu is None or sigma is None:
        st.write("Error: Missing OU parameters.")
        return None
    if theta <= 0 or sigma <= 0:
        st.write("Error: Theta and Sigma must be positive for z-score calculation.")
        st.write(f"Current values: theta={theta}, sigma={sigma}")
        # If theta is non-positive, the stationary distribution variance is undefined or infinite.
        # Returning NaNs or handling this case based on strategy logic.
        # For now, we return None to indicate failure.
        return None
    if (2 * theta) <= 0:
         st.write("Error: 2 * theta must be positive for sqrt.")
         return None


    # Calculate stationary standard deviation: sigma_eq = sigma / sqrt(2 * theta)
    try:
        sigma_eq = sigma / np.sqrt(2 * theta)
    except ValueError:
         st.write("Error: Cannot calculate square root of non-positive value (2*theta).")
         return None


    if sigma_eq == 0:
         st.write("Error: Calculated stationary standard deviation is zero.")
         return None

    # Calculate z-score: (price - mu) / sigma_eq
    z_score = (log_price_series - mu) / sigma_eq
    return z_score

#so essentially with this we took the log prices and the parameter estimates so that then we ensure we can calculate with those and that theta is the assumed greater than 0, 
########################################################################################################################################################################################################################
#so then now we have to define the trading rules so that we can backtest the strategy
#the classic rule is:
# 1. enter short if the price moves significantly above the mean (if the z score exceeds +1.5) so that we bet on it reverting to the mean from above and decrease
# 2. enter a long position if the price moves significantly below the mean, which means that the z-score drops below a negative treshold (-1.5), so that we bet on reverting upward
# 3. exit the position when the price reverts back towards the mean, so when z-score crosses back over an exit threshold, (z-score becomes close to 0)

#here we have to generate a singla generation, meaning that generate a series of signals
# +1 means should be Long in the next period
# -1 means should be Short in the next period
# 0 means we should be flat in the next period

#we should also go on and avoid lookahead bias, where decisions can only be made with information available at the time

def generate_ou_signals(log_price_series, ou_params, entry_threshold=1.5, exit_threshold=0.0):
    """
    Generates trading signals based on OU z-score crossing thresholds.

    Args:
        log_price_series (pd.Series): Time series of log prices.
        ou_params (dict): Dictionary containing OU parameters ('theta', 'mu', 'sigma').
        entry_threshold (float): Z-score level to trigger entry (positive for sell, negative for buy).
        exit_threshold (float): Z-score level to trigger exit (closer to zero).

    Returns:
        pd.DataFrame: DataFrame containing log prices, z-score and trading signals (1: long, -1: short, 0: flat).
                     Returns None if inputs are invalid.
    """
    if log_price_series is None or ou_params is None:
        print("Error: log_price_series or ou_params not provided.")
        return None

    df = pd.DataFrame({'log_price': log_price_series})

    # Calculate z-score
    df['z_score'] = calculate_ou_zscore(df['log_price'], ou_params)

    if df['z_score'].isnull().all():
        print("Error: Z-score calculation failed or resulted in all NaNs.")
        return None

    # Ensure thresholds have correct signs
    entry_threshold = abs(entry_threshold)
    exit_threshold = abs(exit_threshold)

    # Generate signals based on *previous* z-score to avoid lookahead bias
    # Signal determines position for the *current* period based on *previous* period's z-score
    df['signal'] = 0  # Default to flat

    # Conditions based on previous z-score
    prev_z = df['z_score'].shift(1)

    # Entry conditions
    buy_entry_condition = (prev_z < -entry_threshold)
    sell_entry_condition = (prev_z > entry_threshold)

    # Exit conditions
    # Exit long if previously long AND z-score crossed above exit threshold
    buy_exit_condition = (df['signal'].shift(1) == 1) & (prev_z >= -exit_threshold)
    # Exit short if previously short AND z-score crossed below exit threshold
    sell_exit_condition = (df['signal'].shift(1) == -1) & (prev_z <= exit_threshold)

    # Apply signals vectorially - This requires careful state management.
    # Simpler approach: Assign entry signals, then propagate state, then apply exits.
    # Let's try the state propagation method using ffill.

    # 1. Raw entry signals: +1 for buy, -1 for sell, NaN otherwise
    df['raw_signal'] = np.nan
    df.loc[buy_entry_condition, 'raw_signal'] = 1
    df.loc[sell_entry_condition, 'raw_signal'] = -1

    # 2. Propagate signals to represent holding position
    # Fill NaN with the previous valid signal (forward fill)
    # Initial state is flat (0)
    df['position'] = df['raw_signal'].ffill().fillna(0)

    # 3. Generate exit signals (set position to 0)
    # If was long (position==1) and exit condition met -> go flat (0)
    df.loc[(df['position'].shift(1) == 1) & (prev_z >= -exit_threshold), 'position'] = 0
     # If was short (position==-1) and exit condition met -> go flat (0)
    df.loc[(df['position'].shift(1) == -1) & (prev_z <= exit_threshold), 'position'] = 0


    # The 'position' column now represents the desired position for the current period
    # based on information available at the end of the previous period.
    # Let's rename 'position' to 'signal' for consistency with project outline
    df['signal'] = df['position']

    # Drop intermediate columns
    df = df.drop(columns=['raw_signal', 'position'])

    # Remove initial NaN row created by shift
    df = df.iloc[1:]

    return df

#so all in all we are going to calculate the z scores, then use the entry and exit decisions, and then with the ffill() it is going to carry forward the last entry signal and effectively simulate holding the position
########################################################################################################################################################################################################################
#now we are going to visualize and see an example of using this

st.subheader("Step 3: Generate Mean Reversion Trading Signals")
st.write("So here, the trading signals mean that if there is a:") 
st.write("+1: then we are long in the position, the number after that means we are long for that amount of hours") 
st.write("0: this means we are flat in position, have no holdings") 
st.write("-1: means we are short in the position, the number indicating the amount of hours we were in a short position")

entry_thresh = st.slider("Entry Threshold (Z-Score)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
exit_thresh = st.slider("Exit Threshold (Z-Score)", min_value=0.0, max_value=2.0, value=0.0, step=0.1)

if st.button("Generate Signals"):
    log_price_series = st.session_state.get('log_price_series')
    ou_params = st.session_state.get('ou_params')
    if log_price_series is not None and ou_params is not None:
        st.success("Generating trading signals...")

        signals_df = generate_ou_signals(log_price_series, ou_params,
                                         entry_threshold=entry_thresh,
                                         exit_threshold=exit_thresh)

        if signals_df is not None:
            st.session_state['signals_df'] = signals_df  # Save signals_df for later use
            st.write("Signals generated:")
            st.write(signals_df.head())

            st.write("Signal Distribution:")
            st.write(signals_df['signal'].value_counts())

            # Plotting
            st.subheader("Signal Visualization")

            fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

            # Plot 1: Log Price and Buy/Sell Signals
            axes[0].plot(signals_df.index, signals_df['log_price'], label='Log Price', color='skyblue', lw=1.5)
            axes[0].set_ylabel('Log Price')
            axes[0].set_title('Log Price with Trading Signals')

            # Plot buy signals
            buy_points = signals_df[(signals_df['signal'] == 1) & (signals_df['signal'].shift(1) != 1)]
            axes[0].plot(buy_points.index, buy_points['log_price'], '^', color='lime', markersize=8, label='Buy')

            # Plot sell signals
            sell_points = signals_df[(signals_df['signal'] == -1) & (signals_df['signal'].shift(1) != -1)]
            axes[0].plot(sell_points.index, sell_points['log_price'], 'v', color='red', markersize=8, label='Sell')

            axes[0].legend()

            # Plot 2: Z-score with thresholds
            axes[1].plot(signals_df.index, signals_df['z_score'], label='Z-Score', color='grey', lw=1.5)
            axes[1].axhline(entry_thresh, color='red', linestyle='--', label='Entry Threshold')
            axes[1].axhline(-entry_thresh, color='lime', linestyle='--', label='Entry Threshold')
            axes[1].axhline(exit_thresh, color='orange', linestyle=':', label='Exit Threshold')
            axes[1].axhline(-exit_thresh, color='orange', linestyle=':', label='Exit Threshold')
            axes[1].set_ylabel('Z-Score')
            axes[1].set_title('Z-Score with Entry/Exit Thresholds')
            axes[1].legend()

            st.pyplot(fig)
        else:
            st.error("Signal generation failed.")
    else:
        st.warning("Please run the previous steps to estimate OU parameters and compute log prices.")
########################################################################################################################################################################################################################
# --- Vectorized Backtesting Engine Implementation ---

#now in this section we are going to look at how this strategy has performed historically
#there is vectorized backtesting that is better than implementing loops because

#doing it with vectorization, and not with loops, which is much faster and easier because it means that there is less code and can be done with a single line
#the core idea is that we represent signals, then positions and returns as series and then there is going to be a mathematical formula applied through these arrays at one single time simultaneously meaning that the calculation is going to be done in one go
#this will lead to more readable code and faster execution

st.subheader("Vectorized Backtesting")
st.write("The objectives are:")
st.write("1. Calculate the asset's periodic log returns")
st.write("2. Determine the strategy's returns before costs (raw returns) by multiplying the position held during a period by the asset's return for that period, where the code already accounts for lookahead bias")
st.write("3. Calculate transaction costs (commission and slippage(difference between the actual price sold and the planned)) whenever a trade happens")
st.write("4. Subtract the costs to get the net strategy returns")
st.write("5. Compute the cumulative performance for the strategy (net and raw) and compare it against a simple Buy & Hold benchmark")

comm_inp = st.number_input("Please determine the commission rate:", value=0.001, min_value=0.0, max_value=0.01, step=0.0001, format="%.4f")
slip_inp = st.number_input("Please determine the slippage cost rate:", value=0.0005, min_value=0.0, max_value=0.002, step=0.0001, format="%.4f")

def run_vectorized_backtest(signals_df, commission_rate=comm_inp, slippage_rate=slip_inp):
    """
    Performs a purely vectorized backtest of the trading strategy, including transaction costs.

    Args:
        signals_df (pd.DataFrame): DataFrame with 'log_price' and 'signal' columns.
                                   Signal determines the position for the *next* period.
        commission_rate (float): Proportional commission cost per trade (fraction of value).
        slippage_rate (float): Proportional slippage cost per trade (fraction of value).

    Returns:
        pd.DataFrame: DataFrame containing detailed backtesting results including returns,
                      costs and equity curves. Returns None if input is invalid.
    """
    if signals_df is None or not isinstance(signals_df, pd.DataFrame) or \
       'log_price' not in signals_df or 'signal' not in signals_df:
        st.write("Error: Invalid or incomplete signals_df provided for backtesting.")
        return None
    if signals_df.empty:
        st.write("Error: signals_df is empty.")
        return None

    # Make a copy to prevent modifications to the original DataFrame
    backtest_df = signals_df.copy()

    # --- 1. Calculate Asset Log Returns ---
    # log_returns[t] = log_price[t] - log_price[t-1]
    backtest_df['log_returns'] = backtest_df['log_price'].diff()

    # --- 2. Calculate Raw Strategy Log Returns (Ignoring Costs) ---
    # The signal generated at time t-1 determines the position held during the interval (t-1, t].
    # The return for this interval is log_returns[t].
    # Strategy return = position[t] * asset_return[t]
    # Since 'signal' column already represents the position for the period, no further shift needed here.
    backtest_df['raw_strategy_returns'] = backtest_df['signal'] * backtest_df['log_returns']

    # --- 3. Calculate Transaction Costs (Vectorized) ---
    # Costs are incurred when the position changes from t-1 to t.
    # Calculate the change in position size (absolute value indicates trade magnitude)
    previous_signal = backtest_df['signal'].shift(1).fillna(0) # Assume starting position is flat (0)
    position_change = (backtest_df['signal'] - previous_signal).abs()

    # Calculate costs as a fraction of the return/price movement.
    # This is an approximation; more complex models exist, but this is common for vectorized backtests.
    # Assume costs are subtracted directly from the log returns for simplicity.
    backtest_df['commission_cost'] = position_change * commission_rate
    backtest_df['slippage_cost'] = position_change * slippage_rate
    backtest_df['total_costs'] = backtest_df['commission_cost'] + backtest_df['slippage_cost']

    # --- 4. Calculate Net Strategy Log Returns ---
    # Subtract total costs from the raw strategy returns for the period.
    backtest_df['net_strategy_returns'] = backtest_df['raw_strategy_returns'] - backtest_df['total_costs']

    # --- Clean up initial NaN values ---
    # Drop the first row where log_returns is NaN due to diff()
    backtest_df = backtest_df.iloc[1:].copy() # Use .copy() to avoid SettingWithCopyWarning

    if backtest_df.empty:
        st.write("Error: DataFrame became empty after handling initial NaN values.")
        return None

    # --- 5. Calculate Cumulative Returns and Equity Curves ---
    initial_capital = 1.0 # Start with unit capital for relative comparison

    # Calculate cumulative log returns
    backtest_df['cumulative_raw_strategy_returns'] = backtest_df['raw_strategy_returns'].cumsum()
    backtest_df['cumulative_net_strategy_returns'] = backtest_df['net_strategy_returns'].cumsum()
    backtest_df['cumulative_bnh_returns'] = backtest_df['log_returns'].cumsum()

    # Calculate equity curves by exponentiating cumulative log returns
    backtest_df['raw_equity_curve'] = initial_capital * np.exp(backtest_df['cumulative_raw_strategy_returns'])
    backtest_df['net_equity_curve'] = initial_capital * np.exp(backtest_df['cumulative_net_strategy_returns'])
    backtest_df['bnh_equity_curve'] = initial_capital * np.exp(backtest_df['cumulative_bnh_returns'])

    st.write("Vectorized backtest calculation complete.")
    return backtest_df
########################################################################################################################################################################################################################
# --- Execution of the Backtesting --- 
if st.button("Run OU Backtest"):
    signals_df = st.session_state.get('signals_df', None)
    if signals_df is not None:
        # Run and display results
        backtest_results = run_vectorized_backtest(signals_df, comm_inp, slip_inp)

        st.write("Backtest Results Summary (First 5 Rows):")
        st.dataframe(backtest_results[['log_returns', 'signal', 'raw_strategy_returns',
                                       'total_costs', 'net_strategy_returns',
                                       'net_equity_curve', 'bnh_equity_curve']].head())

        st.write("Equity Curve Plot:")
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(backtest_results.index, backtest_results['net_equity_curve'], label='Strategy Equity (Net Costs)', color='mediumblue', linewidth=2.0)
        ax.plot(backtest_results.index, backtest_results['raw_equity_curve'], label='Strategy Equity (Raw)', color='deepskyblue', linestyle='--', linewidth=1.0, alpha=0.6)
        ax.plot(backtest_results.index, backtest_results['bnh_equity_curve'], label='Buy & Hold Equity', color='dimgray', linewidth=1.5)
        ax.set_title('Ornstein-Uhlenbeck Mean Reversion Strategy: Equity Curve', fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Value (Log Scale)', fontsize=12)
        ax.set_yscale('log')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, which='both', linestyle=':', linewidth=0.6, color='lightgrey')
        st.pyplot(fig)

        # Save for next section
        strategy_returns_net = backtest_results['net_strategy_returns']
        st.session_state['strategy_returns_net'] = strategy_returns_net
        st.session_state['backtest_results'] = backtest_results
    else:
        st.warning("Please generate trading signals before running the backtest.")
########################################################################################################################################################################################################################
# --- Performance Metrics and Analysis of the trading strategy

st.subheader("Key Performance Metrics and Analysis of the Trading Strategy")

st.write("The most important part after conducting and backtesting a trading strategy is to evaluate its performance.")
st.write("For this, we can use performance metrics that will help us quantify the risk levels taken, and other characteristics")
st.write("The risk-adjusted returns such as the Sharpe and Sortino ratios help us understand this, and also look at how painful the strategy was through the maximum drawdown.")
st.write("Note: all of the performance metrics will be calculated from the after costs returns")

rf_rate_inp = st.number_input("Please input the risk-free rate:", value=0.03, min_value=0.0, max_value=0.1, step=0.01)

def calculate_performance_metrics(returns_series, risk_free_rate=0.0, periods_per_year=24*365):
    """
    Calculates various performance metrics for a strategy returns series.

    Args:
        returns_series (pd.Series): Time series of strategy log returns (net of costs).
        risk_free_rate (float): The annual risk-free rate (default: 0.0).
        periods_per_year (int): Number of trading periods in a year (e.g., 252 for daily, 24*365 for hourly).

    Returns:
        dict: A dictionary containing calculated performance metrics.
    """
    if returns_series is None or returns_series.empty:
        st.error("Error: Returns series is empty or None. Cannot calculate metrics.")
        return None
    if not isinstance(returns_series, pd.Series):
        st.error("Error: returns_series must be a pandas Series.")
        return None

    # Ensure no NaNs in returns
    returns_series = returns_series.dropna()
    if returns_series.empty:
        st.error("Error: Returns series is empty after dropping NaNs.")
        return None

    metrics = {}

    # --- Return Metrics ---
    # Total Cumulative Return (Geometric)
    total_log_return = returns_series.sum()
    metrics['Cumulative Return'] = np.exp(total_log_return) - 1

    # Annualized Return (Geometric)
    num_periods = len(returns_series)
    if num_periods == 0:
        st.warning("Warning: No periods in returns series.")
        return None # Or handle as appropriate

    # Calculate mean log return per period
    mean_log_return_per_period = total_log_return / num_periods
    metrics['Annualized Return'] = np.exp(mean_log_return_per_period * periods_per_year) - 1

    # --- Risk Metrics ---
    # Annualized Volatility (Standard Deviation of Log Returns)
    std_dev_log_return_per_period = returns_series.std()
    metrics['Annualized Volatility'] = std_dev_log_return_per_period * sqrt(periods_per_year)

    # --- Risk-Adjusted Return Metrics ---
    annualized_return = metrics['Annualized Return']
    annualized_volatility = metrics['Annualized Volatility']

    # Sharpe Ratio
    if annualized_volatility == 0:
        metrics['Sharpe Ratio'] = np.nan # Undefined if volatility is zero
    else:
        # Adjust risk-free rate for the return period if necessary, but simpler to use annualized values
        # Risk-free rate needs to be per period if comparing to period returns
        # Here we use annualized return and volatility, so use annual RF rate
        metrics['Sharpe Ratio'] = (annualized_return - risk_free_rate) / annualized_volatility

    # Sortino Ratio
    negative_returns = returns_series[returns_series < 0]
    if negative_returns.empty:
        # No downside deviation if no negative returns
        annualized_downside_volatility = 0
    else:
        # Calculate downside deviation (std dev of negative returns)
        downside_deviation_per_period = negative_returns.std()
        annualized_downside_volatility = downside_deviation_per_period * sqrt(periods_per_year)

    if annualized_downside_volatility == 0:
        metrics['Sortino Ratio'] = np.nan # Undefined if downside volatility is zero
    else:
         metrics['Sortino Ratio'] = (annualized_return - risk_free_rate) / annualized_downside_volatility

    # --- Drawdown Metrics ---
    # Calculate Equity Curve (assuming starting capital of 1)
    cumulative_log_returns = returns_series.cumsum()
    equity_curve = np.exp(cumulative_log_returns)

    # Maximum Drawdown (MDD)
    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1
    metrics['Maximum Drawdown (MDD)'] = drawdown.min() # MDD is the most negative value

    # Calmar Ratio
    mdd = metrics['Maximum Drawdown (MDD)']
    if mdd == 0:
        metrics['Calmar Ratio'] = np.nan # Undefined if MDD is zero
    else:
        metrics['Calmar Ratio'] = annualized_return / abs(mdd)

    # --- Other Statistics ---
    metrics['Skewness'] = returns_series.skew()
    metrics['Kurtosis'] = returns_series.kurt() # Fisher's kurtosis (normal=0)
    metrics['Win Rate'] = (returns_series > 0).sum() / num_periods if num_periods > 0 else 0.0

    return metrics
########################################################################################################################################################################################################################
# --- Defining the performance evaluation function ---

def plot_performance_analysis(backtest_results, strategy_returns_net, periods_per_year=24*365, rolling_window_days=30):
    """
    Generates plots for performance analysis: Drawdown, Rolling Sharpe, Returns Distribution.

    Args:
        backtest_results (pd.DataFrame): DataFrame from the backtester containing 'net_equity_curve'.
        strategy_returns_net (pd.Series): Time series of net strategy log returns.
        periods_per_year (int): Number of trading periods in a year.
        rolling_window_days (int): Window size in days for rolling calculations.
    """
    if backtest_results is None or strategy_returns_net is None:
        st.error("Missing backtest_results or strategy_returns_net for plotting.")
        return
    if 'net_equity_curve' not in backtest_results:
        st.error("'net_equity_curve' not found in backtest_results.")
        return

    strategy_returns_net = strategy_returns_net.dropna()
    if strategy_returns_net.empty:
        st.error("Strategy returns series is empty after dropping NaNs.")
        return

    rolling_window_periods = rolling_window_days * 24

    plt.style.use('seaborn-v0_8-darkgrid')

    # --- 1. Drawdown Plot ---
    equity_curve = backtest_results['net_equity_curve']
    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1

    fig_dd, ax_dd = plt.subplots(figsize=(14, 7))
    drawdown.plot(ax=ax_dd, kind='area', color='salmon', alpha=0.5, label='Drawdown')
    ax_dd.fill_between(drawdown.index, drawdown, 0, color='salmon', alpha=0.5)
    ax_dd.set_ylabel('Drawdown')
    ax_dd.set_title('Strategy Drawdown Curve')
    ax_dd.set_xlabel('Date')
    ax_dd.legend()
    ax_dd.yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))
    plt.tight_layout()
    st.pyplot(fig_dd)
    plt.close(fig_dd)

    # --- 2. Rolling Sharpe Ratio Plot ---
    if len(strategy_returns_net) > rolling_window_periods:
        rolling_mean_log_ret = strategy_returns_net.rolling(window=rolling_window_periods).mean()
        rolling_ann_ret = np.exp(rolling_mean_log_ret * periods_per_year) - 1

        rolling_std_log_ret = strategy_returns_net.rolling(window=rolling_window_periods).std()
        rolling_ann_vol = rolling_std_log_ret * sqrt(periods_per_year)

        rolling_sharpe = rolling_ann_ret / rolling_ann_vol

        fig_sharpe, ax_sharpe = plt.subplots(figsize=(14, 7))
        rolling_sharpe.plot(ax=ax_sharpe, color='darkcyan', label=f'Rolling Sharpe Ratio ({rolling_window_days}-Day Window)')
        ax_sharpe.set_ylabel('Sharpe Ratio')
        ax_sharpe.set_title('Rolling Sharpe Ratio')
        ax_sharpe.axhline(0, color='grey', linestyle='--', linewidth=0.8)
        ax_sharpe.set_xlabel('Date')
        ax_sharpe.legend()
        plt.tight_layout()
        st.pyplot(fig_sharpe)
        plt.close(fig_sharpe)
    else:
        st.warning(f"Not enough data ({len(strategy_returns_net)} periods) for rolling Sharpe with window {rolling_window_periods}. Skipping plot.")

    # --- 3. Distribution of Returns Plot ---
    fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
    sns.histplot(strategy_returns_net, kde=True, bins=50, ax=ax_dist, color='steelblue', stat='density')

    mu, std = strategy_returns_net.mean(), strategy_returns_net.std()
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = (1/(std * np.sqrt(2 * np.pi))) * np.exp(- (x - mu)**2 / (2 * std**2))
    ax_dist.plot(x, p, 'k--', linewidth=1, label='Normal Distribution Fit')

    ax_dist.set_title('Distribution of Strategy Hourly Log Returns (Net)')
    ax_dist.set_xlabel('Hourly Log Return')
    ax_dist.set_ylabel('Density')
    ax_dist.legend()
    plt.tight_layout()
    st.pyplot(fig_dist)
    plt.close(fig_dist)
########################################################################################################################################################################################################################
if st.button("Calculate Performance Metrics"):
    # Retrieve variables from session_state to ensure Streamlit compatibility
    strategy_returns_net = st.session_state.get('strategy_returns_net', None)
    backtest_results = st.session_state.get('backtest_results', None)

    if strategy_returns_net is None:
        st.error("Error: 'strategy_returns_net' Series not found. Please run the backtest first.")
    elif backtest_results is None:
        st.error("Error: 'backtest_results' DataFrame not found. Please run the backtest first.")
    else:
        # Calculate metrics
        performance_metrics = calculate_performance_metrics(strategy_returns_net, risk_free_rate=rf_rate_inp, periods_per_year=24*365)
        st.success("Calculating...")

        if performance_metrics:
            st.write("### Strategy Performance Metrics")
            for key, value in performance_metrics.items():
                if isinstance(value, float):
                    if key in ['Cumulative Return', 'Annualized Return', 'Maximum Drawdown (MDD)', 'Win Rate']:
                        st.write(f"{key}: {value:.2%}")
                    elif key in ['Annualized Volatility']:
                        st.write(f"{key}: {value:.4f}")
                    elif key in ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']:
                        st.write(f"{key}: {value:.2f}")
                    else:
                        st.write(f"{key}: {value:.4f}")
                else:
                    st.write(f"{key}: {value}")

            st.success("Generating Performance Plots...")
            plot_performance_analysis(backtest_results, strategy_returns_net, periods_per_year=24*365)
            st.info("Performance plots are shown above. If you do not see them, please scroll up or check for errors in previous steps.")
        else:
            st.error("Failed to calculate performance metrics.")
########################################################################################################################################################################################################################
