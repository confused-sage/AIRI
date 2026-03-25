import pandas as pd
import numpy as np
import yfinance as yf
import time

n_investors = 1000
stock_pool = [
    ("RELIANCE.NS", "Energy", "Low"),
    ("TCS.NS", "IT", "Low"),
    ("INFY.NS", "IT", "Low"),
    ("HDFCBANK.NS", "Banking", "Low"),
    ("ICICIBANK.NS", "Banking", "Low"),
    ("SBIN.NS", "Banking", "Medium"),
    ("HINDUNILVR.NS", "FMCG", "Low"),
    ("ITC.NS", "FMCG", "Low"),
    ("LT.NS", "Infra", "Medium"),
    ("ADANIENT.NS", "Infra", "High"),
    ("MARUTI.NS", "Auto", "Medium"),
    ("SUNPHARMA.NS", "Pharma", "Low"),
    ("DRREDDY.NS", "Pharma", "Medium"),
    ("BAJFINANCE.NS", "Finance", "High"),
    ("KOTAKBANK.NS", "Banking", "Low"),
    ("NYKAA.NS", "Retail", "High"),
    ("DMART.NS", "Retail", "Medium")
]

age = np.random.choice(range(21, 60), size=n_investors, 
                       p=np.linspace(1, 2, 39)/np.sum(np.linspace(1, 2, 39)))

income = np.random.lognormal(mean=13, sigma=0.5, size=n_investors)
income = np.clip(income, 100_000, 80_000_000).astype(int)

investment_ratio = np.random.uniform(0.1, 0.4, n_investors)
investment = (income * investment_ratio).astype(int)

equity_percent = np.random.choice([30, 40, 50, 60, 70, 80], size=n_investors, 
                                  p=[0.1, 0.15, 0.25, 0.25, 0.15, 0.1])
debt_percent = 100 - equity_percent

trades = []
for a in age:
    if a < 30:
        trades.append(np.random.choice([10, 15, 20, 25], p=[0.2, 0.3, 0.3, 0.2]))
    elif a < 45:
        trades.append(np.random.choice([5, 10, 15], p=[0.3, 0.5, 0.2]))
    else:
        trades.append(np.random.choice([2, 5, 8], p=[0.4, 0.4, 0.2]))
trades = np.array(trades)

risk_personality = np.random.choice(["Conservative", "Balanced", "Aggressive"], size=n_investors, p=[0.3, 0.5, 0.2])

investors = pd.DataFrame({
    "Investor_ID": [f"INV_{i+1}" for i in range(n_investors)],
    "Age": age,
    "Income": income,
    "Investment": investment,
    "Equity_Percent": equity_percent,
    "Debt_Percent": debt_percent,
    "Trades_Per_Month": trades,
    "Risk_Personality": risk_personality
})

investors["Equity_Amount"] = (investors["Investment"] * investors["Equity_Percent"] / 100).astype(int)
investors["Debt_Amount"] = (investors["Investment"] * investors["Debt_Percent"] / 100).astype(int)

def fetch_stock_data(ticker):
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if df.empty or "Close" not in df:
            return None
        close = df["Close"].dropna()
        returns = close.pct_change().dropna()
        return {
            "price": float(close.iloc[-1]),
            "vol": float(returns.std() * np.sqrt(252)),
            "ret": float(returns.mean() * 252),
            "returns_list": returns.values.tolist()
        }
    except:
        return None

print("Fetching stock data...")
market_data = []
for ticker, sector, risk_tag in stock_pool:
    result = fetch_stock_data(ticker)
    if result:
        market_data.append({
            "Ticker": ticker,
            "Sector": sector,
            "Risk_Tag": risk_tag,
            "Price": result["price"],
            "Volatility": result["vol"],
            "Expected_Return": result["ret"],
            "Returns_List": result["returns_list"]
        })
    else:
        print(f"Skipping {ticker}")
    time.sleep(0.3)

market_df = pd.DataFrame(market_data)
if len(market_df) < 5:
    raise ValueError("Not enough valid stocks fetched.")

nifty = yf.download("^NSEI", period="6mo", interval="1d", progress=False)
market_returns = nifty["Close"].pct_change().dropna().values.tolist()


holdings_list = []

for _, inv in investors.iterrows():
    equity_budget = inv["Equity_Amount"]
    num_stocks = np.random.randint(4, min(10, len(market_df)))
    
    selected_stocks = market_df.sample(num_stocks, replace=True).reset_index(drop=True)
    
    alpha = np.random.uniform(0.5, 2.5, num_stocks)
    weights = np.random.dirichlet(alpha)
    
    if inv["Risk_Personality"] == "Aggressive":
        weights *= np.random.uniform(1.0, 1.2, num_stocks)
    elif inv["Risk_Personality"] == "Conservative":
        weights *= np.random.uniform(0.8, 1.0, num_stocks)
    
    weights /= weights.sum()
    
    sector_weights = {}
    
    for i, stock in selected_stocks.iterrows():
        sector = stock["Sector"]
        if sector_weights.get(sector, 0) > 0.6:
            continue
        
        allocation = weights[i] * equity_budget
        price = stock["Price"]
        quantity = max(1, int(allocation / price))
        weight = allocation / equity_budget if equity_budget > 0 else 0
        sector_weights[sector] = sector_weights.get(sector, 0) + weight
        
        vol = stock["Volatility"] * np.random.uniform(0.95, 1.05)
        exp_ret = stock["Expected_Return"] * np.random.uniform(0.95, 1.05)
        
        holdings_list.append({
            "Investor_ID": inv["Investor_ID"],
            "Ticker": stock["Ticker"],
            "Sector": sector,
            "Risk_Tag": stock["Risk_Tag"],
            "Quantity": quantity,
            "Price": price,
            "Volatility": vol,
            "Expected_Return": exp_ret,
            "Returns_List": stock["Returns_List"],
            "Market_Returns_List": market_returns
        })

holdings = pd.DataFrame(holdings_list)


investors.to_csv("realistic_investor_data.csv", index=False)
holdings.to_csv("holdings_data.csv", index=False)

print("\nDatasets generated successfully!")
print(f"Investors: {len(investors)}")
print(f"Holdings: {len(holdings)}")
print(f"Avg stocks per investor: {holdings.groupby('Investor_ID').size().mean():.2f}")