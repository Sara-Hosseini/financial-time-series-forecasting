Stock Transformer Project — Local Setup Guide (Private)
Objective

Create a GitHub-based project, connect it to VS Code, set up a Python environment, install required packages, configure API keys, and prepare the pipeline for downloading historical stock price data from Alpha Vantage and Yahoo Finance.

1) Create a Repository on GitHub

GitHub → “+” → New repository

Repository name: stock-transformer-project

Add README ✅

License: MIT (optional but professional)

Create repository

2) Create Folder Structure on GitHub (Using .gitkeep)

Since GitHub does not track empty folders:

Add file → Create new file → data/raw/.gitkeep → Commit

Add file → Create new file → data/processed/.gitkeep → Commit

Add file → Create new file → notebooks/.gitkeep → Commit

3) Configure .gitignore

Purpose: prevent sensitive or heavy files from being pushed to GitHub.

Important entries:

.env

.venv/

__pycache__/

*.pyc

.ipynb_checkpoints/

data/raw/*.csv (optional)

4) Install Git on Windows

In PowerShell:

git --version


If Git is not installed:

Install Git for Windows

Run again:

git --version


A version number should be displayed.

5) Open the Project in VS Code

VS Code → File → Open Folder

Project path:
D:\lessons\season 4\code\stock-transformer-project

6) Create a Virtual Environment (venv) and Install Packages
Create venv
python -m venv .venv

Activate venv
.\.venv\Scripts\Activate.ps1


If an execution policy error occurs:

Set-ExecutionPolicy -Scope CurrentUser RemoteSigned


Then activate again.

Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

7) Select Python Interpreter in VS Code

Ctrl + Shift + P

Python: Select Interpreter

Choose the interpreter inside .venv:

...\stock-transformer-project\.venv\Scripts\python.exe


Verification:

python -c "import sys; print(sys.executable)"


The output should point to .venv.

8) Create .env File and Configure Alpha Vantage API Key

Create a file named .env in the project root.

Content (single line only):

ALPHAVANTAGE_API_KEY=YOUR_REAL_KEY


Note: No other commands should be written inside .env.

Install dotenv:

pip install python-dotenv


Test:

python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('OK' if os.getenv('ALPHAVANTAGE_API_KEY') else 'MISSING')"

9) Create src Folder and Code Files

Correct structure:

src/
  data_collection.py


Common mistake: creating a file named src instead of a folder.

10) Security Notes

.env must not be pushed to GitHub (included in .gitignore)

.venv must not be pushed to GitHub

Current Checklist Status

Repository created ✅

Project opened in VS Code ✅

Git installed ✅

.venv created and activated ✅

Packages installed ✅

.env and API key configured ✅

python-dotenv installed ✅

Interpreter set to .venv ✅

11) Writing the Alpha Vantage Data Collection Script

File created:

src/data_collection.py

Purpose of this script

Connect to Alpha Vantage API

Download daily stock price data

Convert JSON response into a Pandas DataFrame

Save raw data to data/raw/

Core logic

Load API key from .env

Send HTTP request to Alpha Vantage

Handle API errors (rate limit / subscription info)

Clean and format data

Save data as CSV

12) Run the Script and Download Data

Command (with .venv activated):

python src/data_collection.py

Successful output

Generated file:

data/raw/AAPL_daily_compact.csv


Terminal output:

Saved: data\raw\AAPL_daily_compact.csv
Shape: (100, 5)


Columns:

Open

High

Low

Close

Volume

Important note

Using Alpha Vantage free tier:

Only last 100 trading days available

outputsize=compact

Endpoint: TIME_SERIES_DAILY

13) Commit Changes to GitHub (Work Hours Tracking)

After successful execution:

git status
git add src/data_collection.py .gitignore
git commit -m "Fix Alpha Vantage free-tier usage and improve data collection output"
git push

Result

Changes saved on GitHub

Commit includes timestamp, author, and description

✅ This commit is considered valid evidence of work hours

14) Start EDA in Jupyter Notebook

Notebook created:

notebooks/02_eda.ipynb

EDA steps performed

Load raw CSV data

Convert date column to DateTime

Set Date as index

Perform initial data inspection

Data loading code
df = pd.read_csv("../data/raw/AAPL_daily_compact.csv",
                 parse_dates=["Unnamed: 0"])
df = df.rename(columns={"Unnamed: 0": "Date"})
df = df.set_index("Date")

15) Basic Feature Engineering (Important for Transformers)
15.1 Daily Returns
df["Return"] = df["Close"].pct_change()


Meaning:

Percentage change of today’s price compared to yesterday

One of the most important features in financial modeling

15.2 Rolling Window (Foundation for LSTM & Transformer)
df["MA_5"] = df["Close"].rolling(5).mean()


Meaning:

5-day moving average

Enables the model to learn temporal windows

15.3 Resampling (Change Time Frequency)
df.resample("W").mean()


Meaning:

Convert daily data to weekly

Reduce noise

Useful for long-term analysis

16) Save Notebook and Commit to GitHub
git add notebooks/02_eda.ipynb
git commit -m "Add EDA notebook: quality checks, returns, rolling features, resampling"
git push

Result

Notebook saved on GitHub

A concrete step toward thesis-quality and resume-ready project

17) Current Project Status

At this stage, the project includes:

✅ Professional GitHub repository

✅ Standard development environment

✅ Connection to real financial APIs

✅ Download of real market data

✅ EDA and basic feature engineering

✅ Full version control history

Adding Yahoo Finance as a 10-Year Historical Data Source
Package installation
pip install yfinance

Update requirements

Add yfinance to requirements.txt (preferably minimal)

Implementation

Add fetch_yahoo_daily(...) function to src/data_collection.py

Define 10-year time range:

start_date = "2015-01-01"
end_date   = "2026-01-01"   # end date is typically exclusive in yfinance

Output
data/raw/AAPL_daily_10y_yahoo.csv

Test
python src/data_collection.py


Expected:

~2700 rows

Date range: 2015 → present

18) Commit and Push Yahoo Finance Integration
git status
git add src/data_collection.py requirements.txt .gitignore
git commit -m "Add Yahoo Finance 10-year data source and keep Alpha compact pipeline"
git push

Work Log

2025-12-28

Implemented data collection pipeline using Alpha Vantage (free-tier, compact)

Added Yahoo Finance as a secondary source for 10-year historical price data

Verified data integrity and date ranges

Updated project setup documentation and requirements

Approximate effort: 4–5 hours
## Work Log – Time Series EDA (Yahoo Finance)

### Date
2026-01-16

### Summary
Performed a comprehensive exploratory data analysis (EDA) on the 10-year historical stock price dataset obtained from Yahoo Finance (AAPL). The focus was on ensuring data correctness, resolving multi-level CSV header issues, and preparing the dataset for time-series modeling.

### Tasks Completed
- Reloaded and validated the full 10-year Yahoo Finance dataset
- Fixed CSV parsing issues related to multi-level headers and unnamed columns
- Standardized the dataset schema to OHLCV format
- Verified datetime indexing, data types, and absence of missing values
- Conducted time-series EDA including:
  - Price trend analysis and non-stationarity assessment
  - Dual-axis visualization of price and trading volume
  - Log return computation and analysis
  - Rolling volatility analysis and identification of volatility clustering
- Identified and resolved unintended dataframe slicing affecting visualization range
- Structured EDA outputs to align with academic expectations for time-series analysis

### Outcome
The dataset is now fully validated and the EDA is complete and ready for academic submission. Results confirm non-stationarity in price levels, more stable behavior in log returns, and time-varying volatility, motivating the use of sequence-based models in subsequent stages.

### Estimated Effort
~4–6 hours
