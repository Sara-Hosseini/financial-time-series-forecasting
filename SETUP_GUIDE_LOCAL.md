# Stock Transformer Project — Local Setup Guide (Private)

## هدف
ساخت پروژه در GitHub، اتصال به VS Code، ساخت محیط پایتون، نصب پکیج‌ها، تنظیم API Key و آماده‌سازی برای دانلود داده از Alpha Vantage.

---

## 1) ساخت Repository در GitHub
1. GitHub → علامت + → New repository
2. Repository name: stock-transformer-project
3. Add README ✅
4. License: MIT (اختیاری ولی حرفه‌ای)
5. Create repository

---

## 2) ساخت ساختار پوشه‌ها روی GitHub (با .gitkeep)
چون GitHub پوشه خالی را نگه نمی‌دارد:
- Add file → Create new file → data/raw/.gitkeep → Commit
- Add file → Create new file → data/processed/.gitkeep → Commit
- Add file → Create new file → notebooks/.gitkeep → Commit

---

## 3) تنظیم .gitignore
هدف: جلوگیری از آپلود موارد حساس/سنگین
موارد مهم:
- .env
- .venv/
- __pycache__/
- *.pyc
- .ipynb_checkpoints/
- data/raw/*.csv (اختیاری)

---

## 4) نصب Git روی ویندوز
در PowerShell:
git --version
اگر خطا داد:
- Git for Windows نصب کن
دوباره:
git --version
باید نسخه نشان دهد.

---

## 5) باز کردن پروژه در VS Code
1. VS Code → File → Open Folder
2. مسیر پروژه: D:\lessons\season 4\code\stock-transformer-project

---

## 6) ساخت محیط مجازی (venv) و نصب پکیج‌ها
### ساخت venv
python -m venv .venv

### فعال‌سازی venv
.\.venv\Scripts\Activate.ps1

اگر خطای execution policy داد:
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

بعد دوباره Activate.

### آپدیت pip و نصب requirements
python -m pip install --upgrade pip
pip install -r requirements.txt

---

## 7) انتخاب Interpreter در VS Code
1. Ctrl + Shift + P
2. Python: Select Interpreter
3. گزینه‌ای که شامل .venv باشد:
...\stock-transformer-project\.venv\Scripts\python.exe

تست:
python -c "import sys; print(sys.executable)"
باید مسیر .venv را بدهد.

---

## 8) ساخت فایل .env و تنظیم Alpha Vantage API Key
در ریشه پروژه فایل بساز:
.env

داخلش فقط یک خط:
ALPHAVANTAGE_API_KEY=YOUR_REAL_KEY

نکته: هیچ دستور دیگری داخل .env نوشته نمی‌شود.

نصب dotenv:
pip install python-dotenv

تست:
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('OK' if os.getenv('ALPHAVANTAGE_API_KEY') else 'MISSING')"

---

## 9) ساخت src و فایل‌های کد
ساختار درست:
src/  (folder)
  data_collection.py

نکته رایج: اشتباه ساختن فایل src به جای پوشه src

---

## 10) نکته امنیتی
- .env نباید روی GitHub برود (در .gitignore باشد)
- .venv نباید روی GitHub برود

---

## وضعیت فعلی چک‌لیست
- Repo ساخته شد ✅
- VS Code پروژه را باز کرد ✅
- Git نصب شد ✅
- .venv ساخته و فعال شد ✅
- packages نصب شدند ✅
- .env و API Key آماده شد ✅
- python-dotenv نصب شد ✅
- Interpreter روی .venv تنظیم شد ✅
11) نوشتن اسکریپت دریافت داده از Alpha Vantage

فایل زیر را داخل مسیر src/data_collection.py ساختیم.

هدف این فایل

اتصال به API آلفا وینتیج

دریافت داده قیمت روزانه سهام

تبدیل پاسخ JSON به DataFrame

ذخیره داده خام در data/raw/

منطق کلی کد

خواندن API Key از .env

ارسال درخواست HTTP به Alpha Vantage

بررسی خطاها (rate limit / اطلاعات اشتراک)

تمیزسازی داده

ذخیره CSV

12) اجرای اسکریپت و دانلود داده

در ترمینال (با فعال بودن .venv) اجرا شد:

python src/data_collection.py


خروجی موفق:

فایل ساخته شد:

data/raw/AAPL_daily_compact.csv


پیام ترمینال:

Saved: data\raw\AAPL_daily_compact.csv
Shape: (100, 5)


ستون‌ها:

Open

High

Low

Close

Volume

نکته مهم:

چون از پلن رایگان Alpha Vantage استفاده می‌شود:

فقط 100 روز اخیر

outputsize=compact

endpoint: TIME_SERIES_DAILY

13) ثبت تغییرات در GitHub (ساعت کاری)

بعد از اجرای موفق کد:

git status
git add src/data_collection.py .gitignore
git commit -m "Fix Alpha Vantage free-tier usage and improve data collection output"
git push


نتیجه:

تغییرات در GitHub ثبت شد

commit دارای:

زمان

توضیح

نام نویسنده

✅ این commit به‌عنوان نشانه‌ی ساعت کاری معتبر حساب می‌شود.

14) شروع EDA در Jupyter Notebook

فایل ساخته شد:

notebooks/02_eda.ipynb

کارهای انجام‌شده در EDA

بارگذاری داده خام CSV

تبدیل ستون تاریخ به DateTime

تنظیم Date به‌عنوان index

بررسی اولیه داده

کد بارگذاری:

df = pd.read_csv("../data/raw/AAPL_daily_compact.csv",
                 parse_dates=["Unnamed: 0"])
df = df.rename(columns={"Unnamed: 0": "Date"})
df = df.set_index("Date")

15) Feature Engineering پایه (مهم برای Transformer)
15.1 محاسبه بازده روزانه
df["Return"] = df["Close"].pct_change()


معنی:

درصد تغییر قیمت امروز نسبت به دیروز

یکی از مهم‌ترین ویژگی‌ها در مدل‌های مالی

15.2 Rolling Window (پایه‌ی LSTM و Transformer)
df["MA_5"] = df["Close"].rolling(5).mean()


معنی:

میانگین قیمت ۵ روز اخیر

مدل یاد می‌گیرد «پنجره زمانی» را ببیند

15.3 Resampling (تغییر فرکانس زمانی)
df.resample("W").mean()


معنی:

تبدیل داده روزانه به هفتگی

کاهش نویز

مناسب برای تحلیل‌های بلندمدت

16) ذخیره نوت‌بوک و ثبت در GitHub
git add notebooks/02_eda.ipynb
git commit -m "Add EDA notebook: quality checks, returns, rolling features, resampling"
git push


نتیجه:

نوت‌بوک کامل روی GitHub ذخیره شد

یک قدم واقعی به سمت پروژه پایان‌نامه / رزومه

17) وضعیت فعلی پروژه

در این مرحله پروژه شامل:

✅ ساخت Repo حرفه‌ای

✅ محیط توسعه استاندارد

✅ اتصال به API واقعی

✅ دریافت داده مالی واقعی

✅ EDA و feature engineering پایه

✅ ثبت تمام مراحل در GitHub
افزودن منبع Yahoo Finance برای دیتای 10 ساله

نصب پکیج:

pip install yfinance

افزودن yfinance به requirements (ترجیحاً مینیمال)

افزودن تابع fetch_yahoo_daily(...) داخل src/data_collection.py

تعریف بازه زمانی 10 سال:

start_date = "2015-01-01"

end_date = "2026-01-01" (در yfinance تاریخ پایان معمولاً exclusive است)

ذخیره خروجی:

data/raw/AAPL_daily_10y_yahoo.csv

تست:

python src/data_collection.py

انتظار: ~2700 ردیف و بازه 2015 تا امروز

12) کامیت و پوش تغییرات

git status

git add src/data_collection.py requirements.txt .gitignore

git commit -m "Add Yahoo Finance 10-year data source and keep Alpha compact pipeline"

git push