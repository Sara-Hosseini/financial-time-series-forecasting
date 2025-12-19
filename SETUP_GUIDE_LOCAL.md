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
