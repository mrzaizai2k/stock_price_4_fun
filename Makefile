install:
#	# python -m venv venv
#	source ./venv/bin/activate
	pip install -r setup.txt
freeze:
	pip freeze > setup.txt
bot:
	python src/stock_bot.py
test:
	python src/PayBackTime.py
	python src/motif.py
	python src/Indicators.py
	python src/support_resist.py
	python src/summarize_text.py
	
all: install bot