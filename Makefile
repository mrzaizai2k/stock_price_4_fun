install:
#	# python -m venv venv
#	# venv/bin/activate
	pip install -r setup.txt
freeze:
	pip freeze > setup.txt
bot:
	python src/stock_bot.py

all: install bot