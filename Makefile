install:
#	# python -m venv venv
#	source ./venv/bin/activate
	pip install -r setup.txt
freeze:
	pip freeze > setup.txt

bot:
	mkdir -p logging
	rm	-f logging/out.txt
	touch logging/out.txt
	python -u src/stock_bot.py | tee logging/out.txt

test:
	python src/PayBackTime.py
	python src/motif.py
	python src/Indicators.py
	python src/support_resist.py
	python src/summarize_text.py

list: #List all the running bots 
	ps aux | grep python
# To kill, use command (PID is the process ID): kill PID
kill:
	pkill -f "python src/stock_bot.py"

all: install bot
