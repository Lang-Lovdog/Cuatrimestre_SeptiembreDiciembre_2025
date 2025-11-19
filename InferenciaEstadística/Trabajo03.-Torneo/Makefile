MAINDIR=lovdogClassifier/src/lovdogClassify
STATDIR=PruebasEstadísticas/FullTesting_py
TEST_SCRIPT=test_models.py
TRAIN_SCRIPT=train_models.py
STATSCRIPTM=mine.py
STATSCRIPT=mine.py
PY_INTERPRETER=python3
VENV=venv

train:
	. $(VENV)/bin/activate &&\
	cd $(MAINDIR) &&\
	$(PY_INTERPRETER) $(TRAIN_SCRIPT)

test:
	. $(VENV)/bin/activate &&\
	cd $(MAINDIR) &&\
	$(PY_INTERPRETER) $(TEST_SCRIPT)

stat_test:
	. $(VENV)/bin/activate &&\
	cd $(STATDIR) &&\
	$(PY_INTERPRETER) $(STATSCRIPT)

stat_mtest:
	. $(VENV)/bin/activate &&\
	cd $(STATDIR) &&\
	$(PY_INTERPRETER) $(STATSCRIPTM)
