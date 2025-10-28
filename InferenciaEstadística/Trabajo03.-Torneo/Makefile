MAINDIR=lovdogClassifier/src/lovdogClassify
TEST_SCRIPT=test_models.py
TRAIN_SCRIPT=train_models.py
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
