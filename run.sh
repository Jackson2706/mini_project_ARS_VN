#!/bin/bash
pip install -U scikit-learn
pip install -r requirements.txt
python prepare_data.py
python src/select_model.py
