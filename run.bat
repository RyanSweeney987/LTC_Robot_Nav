::start cmd /k tensorboard "--logdir=./models/logs/fit/"
start cmd /k py -3 ./scripts/aisystem.py
timeout /T 30
start cmd /k py -2 ./scripts/robotsystem.py