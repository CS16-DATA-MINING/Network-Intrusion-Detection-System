# OWNER: App team, with integration input from parser and model teams.
# WHAT THIS FILE CONTAINS: Only the small orchestration needed to connect
# validated upload -> pcap_parser.py -> preprocessing.py/models.py -> results.
# Add it if app.py becomes hard to follow; otherwise keep the flow in app.py.
# WHAT IT MUST NOT CONTAIN: A second parser, duplicate transformations, model
# definitions, training, or notebook execution during a prediction request.
