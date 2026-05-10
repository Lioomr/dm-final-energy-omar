"""Streamlit Community Cloud entrypoint.

The main dashboard code lives in dashboard/app.py. This small root file makes
deployment simpler because Streamlit Cloud can use streamlit_app.py directly.
"""

import dashboard.app  # noqa: F401

