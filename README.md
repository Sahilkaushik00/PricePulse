# PricePulse Agent - Streamlit Deployment

This folder contains a standalone Streamlit application powered by a LangGraph agent.

## Deployment Instructions

1.  **GitHub**: Push the contents of this `streamlit_app/` folder to a new GitHub repository.
2.  **Streamlit Cloud**: 
    - Connect your GitHub account to [Streamlit Cloud](https://share.streamlit.io/).
    - Select your repository.
    - Set the main file path to `streamlit_app.py`.
3.  **Secrets**: In Streamlit Cloud settings, add your `GEMINI_API_KEY` to the Secrets panel.

## Local Setup

1.  Clone the repo.
2.  Install dependencies: `pip install -r requirements.txt`.
3.  Create a `.env` file with your `GEMINI_API_KEY`.
4.  Run the app: `streamlit run streamlit_app.py`.
