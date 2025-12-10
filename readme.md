# MLOps Sample Project
## How to Use:
1. `git clone` the repo, `cd` into it, start a virtual environment in it and source it (preferred)
2. Open one terminal window, run the mlflow server there: `mlflow server --port 5000`
3. Train and log the model using `python train.py` in another shell window
4. In yet another shell window, run `uvicorn --host localhost --port 8000 app:app`
5. OPen yet another shell window and run `streamlit run streamlit_frontend.py`

## To-Dos:
1. Automate model training and deployment
2. Create config workflows to deploy on a server
3. Document the how-tos and let viewers learn