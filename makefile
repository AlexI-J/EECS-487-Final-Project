run:
	@echo "Starting FastAPI and Streamlit..."
	@uvicorn main:app --reload & \
	streamlit run app.py