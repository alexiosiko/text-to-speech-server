# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from text_to_speech import router as text_to_speech_router

app = FastAPI()

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Test route to verify server
@app.get("/")
def test():
    return "Hello world!"

# Include the text-to-speech router
app.include_router(text_to_speech_router)

# Run app with uvicorn if needed for local testing
# uvicorn main:app --host 0.0.0.0 --port 8000
