from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from src.api.routes.data import router as data_router

def get_application():
    app = FastAPI()

    # Include the routers
    app.include_router(data_router)

    # Define the root endpoint to redirect to the Swagger documentation
    @app.get("/")
    async def root():
        return RedirectResponse(url="/docs")

    return app
