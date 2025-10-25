"""
Marketing Brand Intelligence (MBI) System
Main Application Entry Point

This is the FastAPI application that powers the MBI backend.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
import structlog

from app.core.config import settings
from app.core.logging import setup_logging
from app.core.database import init_db, close_db
from app.core.redis import init_redis, close_redis
from app.core.exceptions import (
    MBIException,
    NotFoundException,
    ValidationException,
    UnauthorizedException,
)

# Initialize structured logging
setup_logging()
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Application lifespan manager
    Handles startup and shutdown events
    """
    # Startup
    logger.info("Starting MBI application", env=settings.ENV)
    
    # Initialize database
    await init_db()
    logger.info("Database initialized")
    
    # Initialize Redis
    await init_redis()
    logger.info("Redis initialized")
    
    # Initialize feature store
    # await init_feature_store()
    
    # Start background tasks
    # await start_background_workers()
    
    logger.info("MBI application started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down MBI application")
    
    # Close database connections
    await close_db()
    logger.info("Database connections closed")
    
    # Close Redis connections
    await close_redis()
    logger.info("Redis connections closed")
    
    logger.info("MBI application shut down complete")


# Create FastAPI application
app = FastAPI(
    title="Marketing Brand Intelligence API",
    description="""
    Comprehensive AI-driven marketing intelligence and CRM platform combining:
    - Marketing Mix Modeling (MMM) - Bayesian attribution
    - Multi-Touch Attribution (MTA) - Markov/Shapley values
    - 15+ AI Agents for lead scoring, creative intelligence, crisis detection
    - Full CRM system (Leads, Accounts, Opportunities, Cases)
    - Automated playbooks for optimization
    """,
    version=settings.API_VERSION,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan,
)

# ============================================================================
# MIDDLEWARE
# ============================================================================

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Request ID middleware
@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    """Add request ID to all requests for tracing"""
    import uuid
    request_id = str(uuid.uuid4())
    
    # Add to structlog context
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response

# Request logging middleware
@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    """Log all requests"""
    logger.info(
        "Request started",
        method=request.method,
        path=request.url.path,
        client_ip=request.client.host if request.client else None,
    )
    
    response = await call_next(request)
    
    logger.info(
        "Request completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
    )
    
    return response

# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(MBIException)
async def mbi_exception_handler(request: Request, exc: MBIException):
    """Handle custom MBI exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "details": exc.details,
        },
    )

@app.exception_handler(NotFoundException)
async def not_found_exception_handler(request: Request, exc: NotFoundException):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "not_found",
            "message": exc.message,
        },
    )

@app.exception_handler(ValidationException)
async def validation_exception_handler(request: Request, exc: ValidationException):
    """Handle validation errors"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "validation_error",
            "message": exc.message,
            "details": exc.details,
        },
    )

@app.exception_handler(UnauthorizedException)
async def unauthorized_exception_handler(request: Request, exc: UnauthorizedException):
    """Handle authentication errors"""
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={
            "error": "unauthorized",
            "message": exc.message,
        },
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors"""
    logger.error(
        "Unhandled exception",
        exc_info=exc,
        path=request.url.path,
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred",
        },
    )

# ============================================================================
# ROUTES
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Marketing Brand Intelligence API",
        "version": settings.API_VERSION,
        "status": "operational",
        "docs": "/docs" if settings.DEBUG else None,
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # TODO: Add actual health checks for dependencies
    return {
        "status": "healthy",
        "version": settings.API_VERSION,
        "components": {
            "database": "healthy",
            "redis": "healthy",
            "kafka": "healthy",
        },
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    # TODO: Add readiness checks
    return {"status": "ready"}

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ============================================================================
# API ROUTERS
# ============================================================================

# Import and include API routers
# from app.api.v2 import ingestion, intelligence, crm, decisions, playbooks

# app.include_router(
#     ingestion.router,
#     prefix=f"/api/{settings.API_VERSION}/ingest",
#     tags=["Ingestion"],
# )

# app.include_router(
#     intelligence.router,
#     prefix=f"/api/{settings.API_VERSION}/intelligence",
#     tags=["Intelligence"],
# )

# app.include_router(
#     crm.router,
#     prefix=f"/api/{settings.API_VERSION}/crm",
#     tags=["CRM"],
# )

# app.include_router(
#     decisions.router,
#     prefix=f"/api/{settings.API_VERSION}/decisions",
#     tags=["Decisions"],
# )

# app.include_router(
#     playbooks.router,
#     prefix=f"/api/{settings.API_VERSION}/playbooks",
#     tags=["Playbooks"],
# )

# ============================================================================
# EVENT HANDLERS (commented out until implemented)
# ============================================================================

# from app.events.handlers import register_event_handlers
# register_event_handlers(app)

# ============================================================================
# STARTUP MESSAGE
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info(
        "Starting uvicorn server",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
    )
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info",
    )
