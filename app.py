"""
Production FastAPI service for background removal.
"""

import os
import logging
import uuid
import shutil
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl, field_validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Application configuration from environment variables."""
    
    # Model settings
    MODEL_NAME: str = os.getenv("MODEL_NAME", "ZhengPeng7/BiRefNet")
    DEVICE: str = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    TORCH_PRECISION: str = os.getenv("TORCH_PRECISION", "high")
    
    # Image processing
    IMAGE_SIZE: int = int(os.getenv("IMAGE_SIZE", "1024"))
    NORMALIZE_MEAN: list = [float(x) for x in os.getenv("NORMALIZE_MEAN", "0.485,0.456,0.406").split(",")]
    NORMALIZE_STD: list = [float(x) for x in os.getenv("NORMALIZE_STD", "0.229,0.224,0.225").split(",")]
    ALPHA_THRESHOLD: int = int(os.getenv("ALPHA_THRESHOLD", "100"))
    
    # File handling
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    ALLOWED_FORMATS: set = set(os.getenv("ALLOWED_FORMATS", "jpg,jpeg,png,webp,bmp").split(","))
    TEMP_DIR: str = os.getenv("TEMP_DIR", "temp")
    
    # S3 settings
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "")
    S3_ROOT_FOLDER: str = os.getenv("S3_ROOT_FOLDER", "bg_removed_avatar")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-west-1")
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    S3_PRESIGNED_URL_EXPIRATION: int = int(os.getenv("S3_PRESIGNED_URL_EXPIRATION", "86400"))
    
    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if not cls.S3_BUCKET_NAME:
            logger.warning("S3_BUCKET_NAME not configured - S3 uploads will be disabled")
        
        if cls.IMAGE_SIZE <= 0:
            raise ValueError(f"IMAGE_SIZE must be positive, got {cls.IMAGE_SIZE}")
        
        if not (0 <= cls.ALPHA_THRESHOLD <= 255):
            raise ValueError(f"ALPHA_THRESHOLD must be 0-255, got {cls.ALPHA_THRESHOLD}")
        
        logger.info(f"Config validated: Device={cls.DEVICE}, Model={cls.MODEL_NAME}")


# ============================================================================
# Models and Services
# ============================================================================

class BackgroundRemovalService:
    """Service for background removal operations."""
    
    def __init__(self):
        self._model: Optional[nn.Module] = None
        self._transform: Optional[transforms.Compose] = None
        self._s3_client = None
    
    @property
    def model(self) -> nn.Module:
        """Lazy load the model."""
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def transform(self) -> transforms.Compose:
        """Get image transformation pipeline."""
        if self._transform is None:
            self._transform = transforms.Compose([
                transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(Config.NORMALIZE_MEAN, Config.NORMALIZE_STD),
            ])
        return self._transform
    
    @property
    def s3_client(self):
        """Lazy load S3 client."""
        if self._s3_client is None and Config.S3_BUCKET_NAME:
            try:
                self._s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
                    region_name=Config.AWS_REGION,
                )
                logger.info("S3 client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize S3 client: {e}")
        return self._s3_client
    
    def _load_model(self):
        """Load the BiRefNet model."""
        try:
            logger.info(f"Loading model: {Config.MODEL_NAME}")
            torch.set_float32_matmul_precision(Config.TORCH_PRECISION)
            
            from transformers import AutoModelForImageSegmentation
            
            self._model = AutoModelForImageSegmentation.from_pretrained(
                Config.MODEL_NAME,
                trust_remote_code=True
            )
            self._model.to(Config.DEVICE)
            self._model.eval()
            
            logger.info(f"Model loaded on {Config.DEVICE}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def remove_background(self, image: Image.Image) -> Image.Image:
        """Remove background from image."""
        try:
            original_size = image.size
            
            # Ensure RGB mode
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Preprocess
            input_tensor = self.transform(image).unsqueeze(0).to(Config.DEVICE)
            
            # Inference
            with torch.no_grad():
                predictions = self.model(input_tensor)[-1].sigmoid().cpu()
            
            # Post-process
            pred_mask = predictions[0].squeeze()
            pred_pil = transforms.ToPILImage()(pred_mask)
            mask = pred_pil.resize(original_size, Image.Resampling.LANCZOS)
            
            # Apply alpha channel
            result = image.copy()
            result.putalpha(mask)
            
            logger.info("Background removed successfully")
            return result
        
        except torch.cuda.OutOfMemoryError:
            logger.error("GPU out of memory")
            raise HTTPException(
                status_code=500,
                detail="GPU out of memory. Try reducing image size."
            )
        except Exception as e:
            logger.error(f"Background removal failed: {e}")
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    def generate_mask(self, image: Image.Image) -> Image.Image:
        """Generate binary mask from alpha channel."""
        try:
            if image.mode != "RGBA":
                raise ValueError("Image must have alpha channel (RGBA mode)")
            
            alpha = image.split()[-1]
            binary_mask = alpha.point(
                lambda p: 0 if p < Config.ALPHA_THRESHOLD else 255
            )
            
            logger.info("Mask generated successfully")
            return binary_mask
        except Exception as e:
            logger.error(f"Mask generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Mask generation failed: {str(e)}")
    
    def upload_to_s3(self, local_folder: str, subfolder: str) -> dict[str, str]:
        """Upload folder to S3 and return presigned URLs mapped by filename prefixes."""
        if not self.s3_client:
            raise HTTPException(status_code=500, detail="S3 not configured")
        
        try:
            presigned_urls = {}
            
            for root, _, files in os.walk(local_folder):
                for filename in files:
                    local_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(local_path, local_folder)
                    
                    s3_key = f"{Config.S3_ROOT_FOLDER}/{subfolder}/{relative_path}"
                    
                    # Upload file
                    self.s3_client.upload_file(
                        Filename=local_path,
                        Bucket=Config.S3_BUCKET_NAME,
                        Key=s3_key
                    )
                    
                    # Generate presigned URL
                    url = self.s3_client.generate_presigned_url(
                        "get_object",
                        Params={"Bucket": Config.S3_BUCKET_NAME, "Key": s3_key},
                        ExpiresIn=Config.S3_PRESIGNED_URL_EXPIRATION,
                    )
                    
                    # Map filename to presigned URL based on filename prefix
                    if filename.startswith('mask'):
                        presigned_urls['mask'] = url
                    elif filename.startswith('output'):
                        presigned_urls['output'] = url
                    # Add more conditions for other filename patterns if needed
                    else:
                        # Use the filename without extension as key, or keep original logic
                        file_key = os.path.splitext(filename)[0]
                        presigned_urls[file_key] = url
            
            logger.info(f"Uploaded {len(presigned_urls)} files to S3")
            return presigned_urls
        
        except ClientError as e:
            logger.error(f"S3 upload failed: {e}")
            raise HTTPException(status_code=500, detail=f"S3 upload failed: {str(e)}")
        
    def cleanup(self):
        """Cleanup resources."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cleanup completed")


# Global service instance
service = BackgroundRemovalService()


# ============================================================================
# API Models
# ============================================================================

class RemoveBackgroundRequest(BaseModel):
    """Request model for background removal from URL."""
    image_url: HttpUrl
    upload_to_s3: bool = True
    
    @field_validator('image_url')
    @classmethod
    def validate_url(cls, v):
        url_str = str(v)
        if not url_str.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v


class RemoveBackgroundResponse(BaseModel):
    """Response model for background removal."""
    success: bool
    message: str
    job_id: str
    s3_urls: dict = None


# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app."""
    # Startup
    logger.info("Starting Background Removal API")
    Config.validate()
    os.makedirs(Config.TEMP_DIR, exist_ok=True)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Background Removal API")
    service.cleanup()


app = FastAPI(
    title="Background Removal API",
    description="API for removing backgrounds from images",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================================
# Helper Functions
# ============================================================================

def create_job_directory() -> tuple[str, str]:
    """Create unique job directory."""
    job_id = uuid.uuid4().hex
    job_dir = os.path.join(Config.TEMP_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    return job_id, job_dir


def cleanup_job_directory(job_dir: str):
    """Clean up job directory."""
    try:
        shutil.rmtree(job_dir, ignore_errors=True)
        logger.info(f"Cleaned up job directory: {job_dir}")
    except Exception as e:
        logger.error(f"Failed to cleanup {job_dir}: {e}")


async def download_image(url: str, save_path: str) -> str:
    """Download image from URL."""
    import httpx
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            # Check file size
            content_length = response.headers.get('content-length')
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                if size_mb > Config.MAX_FILE_SIZE_MB:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File size ({size_mb:.2f}MB) exceeds limit ({Config.MAX_FILE_SIZE_MB}MB)"
                    )
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            return save_path
    
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Background Removal API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "device": Config.DEVICE,
        "model_loaded": service._model is not None
    }


@app.post("/remove-background/upload", response_model=RemoveBackgroundResponse)
async def remove_background_upload(
    file: UploadFile = File(...),
    upload_to_s3: bool = False,
    background_tasks: BackgroundTasks = None
):
    """
    Remove background from uploaded image.
    
    Args:
        file: Image file to process
        upload_to_s3: Whether to upload results to S3
    
    Returns:
        RemoveBackgroundResponse with job details
    """
    job_id, job_dir = create_job_directory()
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save uploaded file
        input_path = os.path.join(job_dir, f"input{Path(file.filename).suffix}")
        with open(input_path, 'wb') as f:
            content = await file.read()
            
            # Check file size
            size_mb = len(content) / (1024 * 1024)
            if size_mb > Config.MAX_FILE_SIZE_MB:
                raise HTTPException(
                    status_code=400,
                    detail=f"File size ({size_mb:.2f}MB) exceeds limit ({Config.MAX_FILE_SIZE_MB}MB)"
                )
            
            f.write(content)
        
        # Process image
        image = Image.open(input_path)
        bg_removed = service.remove_background(image)
        
        # Save results
        output_path = os.path.join(job_dir, "output.png")
        bg_removed.save(output_path, "PNG", optimize=True)
        
        mask = service.generate_mask(bg_removed)
        mask_path = os.path.join(job_dir, "mask.png")
        mask.save(mask_path, "PNG", optimize=True)
        
        response_data = {
            "success": True,
            "message": "Background removed successfully",
            "job_id": job_id,
            "output_image_url": f"/results/{job_id}/output.png",
            "mask_url": f"/results/{job_id}/mask.png"
        }
        
        # Upload to S3 if requested
        if upload_to_s3:
            try:
                s3_urls = service.upload_to_s3(job_dir, job_id)
                response_data["s3_urls"] = s3_urls
            except Exception as e:
                logger.error(f"S3 upload failed: {e}")
                response_data["message"] += " (S3 upload failed)"
        
        # Schedule cleanup
        if background_tasks:
            background_tasks.add_task(cleanup_job_directory, job_dir)
        
        return RemoveBackgroundResponse(**response_data)
    
    except HTTPException:
        cleanup_job_directory(job_dir)
        raise
    except Exception as e:
        cleanup_job_directory(job_dir)
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/remove-background/url", response_model=RemoveBackgroundResponse)
async def remove_background_url(
    request: RemoveBackgroundRequest,
    # background_tasks: BackgroundTasks
):
    """
    Remove background from image URL.
    
    Args:
        request: Request containing image URL and options
    
    Returns:
        RemoveBackgroundResponse with job details
    """
    job_id, job_dir = create_job_directory()
    
    try:
        # Download 
        print("Request received:")
        print(request)
        image_url = str(request)
        ext = Path(image_url).suffix or '.png'
        input_path = os.path.join(job_dir, f"input{ext}")
        
        await download_image(image_url, input_path)
        
        # Process image
        image = Image.open(input_path)
        bg_removed = service.remove_background(image)
        
        # Save results
        output_path = os.path.join(job_dir, "output.png")
        bg_removed.save(output_path, "PNG", optimize=True)
        
        mask = service.generate_mask(bg_removed)
        mask_path = os.path.join(job_dir, "mask.png")
        mask.save(mask_path, "PNG", optimize=True)
        
        response_data = {
            "success": True,
            "message": "Background removed successfully",
            "job_id": job_id,
            "input_image_url": None,
            "image_url": None,
            "mask_url": None
        }
        
        # Upload to S3 
      
        try:
            s3_urls = service.upload_to_s3(job_dir, job_id)
            response_data["s3_urls"] = s3_urls
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            response_data["message"] += " (S3 upload failed)"
        
        # Schedule cleanup
        # background_tasks.add_task(cleanup_job_directory, job_dir)
        
        return RemoveBackgroundResponse(**response_data)
    
    except HTTPException:
        cleanup_job_directory(job_dir)
        raise
    except Exception as e:
        cleanup_job_directory(job_dir)
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/results/{job_id}/{filename}")
async def get_result_file(job_id: str, filename: str):
    """Retrieve processed result file."""
    from fastapi.responses import FileResponse
    
    file_path = os.path.join(Config.TEMP_DIR, job_id, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path, media_type="image/png")


# if __name__ == "__main__":
#     import uvicorn
    
#     uvicorn.run(
#         "app:app",
#         # host="0.0.0.0",
#         port=8000,
#         reload=True,
#         log_level="info"
#     )
