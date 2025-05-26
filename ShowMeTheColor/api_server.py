# api_server.py

import os
import shutil
import tempfile
import sys
from typing import Dict, Any, List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import logging
import httpx # S3에서 이미지 다운로드를 위해 추가

# 응답 및 요청 모델 정의 (FastAPI Pydantic) 임포트
# 같은 패키지 내의 schemas.py에서 모델들을 임포트합니다.
from .schemas import ( # <-- 여기에 점(.) 하나 추가!
    AnalyzeS3ImageRequest,
    AnalyzeS3ImageResponse,
)

# 로깅 설정: 서버 동작 과정을 콘솔에서 볼 수 있게 합니다.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 동적 경로 추가 ---
# personal_color_analysis 패키지가 올바르게 임포트되도록 합니다.
# api_server.py가 BUILDUP/ShowMeTheColor/ 안에 있으므로,
# 'src' 폴더는 api_server.py와 같은 'ShowMeTheColor' 폴더 안에 있는 형제 폴더입니다.
current_dir = os.path.dirname(os.path.abspath(__file__))

# 여기가 수정된 부분입니다: 'ShowMeTheColor'를 한 번 더 붙이지 않습니다.
src_path = os.path.join(current_dir, 'src')

# src_path가 존재하고 유효한지 확인
if not os.path.exists(src_path):
    logger.critical(f"FATAL ERROR: Could not find 'src' directory at {src_path}. "
                    f"Please ensure your project structure is correct.")
    raise RuntimeError("Required 'src' directory not found. Server cannot start.")
else:
    logger.info(f"Adding '{src_path}' to sys.path for module imports.")
sys.path.insert(0, src_path)

# --- 핵심 분석 함수 임포트 ---
# personal_color.py는 'ShowMeTheColor/src/personal_color_analysis/' 안에 있으므로,
# 'personal_color_analysis' 패키지에서 'personal_color' 모듈을 임포트합니다.
try:
    from personal_color_analysis import personal_color
    logger.info("Successfully imported 'personal_color' module from 'personal_color_analysis' package.")
except ImportError as e:
    logger.critical(f"FATAL ERROR: Could not import 'personal_color' from 'personal_color_analysis'. "
                    f"Please ensure ShowMeTheColor/src/personal_color_analysis/personal_color.py exists and the package structure is correct. Error: {e}")
    personal_color = None # 불러오지 못했음을 표시


# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI(
    title="Personal Color Analysis API",
    description="Upload an image to get personal color analysis results (using existing code).",
    version="1.0.0"
)

# --- 유틸리티 함수: 이미지 파일 유효성 검사 ---
def is_image_file(filename: str) -> bool:
    """주어진 파일 이름이 일반적인 이미지 확장자를 가지는지 확인합니다."""
    return filename.lower().endswith(('.png', '.jpg', '.jpeg'))

# --- API 엔드포인트: 기본 경로 ---
@app.get("/")
async def root():
    print("서버 가동 중")
    return {"message": "피부 분석 서버입니다."}

### API 엔드포인트: `/analyze-image/` (기존 UploadFile 방식)

@app.post("/analyze-image/", summary="Analyze a single image for personal color")
async def analyze_single_image(
    file: UploadFile = File(..., description="The image file to analyze (PNG, JPG, JPEG).")
) -> Dict[str, Any]: # 여기서는 JSONResponse를 반환하므로 Dict[str, Any]를 유지합니다.
    """
    **단일 이미지**를 업로드하여 퍼스널 컬러 분석을 수행합니다.

    - **file**: 업로드할 이미지 파일. 지원되는 형식: PNG, JPG, JPEG.
    """
    if personal_color is None:
        logger.error("Personal color module was not loaded correctly at server startup. Service is unavailable.")
        raise HTTPException(
            status_code=503, # Service Unavailable
            detail="Personal color analysis service is currently unavailable due to a server configuration issue. Please check server logs."
        )

    logger.info(f"Received request for image analysis: {file.filename}")

    # 1. 파일 유형 검사
    if not is_image_file(file.filename):
        logger.warning(f"Invalid file type uploaded: {file.filename}. Rejecting request.")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type for '{file.filename}'. Only PNG, JPG, JPEG files are allowed."
        )

    # 2. 임시 디렉토리에 파일 저장
    # 'tempfile.TemporaryDirectory()'는 'with' 블록이 끝나면 자동으로 디렉토리와 그 안의 파일을 삭제합니다.
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_filepath = os.path.join(tmp_dir, file.filename)
        logger.info(f"Saving uploaded file temporarily to: {temp_filepath}")

        try:
            # 업로드된 파일의 내용을 비동기적으로 읽어 임시 파일에 씁니다.
            with open(temp_filepath, "wb") as buffer:
                while True:
                    chunk = await file.read(1024 * 1024) # 1MB 청크
                    if not chunk:
                        break
                    buffer.write(chunk)
            
            logger.info(f"File '{file.filename}' saved successfully. Calling personal color analysis function.")
            
            # personal_color.analysis 함수로부터 결과를 받습니다.
            analysis_result_tone = personal_color.analysis(temp_filepath)
            
            logger.info(f"Analysis process initiated for '{file.filename}'Result: {analysis_result_tone}. Check server console for detailed output.")
            
            # 4. JSON 응답 반환
            # 이제 결과값을 JSON 응답에 포함시킵니다.
            return JSONResponse(content={
                "message": "Image analysis successful.",
                "filename": file.filename,
                "personal_color_tone": analysis_result_tone,
                "note": "The detailed result is now directly included in this JSON response."
            })

        except HTTPException: # FastAPI의 HTTP 예외는 다시 발생시킵니다.
            raise
        except Exception as e:
            # 파일 처리 또는 분석 함수 내부에서 발생할 수 있는 예상치 못한 모든 오류를 처리합니다.
            logger.error(f"Unexpected error processing image '{file.filename}': {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error occurred while processing image '{file.filename}': {e}. Please check server logs."
            )

### API 엔드포인트: `/analyze-s3-image/` (S3 URL로 이미지 분석)

@app.post("/analyze-s3-image/", response_model=AnalyzeS3ImageResponse, summary="Analyze an image from an S3 URL")
async def analyze_image_from_s3(request: AnalyzeS3ImageRequest):
    """
    **S3 URL**을 통해 이미지를 받아 퍼스널 컬러 분석을 수행합니다.

    - **s3_url**: 분석할 이미지의 S3 URL.
    - **filename**: 원래 이미지 파일 이름 (임시 파일 저장 시 사용).
    """
    if personal_color is None:
        logger.error("Personal color module was not loaded correctly. Service unavailable.")
        raise HTTPException(
            status_code=503,
            detail="Personal color analysis service is unavailable."
        )

    logger.info(f"Received request for S3 image analysis: {request.s3_url} (filename: {request.filename})")

    image_contents = None
    try:
        # httpx를 사용하여 S3 URL에서 이미지 데이터를 비동기적으로 다운로드합니다.
        # 이 예시는 공개적으로 접근 가능한 S3 URL 또는 미리 서명된(presigned) URL을 가정합니다.
        # 만약 S3 버킷이 비공개(private)이고 AWS 자격 증명을 사용하여 접근해야 한다면,
        # boto3 라이브러리를 사용하여 S3 client.get_object를 호출해야 합니다.
        async with httpx.AsyncClient() as client:
            response = await client.get(request.s3_url)
            response.raise_for_status() # HTTP 4xx/5xx 에러 발생 시 예외를 발생시킵니다.
            image_contents = response.content # bytes 형태로 이미지 데이터 받기

        if not image_contents:
            raise ValueError("Downloaded image content is empty.")

    except httpx.HTTPStatusError as exc:
        logger.error(f"Failed to download image from S3 URL: {request.s3_url} - {exc.response.status_code} - {exc.response.text}")
        raise HTTPException(status_code=400, detail=f"Failed to download image from S3: {exc.response.text}")
    except httpx.RequestError as exc:
        logger.error(f"Network error during S3 image download: {request.s3_url} - {exc}")
        raise HTTPException(status_code=500, detail=f"Network error downloading image from S3: {exc}")
    except ValueError as e:
        logger.error(f"Error with downloaded content: {e}")
        raise HTTPException(status_code=500, detail=f"Invalid image content: {e}")
    except Exception as e:
        logger.error(f"Unexpected error downloading image from S3: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during S3 download: {e}")

    # 2. 다운로드된 이미지 데이터를 임시 파일로 저장
    # 'tempfile.TemporaryDirectory()'는 'with' 블록이 끝나면 자동으로 디렉토리와 그 안의 파일을 삭제합니다.
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_filepath = os.path.join(tmp_dir, request.filename)
        logger.info(f"Saving downloaded S3 file temporarily to: {temp_filepath}")

        try:
            with open(temp_filepath, "wb") as buffer:
                buffer.write(image_contents)
            
            logger.info(f"S3 image '{request.filename}' saved successfully for analysis. Calling personal color analysis function.")
            
            # personal_color.analysis 함수 호출 (이제 'tone' 값을 반환합니다)
            analysis_result_tone = personal_color.analysis(temp_filepath)
            
            logger.info(f"Analysis completed for '{request.filename}'. Result: {analysis_result_tone}")
            
            # JSON 응답 반환 (AnalyzeS3ImageResponse 모델에 맞춰)
            return AnalyzeS3ImageResponse(
                message="Image analysis successful from S3 URL.",
                filename=request.filename,
                personal_color_tone=analysis_result_tone # 분석 결과 포함
            )

        except Exception as e:
            logger.error(f"Error during analysis of S3 image '{request.filename}': {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error during analysis of S3 image '{request.filename}': {e}"
            )