from pydantic import BaseModel

# 이미지 분석 요청 데이터 모델
class AnalyzeS3ImageRequest(BaseModel):
    s3_url: str
    filename: str

# 이미지 분석 응답 데이터 모델
class AnalyzeS3ImageResponse(BaseModel):
    message: str
    filename: str
    personal_color_tone: str # 분석 결과가 문자열이라고 가정


# (필요하다면 다른 모델 클래스들도 여기에 추가)