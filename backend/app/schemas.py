from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class VideoSummaryResponse(BaseModel):
    main_idea: str
    detailed_summary: str
    key_insights: list[str]
    highlights: list[str]
    final_takeaway: str
