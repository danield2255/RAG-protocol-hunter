from pydantic import BaseModel, Field

class ResponseFormat(BaseModel):
    answer: str = Field(description="The final answer")
    sources: list[str] = Field(description="Sources used")