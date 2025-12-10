from pydantic import BaseModel
from pydantic import Field

class GroundedAnswer(BaseModel):
    answer: bool = Field(description="True if 'yes', False if 'no'")
    evidence: str = Field(description="The evidence from the provided material that supports the answer.")

