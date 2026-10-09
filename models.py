from pydantic import BaseModel

class URLFinderAgentResponse(BaseModel):
    url: str | None
    summary: str