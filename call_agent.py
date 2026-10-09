from dotenv import load_dotenv
import asyncio
from models import URLFinderAgentResponse
from browser_use import Agent, ChatGoogle, Browser, BrowserSession 

load_dotenv()

async def browser_use(scheme_name: str, previous_link: str) -> URLFinderAgentResponse:
    llm = ChatGoogle(model="gemini-3.5-flash-lite")
    browser = Browser(headless=True)
    task = f"""
    Your task is to help find the page for a scheme / program on from its providing organization's website from broken links.
    Search the organization's website and verify that the scheme / program is still active.
    If it is, provide the exact page link to the scheme / program and provide a brief summary of the information about it.
    Else, return None in the url and provide a brief summary of what you found.

    Scheme / Program: "{scheme_name}"
    Previous link: "{previous_link}"
    """
    try:
        agent = Agent(task=task, llm=llm, browser=browser, output_model_schema=URLFinderAgentResponse)
        history = await agent.run()
        response = history.structured_output
        if response is None:
            final_text = history.final_result() or ""
            try:
                response = URLFinderAgentResponse.model_validate_json(final_text)
            except Exception:
                response = URLFinderAgentResponse(url=None, summary=final_text)
        return response
    finally:
        try:
            await browser.close()
        except Exception:
            pass
