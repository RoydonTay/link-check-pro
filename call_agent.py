import logging
import os
from dotenv import load_dotenv
from langfuse import get_client, observe, propagate_attributes
from browser_use import Agent, ChatGoogle, ChatLiteLLM, Browser
from models import URLFinderAgentResponse

load_dotenv()

logger = logging.getLogger(__name__)


def serialize(value):
    """Convert Browser Use objects to JSON-compatible values."""
    if value is None:
        return None

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)

    if isinstance(value, (list, tuple)):
        return [serialize(item) for item in value]

    if isinstance(value, dict):
        return {str(k): serialize(v) for k, v in value.items()}

    return value


def llm_client(type: str = "google"):
    """Get a Langfuse client for the specified LLM type."""
    client = get_client()

    if type == "google":
        llm = ChatGoogle(model="gemini-3.5-flash-lite")
    elif type == "litellm":
        # Get Azure config from environment
        azure_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        azure_base = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        azure_version = os.getenv("OPENAI_API_VERSION", "2025-01-01-preview")
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5")

        if not azure_key or not azure_base:
            logger.warning("Azure OpenAI not configured, skipping LLM extraction")
            return {}

        # Set env vars for LiteLLM
        os.environ["AZURE_API_KEY"] = azure_key
        os.environ["AZURE_API_BASE"] = azure_base
        os.environ["AZURE_API_VERSION"] = azure_version
        os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = deployment_name
        llm = ChatLiteLLM(
            model=f"azure/{deployment_name}",
            api_key=azure_key,
            api_base=azure_base,
            temperature=0.0,
            max_retries=3,
        )
    else:
        raise ValueError(f"Unsupported LLM type: {type}")

    return client, llm


@observe(name="link-check-agent", as_type="agent")
async def call_link_check_agent(
    scheme_name: str,
    previous_link: str,
    session_id: str,
) -> URLFinderAgentResponse:

    client, llm = llm_client(type="litellm")
    browser = Browser(headless=True)

    task = f"""
    Your task is to help find the page for a scheme / program
    on its providing organization's website from broken links.

    Search the organization's website and verify that the
    scheme / program is still active.

    If it is, provide the exact page link to the scheme / program
    and a brief summary of the information about it.

    Try your best to find the exact page for the scheme / program. Do not just stop at the page that lists all schemes / programs. 

    Otherwise, return None in the URL and provide a brief
    summary of what you found.

    Scheme / Program: "{scheme_name}"
    Previous link: "{previous_link}"
    """

    # Called after each Browser Use agent step
    async def trace_browser_step(
        browser_state,
        agent_output,
        step_number,
    ):
        try:
            state_data = {
                "url": getattr(browser_state, "url", None),
                "title": getattr(browser_state, "title", None),
            }

            output_data = serialize(agent_output)

            # Associate this step with the same session.
            with propagate_attributes(session_id=session_id):

                # Create a child observation under link-check-agent.
                with client.start_as_current_observation(
                    name=f"browser-step-{step_number}",
                    as_type="span",
                    input=state_data,
                ) as observation:

                    observation.update(
                        output=output_data,
                        metadata={
                            "session_id": session_id,
                            "step_number": step_number,
                            "model": "gemini-3.5-flash-lite",
                        },
                    )

            logger.info(
                "Session %s | Browser step %s | URL: %s",
                session_id,
                step_number,
                state_data["url"],
            )

        except Exception:
            logger.exception(
                "Session %s | Failed to trace browser step %s",
                session_id,
                step_number,
            )

    try:
        # Apply session attributes to the agent execution.
        with propagate_attributes(session_id=session_id):

            agent = Agent(
                task=task,
                llm=llm,
                browser=browser,
                output_model_schema=URLFinderAgentResponse,
                register_new_step_callback=trace_browser_step,
            )

            history = await agent.run()

            response = history.structured_output

            if response is None:
                final_text = history.final_result() or ""

                try:
                    response = (
                        URLFinderAgentResponse.model_validate_json(
                            final_text
                        )
                    )
                except Exception:
                    response = URLFinderAgentResponse(
                        url=None,
                        summary=final_text,
                    )

            # Update the enclosing agent observation.
            try:
                client.update_current_span(
                    output={
                        "url": response.url,
                        "summary": response.summary,
                    },
                    metadata={
                        "session_id": session_id,
                        "model": "gemini-3.5-flash-lite",
                        "step_count": history.number_of_steps(),
                        "visited_urls": history.urls()[:20],
                        "has_errors": history.has_errors(),
                    },
                )

            except Exception:
                logger.exception(
                    "Session %s | Failed to update agent span",
                    session_id,
                )

            return response

    finally:
        try:
            await browser.close()
        except Exception:
            logger.exception(
                "Session %s | Failed to close browser",
                session_id,
            )
