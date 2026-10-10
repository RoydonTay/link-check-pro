import argparse
import asyncio
import csv
from datetime import datetime
import logging
import os
import re
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

# Ensure environment variables are loaded before Langfuse or other clients are initialized
load_dotenv()

from langfuse import get_client, observe, propagate_attributes
from models import URLFinderAgentResponse
from call_agent import call_link_check_agent


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval")


def resolve_input_file(custom_path: str | None) -> Path:
    """Find the scheme input CSV file."""
    if custom_path:
        path = Path(custom_path)
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {custom_path}")
        return path

    base_dir = Path(__file__).resolve().parent
    # Check default names
    candidates = [
        base_dir / "scheme_names_and_links.csv",
        base_dir / "schemes_naem_and_links.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not find input CSV file (checked scheme_names_and_links.csv and schemes_naem_and_links.csv)"
    )


def generate_session_id(run_id: str, idx: int, scheme_name: str) -> str:
    """
    Generate a distinct, human-readable session ID for an individual evaluation.
    This guarantees every eval item receives its own unique session in Langfuse
    while grouping all items under the current evaluation run prefix.
    """
    clean_name = re.sub(r"[^\w\-]+", "_", scheme_name.strip()).strip("_")
    if clean_name:
        clean_name = clean_name[:30]
        return f"{run_id}_item{idx:02d}_{clean_name}"
    return f"{run_id}_item{idx:02d}_{uuid.uuid4().hex[:6]}"



@observe(name="eval-scheme-item")
async def evaluate_single_scheme(
    scheme_name: str,
    previous_link: str,
    session_id: str,
    run_id: str,
    idx: int,
    total: int,
) -> dict:
    """
    Evaluate a scheme with Langfuse observability.

    Hierarchy:
      Experiment run -> run_id (tag)
      Individual scheme -> session_id
      Agent execution -> trace
      Browser steps -> child observations
    """

    found_url: str | None = None
    summary: str = ""
    status: str = "Success"
    error_msg: str | None = None

    client = get_client()

    # Experiment-level and scheme-level attributes
    with propagate_attributes(
        trace_name="eval-scheme-item",
        session_id=session_id,
        tags=[
            "eval",
            "browser-use",
            f"run:{run_id}",
        ],
        metadata={
            "run_id": run_id,
            "row_index": str(idx),
            "total_rows": str(total),
            "scheme_name": scheme_name[:200],
        },
    ):
        try:
            # Browser agent inherits the current trace context.
            # Its callbacks create child observations.
            result = await call_link_check_agent(
                scheme_name=scheme_name,
                previous_link=previous_link,
                session_id=session_id,
            )

            if isinstance(result, URLFinderAgentResponse):
                found_url = result.url
                summary = result.summary

            elif (
                hasattr(result, "structured_output")
                and result.structured_output
            ):
                found_url = result.structured_output.url
                summary = result.structured_output.summary

            else:
                summary = str(result)

            if found_url:
                logger.info(
                    "[%d/%d] Active link found: %s",
                    idx,
                    total,
                    found_url,
                )
            else:
                logger.info(
                    "[%d/%d] No active link found",
                    idx,
                    total,
                )

        except Exception as e:
            status = "Error"
            error_msg = str(e)
            summary = f"Execution error: {e}"

            logger.exception(
                "[%d/%d] Error processing '%s'",
                idx,
                total,
                scheme_name,
            )

        # Record evaluation results on the active trace.
        try:
            client.update_current_span(
                input={
                    "scheme_name": scheme_name,
                    "previous_link": previous_link,
                },
                output={
                    "found_url": found_url,
                    "summary": summary,
                    "status": status,
                },
                metadata={
                    "run_id": run_id,
                    "session_id": session_id,
                    "error": error_msg,
                    "link_found": bool(found_url),
                },
            )

            if status == "Success":
                client.score_current_trace(
                    name="link_found",
                    value=1.0 if found_url else 0.0,
                    data_type="BOOLEAN",
                    comment=(
                        "Active link found"
                        if found_url
                        else "No active link found"
                    ),
                )

                client.score_current_trace(
                    name="eval_success",
                    value=1.0,
                    data_type="BOOLEAN",
                )

            else:
                client.score_current_trace(
                    name="eval_success",
                    value=0.0,
                    data_type="BOOLEAN",
                    comment=error_msg,
                )

        except Exception:
            logger.exception(
                "Failed to record Langfuse evaluation data"
            )

    return {
        "Scheme Name": scheme_name,
        "Previous Link": previous_link,
        "Found URL": found_url or "",
        "Summary": summary,
        "Status": status,
        "Session ID": session_id,
        "Run ID": run_id,
    }


async def run_evaluation(
    input_file: Path,
    output_file: Path,
    limit: int | None = None,
    delay_seconds: float = 1.0,
    run_id: str | None = None,
):
    """
    Iterates through the schemes in the input CSV, calls browser_use for each,
    traces each run with a unique Langfuse session, and streams results to the output CSV.
    """
    logger.info("Reading schemes from: %s", input_file)

    with open(input_file, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        logger.warning("Input file is empty or contains no rows.")
        return

    # Identify relevant columns flexibly
    first_row = rows[0]
    scheme_key = next((k for k in first_row if "scheme" in k.lower()), list(first_row.keys())[0])
    link_key = next((k for k in first_row if "link" in k.lower() or "url" in k.lower()), list(first_row.keys())[1])

    logger.info("Using columns: '%s' for scheme name, '%s' for link", scheme_key, link_key)

    if limit is not None and limit > 0:
        rows = rows[:limit]
        logger.info("Evaluating up to %d row(s)", len(rows))
    else:
        logger.info("Evaluating %d row(s)", len(rows))

    if not run_id:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"eval_run_{timestamp_str}_{uuid.uuid4().hex[:6]}"

    logger.info("Evaluation Run ID: %s", run_id)

    # Check Langfuse credentials and status
    langfuse_client = get_client()
    has_lf_creds = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
    if has_lf_creds:
        host = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST") or "https://cloud.langfuse.com"
        logger.info("Langfuse observability active. Export host: %s", host)
    else:
        logger.warning(
            "Langfuse credentials not detected (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY). "
            "Traces will run locally without remote export. Add credentials to your .env to send traces."
        )

    fieldnames = [
        "Scheme Name",
        "Previous Link",
        "Found URL",
        "Summary",
        "Status",
        "Session ID",
        "Run ID",
    ]

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Initialize output CSV with headers
    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    total = len(rows)
    success_count = 0
    active_found = 0

    try:
        for idx, row in enumerate(rows, start=1):
            scheme_name = (row.get(scheme_key) or "").strip()
            previous_link = (row.get(link_key) or "").strip()
            session_id = generate_session_id(run_id=run_id, idx=idx, scheme_name=scheme_name)

            logger.info(
                "[%d/%d] Processing scheme: '%s' (%s) | Session: %s",
                idx,
                total,
                scheme_name,
                previous_link,
                session_id,
            )

            output_row = await evaluate_single_scheme(
                scheme_name=scheme_name,
                previous_link=previous_link,
                session_id=session_id,
                run_id=run_id,
                idx=idx,
                total=total,
            )

            if output_row["Status"] == "Success":
                success_count += 1
            if output_row["Found URL"]:
                active_found += 1

            # Append row immediately so results are preserved on interruption
            with open(output_file, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(output_row)

            # Flush after each eval item so traces are available in Langfuse immediately
            try:
                langfuse_client.flush()
            except Exception:
                pass

            if idx < total and delay_seconds > 0:
                await asyncio.sleep(delay_seconds)

    finally:
        # Guarantee flushing remaining queued traces before exit
        try:
            langfuse_client.flush()
        except Exception:
            pass

    logger.info("Evaluation complete!")
    logger.info(
        "Total schemes: %d | Successful agent runs: %d | Active links found: %d",
        total,
        success_count,
        active_found,
    )
    logger.info("Results saved to: %s", output_file.resolve())


def main():
    parser = argparse.ArgumentParser(description="Evaluate browser agent against scheme links with Langfuse observability.")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to input CSV (defaults to scheme_names_and_links.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="eval_results.csv",
        help="Path to output CSV (defaults to eval_results.csv)",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Limit number of schemes to evaluate (e.g. --limit 3 for quick test)",
    )
    parser.add_argument(
        "--delay",
        "-d",
        type=float,
        default=1.0,
        help="Delay in seconds between browser agent runs (default: 1.0)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Custom run ID prefix for this batch evaluation (defaults to timestamp-based ID)",
    )

    args = parser.parse_args()

    try:
        input_file = resolve_input_file(args.input)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    output_file = Path(args.output)

    asyncio.run(
        run_evaluation(
            input_file=input_file,
            output_file=output_file,
            limit=args.limit,
            delay_seconds=args.delay,
            run_id=args.run_id,
        )
    )


if __name__ == "__main__":
    main()
