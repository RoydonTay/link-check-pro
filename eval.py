import argparse
import asyncio
import csv
import logging
import sys
from pathlib import Path

from models import URLFinderAgentResponse

# Import browser_use from call_agent (fallback to browser_use if needed)
try:
    from call_agent import browser_use
except ImportError:
    from browser_use import browser_use

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


async def run_evaluation(
    input_file: Path,
    output_file: Path,
    limit: int | None = None,
    delay_seconds: float = 1.0,
):
    """
    Iterates through the schemes in the input CSV, calls browser_use for each,
    and streams results to the output CSV.
    """
    logger.info("Reading schemes from: %s", input_file)

    with open(input_file, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # Normalize header keys
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

    fieldnames = ["Scheme Name", "Previous Link", "Found URL", "Summary", "Status"]

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Initialize output CSV with headers
    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    total = len(rows)
    success_count = 0
    active_found = 0

    for idx, row in enumerate(rows, start=1):
        scheme_name = (row.get(scheme_key) or "").strip()
        previous_link = (row.get(link_key) or "").strip()

        logger.info("[%d/%d] Processing scheme: '%s' (%s)", idx, total, scheme_name, previous_link)

        found_url: str | None = None
        summary: str = ""
        status: str = "Success"

        try:
            result = await browser_use(scheme_name=scheme_name, previous_link=previous_link)

            if isinstance(result, URLFinderAgentResponse):
                found_url = result.url
                summary = result.summary
            elif hasattr(result, "structured_output") and result.structured_output:
                found_url = result.structured_output.url
                summary = result.structured_output.summary
            else:
                summary = str(result)

            success_count += 1
            if found_url:
                active_found += 1
                logger.info("[%d/%d] Active link found: %s", idx, total, found_url)
            else:
                logger.info("[%d/%d] No active link found (returned None)", idx, total)

        except Exception as e:
            status = "Error"
            summary = f"Execution error: {e}"
            logger.error("[%d/%d] Error processing '%s': %s", idx, total, scheme_name, e)

        output_row = {
            "Scheme Name": scheme_name,
            "Previous Link": previous_link,
            "Found URL": found_url or "",
            "Summary": summary,
            "Status": status,
        }

        # Append row immediately so results are preserved on interruption
        with open(output_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(output_row)

        if idx < total and delay_seconds > 0:
            await asyncio.sleep(delay_seconds)

    logger.info("Evaluation complete!")
    logger.info("Total schemes: %d | Successful agent runs: %d | Active links found: %d", total, success_count, active_found)
    logger.info("Results saved to: %s", output_file.resolve())


def main():
    parser = argparse.ArgumentParser(description="Evaluate browser agent against scheme links.")
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
        )
    )


if __name__ == "__main__":
    main()
