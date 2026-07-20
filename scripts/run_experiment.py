import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

import config
from constants import (FAILURE_CLASSES, KEYS_DIR, LEADING_ZERO_RE, MAX_PAGE_COUNT,
                       MODEL_ALIASES, NEAR_MISS_RE, PII_PER_PAGE, PROMPTS_DIR,
                       RAW_DIR, RESULTS_DIR, SCRUBBED_DIR, SYSTEM_TEMPLATE,
                       TOOLS, VALID_MASK_RE)


def classify_failure(text, keys):
    """
    Classify an incorrect response by the kind of mask error it contains.

    Args:
        text (str) : response text or tool argument text to inspect
        keys (dict) : masks that exist in the document, mapped to their real values

    Returns:
        str : one of "wrong_existing_mask", "nonexistent_mask", "malformed_mask", "no_mask"
    """
    masks = VALID_MASK_RE.findall(text)
    if any(m in keys for m in masks):
        return "wrong_existing_mask"
    real = [m for m in masks if not LEADING_ZERO_RE.fullmatch(m)]
    if real:
        return "nonexistent_mask"
    if masks or NEAR_MISS_RE.search(text):
        return "malformed_mask"
    return "no_mask"


def score_prose(text, expected_mask, keys):
    """
    Score a prose answer against the expected mask.

    Args:
        text (str) : model response text
        expected_mask (str) : mask the answer is expected to contain
        keys (dict) : masks that exist in the document, mapped to their real values

    Returns:
        tuple : (correct (bool), classification (str), spurious_masks (list))
    """
    masks = VALID_MASK_RE.findall(text)
    spurious = sorted(set(m for m in masks if m != expected_mask))
    if expected_mask in masks:
        return True, "correct", spurious
    return False, classify_failure(text, keys), spurious


def score_tool_call(text, tool_calls, expected_mask, expected_tool, keys):
    """
    Score a tool-call answer against the expected tool and mask.

    Args:
        text (str) : model response text, judged when no tool call was made
        tool_calls (list) : tool calls extracted from the response
        expected_mask (str) : mask the tool argument is expected to be
        expected_tool (str) : name of the tool that should have been called
        keys (dict) : masks that exist in the document, mapped to their real values

    Returns:
        tuple : (correct (bool), classification (str), spurious_masks (list))
    """
    if tool_calls:
        call = tool_calls[0]
        args_text = " ".join(str(v) for v in call["args"].values())
        masks = VALID_MASK_RE.findall(args_text)
        spurious = sorted(set(m for m in masks if m != expected_mask))
        if call["name"] == expected_tool and args_text.strip() == expected_mask:
            return True, "correct", spurious
        if expected_mask in masks:
            return False, "correct_mask_wrong_usage", spurious
        return False, classify_failure(args_text, keys), spurious
    masks = VALID_MASK_RE.findall(text)
    spurious = sorted(set(m for m in masks if m != expected_mask))
    if expected_mask in masks:
        return False, "correct_mask_wrong_usage", spurious
    return False, classify_failure(text, keys), spurious


def load_pages(doc_num):
    """
    Load a scrubbed document and split it into pages.

    Args:
        doc_num (int) : document number to load

    Returns:
        list : page texts, each starting with its "=== PAGE n ===" header
    """
    text = (SCRUBBED_DIR / f"doc_{doc_num}.txt").read_text()
    chunks = re.split(r"^(=== PAGE \d+ ===)$", text, flags=re.M)
    pages = []
    for i in range(1, len(chunks), 2):
        pages.append(chunks[i] + chunks[i + 1])
    return pages


def load_questions(doc_num, page_count, limit):
    """
    Load the question specs for the first page_count pages of a document.

    Args:
        doc_num (int) : document number to load questions for
        page_count (int) : number of leading pages whose questions to include
        limit (int) : maximum number of questions to return, 0 for all

    Returns:
        list : question dicts with page, question, and spec fields
    """
    data = json.loads((PROMPTS_DIR / f"doc_{doc_num}.json").read_text())
    questions = []
    for page in range(1, page_count + 1):
        for question, spec in data[str(page)].items():
            questions.append({"page": page, "question": question, **spec})
    return questions[:limit] if limit else questions


def extract_text(message):
    """
    Concatenate the text blocks of a model response.

    Args:
        message (AIMessage) : response message returned by the model

    Returns:
        str : plain text content of the message
    """
    if isinstance(message.content, str):
        return message.content
    return "".join(block.get("text", "") for block in message.content
                   if isinstance(block, dict) and block.get("type") == "text")


def ask(llm, system_text, item, keys):
    """
    Send one question to the model and score the response.

    Args:
        llm (ChatAnthropic) : model client with the experiment tools bound
        system_text (str) : system prompt containing the document context
        item (dict) : question spec with question, mode, and expected_mask fields
        keys (dict) : masks that exist in the document, mapped to their real values

    Returns:
        dict : the question fields plus scoring, response, latency, and usage data
    """
    messages = [
        ("system", [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]),
        ("human", item["question"]),
    ]
    start = time.monotonic()
    try:
        response = llm.invoke(messages)
    except Exception as exc:
        return {**item, "correct": False, "classification": "api_error",
                "spurious_masks": [], "response_text": f"{type(exc).__name__}: {exc}",
                "tool_calls": [], "latency_s": round(time.monotonic() - start, 3), "usage": {}}
    latency = round(time.monotonic() - start, 3)

    text = extract_text(response)
    tool_calls = [{"name": c["name"], "args": c["args"]} for c in (response.tool_calls or [])]

    if item["mode"] == "prose":
        correct, classification, spurious = score_prose(text, item["expected_mask"], keys)
    else:
        correct, classification, spurious = score_tool_call(
            text, tool_calls, item["expected_mask"], item["tool"], keys)

    return {**item, "correct": correct, "classification": classification,
            "spurious_masks": spurious, "response_text": text, "tool_calls": tool_calls,
            "latency_s": latency, "usage": response.usage_metadata or {}}


def run_cell(llm, model, doc_num, page_count, pages, keys, questions, concurrency, pbar):
    """
    Run every question of one (model, doc, page_count) experiment cell.

    Args:
        llm (ChatAnthropic) : model client with the experiment tools bound
        model (str) : model id, stamped onto each record
        doc_num (int) : document number, stamped onto each record
        page_count (int) : number of pages included in the context
        pages (list) : all page texts of the document
        keys (dict) : masks that exist in the document, mapped to their real values
        questions (list) : question specs to run
        concurrency (int) : worker threads for parallel requests
        pbar (tqdm) : progress bar advanced once per completed question

    Returns:
        list : one scored record dict per question
    """
    system_text = SYSTEM_TEMPLATE.format(document="\n\n".join(pages[:page_count]))
    records = [ask(llm, system_text, questions[0], keys)]
    pbar.update(1)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for record in pool.map(lambda item: ask(llm, system_text, item, keys), questions[1:]):
            records.append(record)
            pbar.update(1)
    for record in records:
        record.update({"model": model, "doc": doc_num, "page_count": page_count})
    return records


def summarize(records):
    """
    Aggregate records into per (model, doc, page_count, mode) summary rows.

    Args:
        records (list) : scored record dicts from all cells

    Returns:
        list : summary row dicts with accuracy and per-class failure rates
    """
    cells = defaultdict(list)
    for r in records:
        cells[(r["model"], r["doc"], r["page_count"], r["mode"])].append(r)

    rows = []
    for (model, doc, page_count, mode), recs in sorted(cells.items()):
        n = len(recs)
        counts = defaultdict(int)
        for r in recs:
            counts[r["classification"]] += 1
        rows.append({
            "model": model, "doc": doc, "page_count": page_count,
            "n_pii_in_context": page_count * PII_PER_PAGE, "mode": mode, "n_questions": n,
            "accuracy": round(sum(r["correct"] for r in recs) / n, 4),
            **{f"{cls}_rate": round(counts[cls] / n, 4) for cls in FAILURE_CLASSES},
        })
    return rows


def write_summary_csv(rows):
    """
    Write summary rows to results/summary.csv.

    Args:
        rows (list) : summary row dicts from summarize

    Returns:
        None : writes the CSV file as a side effect
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    columns = ["model", "doc", "page_count", "n_pii_in_context", "mode", "n_questions",
               "accuracy"] + [f"{cls}_rate" for cls in FAILURE_CLASSES]
    lines = [",".join(columns)]
    lines += [",".join(str(row[c]) for c in columns) for row in rows]
    (RESULTS_DIR / "summary.csv").write_text("\n".join(lines) + "\n")


def print_summary(records, rows):
    """
    Print the output locations and the final accuracy table.

    Args:
        records (list) : scored record dicts from all cells
        rows (list) : summary row dicts from summarize

    Returns:
        None : prints the summary to stdout
    """
    print(f"\nWrote {len(rows)} summary rows to results/summary.csv "
          f"and {len(records)} raw records under results/raw/.")

    acc = defaultdict(lambda: [0, 0])
    for r in records:
        key = (r["model"], r["page_count"], r["mode"])
        acc[key][0] += r["correct"]
        acc[key][1] += 1

    models = sorted(set(r["model"] for r in records))
    page_counts = sorted(set(r["page_count"] for r in records))

    def fmt(model, page_count, mode):
        """
        Format the accuracy of one table cell.

        Args:
            model (str) : model id of the cell
            page_count (int) : page count of the cell
            mode (str) : "prose" or "tool_call"

        Returns:
            str : accuracy percentage, or a dash when the cell has no records
        """
        ok, n = acc[(model, page_count, mode)]
        return f"{ok / n:6.1%}" if n else "     -"

    print("\nAccuracy by model and page_count (aggregated over docs), prose / tool_call:")
    header = f"{'pages':>5} {'n_pii':>6}" + "".join(f" | {m:^17}" for m in models)
    print(header)
    print("-" * len(header))
    for page_count in page_counts:
        row = f"{page_count:>5} {page_count * PII_PER_PAGE:>6}"
        for model in models:
            row += f" | {fmt(model, page_count, 'prose')} {fmt(model, page_count, 'tool_call')}"
        print(row)


def main():
    """
    Run the experiment grid described by config.py, and write and print results.

    Returns:
        None : writes raw records and summary.csv, then prints the final summary
    """
    if config.DOCS and config.NUM_DOCS:
        sys.exit("error: set only one of config.DOCS and config.NUM_DOCS")
    if config.PAGES and config.MAX_PAGES:
        sys.exit("error: set only one of config.PAGES and config.MAX_PAGES")

    models = [MODEL_ALIASES.get(m, m) for m in config.MODELS]
    available_docs = sorted(int(p.stem.split("_")[1]) for p in SCRUBBED_DIR.glob("doc_*.txt"))
    docs = config.DOCS if config.DOCS else available_docs[:config.NUM_DOCS] if config.NUM_DOCS else available_docs
    page_counts = config.PAGES if config.PAGES else list(range(1, (config.MAX_PAGES or MAX_PAGE_COUNT) + 1))

    bad = [d for d in docs if d not in available_docs]
    if bad:
        sys.exit(f"error: unknown document number(s) {bad}; available: {available_docs}")
    bad = [k for k in page_counts if not 1 <= k <= MAX_PAGE_COUNT]
    if bad:
        sys.exit(f"error: page_count values must be in 1..{MAX_PAGE_COUNT}, got {bad}")

    per_model_q = sum(min(config.LIMIT, 10 * k) if config.LIMIT else 10 * k
                      for _ in docs for k in page_counts)

    from langchain_anthropic import ChatAnthropic

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_records = []
    with tqdm(total=len(models) * per_model_q, unit="q") as pbar:
        for model in models:
            llm = ChatAnthropic(model=model, max_tokens=1024, max_retries=8,
                                timeout=300).bind_tools(TOOLS)
            for doc_num in docs:
                pages = load_pages(doc_num)
                keys = json.loads((KEYS_DIR / f"doc_{doc_num}.json").read_text())
                for page_count in page_counts:
                    out_path = RAW_DIR / f"{model}_doc{doc_num}_pages{page_count}.jsonl"
                    pbar.set_postfix_str(out_path.stem)
                    if out_path.exists() and not config.OVERWRITE:
                        records = [json.loads(line) for line in out_path.read_text().splitlines()]
                        pbar.update(len(records))
                    else:
                        questions = load_questions(doc_num, page_count, config.LIMIT)
                        records = run_cell(llm, model, doc_num, page_count, pages, keys,
                                           questions, config.CONCURRENCY, pbar)
                        out_path.write_text("".join(json.dumps(r) + "\n" for r in records))
                    all_records += records

    if not all_records:
        sys.exit("No records produced.")

    rows = summarize(all_records)
    write_summary_csv(rows)
    print_summary(all_records, rows)


if __name__ == "__main__":
    main()
