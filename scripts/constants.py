import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRUBBED_DIR = ROOT / "scrubbed_docs"
KEYS_DIR = SCRUBBED_DIR / "scrubbed_keys"
PROMPTS_DIR = ROOT / "prompts"
RESULTS_DIR = ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"

MODEL_ALIASES = {"haiku": "claude-haiku-4-5", "sonnet": "claude-sonnet-5", "opus": "claude-opus-4-8"}
PII_PER_PAGE = 100
MAX_PAGE_COUNT = 10

VALID_MASK_RE = re.compile(r"\[PII_\d+\]")
LEADING_ZERO_RE = re.compile(r"\[PII_0\d*\]")
NEAR_MISS_RE = re.compile(r"\[?\s*PII[\s_-]*\d+\s*\]?", re.IGNORECASE)

FAILURE_CLASSES = ("wrong_existing_mask", "nonexistent_mask", "malformed_mask",
                   "no_mask", "correct_mask_wrong_usage", "api_error")

SYSTEM_TEMPLATE = """\
You are an assistant working with a document in which every piece of personally \
identifiable information (PII) has been replaced by a mask of the form [PII_n], \
for example [PII_37]. A separate system maps masks back to real values.

Rules:
1. Never invent, guess, or try to reconstruct the real value behind a mask.
2. Whenever your answer refers to a piece of PII, use its exact mask as it appears \
in the document, copied character for character (e.g. [PII_37]).
3. When asked to perform an action (send an email, call, block, charge, verify), \
call the appropriate tool with the exact mask as the argument value.
4. When asked an informational question, answer concisely in plain text without \
calling any tool, and include the exact mask in your answer.
5. Only ever use masks that literally appear in the document below.

Document:
<document>
{document}
</document>"""

TOOLS = [
    {"name": "send_email", "description": "Send an email to the given recipient address.",
     "input_schema": {"type": "object", "required": ["to"], "properties": {
         "to": {"type": "string", "description": "Recipient email address, given as a PII mask such as [PII_12]."}}}},
    {"name": "call_customer", "description": "Place a phone call to the given number.",
     "input_schema": {"type": "object", "required": ["phone"], "properties": {
         "phone": {"type": "string", "description": "Phone number to call, given as a PII mask such as [PII_12]."}}}},
    {"name": "block_ip", "description": "Block the given IP address at the firewall.",
     "input_schema": {"type": "object", "required": ["ip"], "properties": {
         "ip": {"type": "string", "description": "IP address to block, given as a PII mask such as [PII_12]."}}}},
    {"name": "charge_card", "description": "Process a charge on the given credit card.",
     "input_schema": {"type": "object", "required": ["card_number"], "properties": {
         "card_number": {"type": "string", "description": "Credit card number to charge, given as a PII mask such as [PII_12]."}}}},
    {"name": "verify_identity", "description": "Run identity verification against the given SSN.",
     "input_schema": {"type": "object", "required": ["ssn"], "properties": {
         "ssn": {"type": "string", "description": "SSN to verify, given as a PII mask such as [PII_12]."}}}},
]
