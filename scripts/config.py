# Which models to evaluate.
# Accepts the shorthands "haiku", "sonnet", "opus" (expanded via
# constants.MODEL_ALIASES) or any full model id such as "claude-sonnet-5".
#
# Use cases:
#   ["haiku"]                    -> cheap smoke test of the whole pipeline
#   ["haiku", "sonnet", "opus"]  -> the full cross-model comparison
MODELS = ["haiku", "sonnet", "opus"]

# Which documents to run, chosen by AT MOST one of DOCS / NUM_DOCS
# (setting both is an error).
#
# DOCS: explicit document numbers, matching scrubbed_docs/doc_<n>.txt.
# NUM_DOCS: take the first N available documents.
# Leave both as None to run every available document.
#
# Use cases:
#   DOCS = [1, 3]  -> re-run only the documents that showed anomalies
#   NUM_DOCS = 2   -> a smaller, faster grid while iterating on prompts
DOCS = None
NUM_DOCS = None

# How much document context each cell gets, chosen by AT MOST one of
# PAGES / MAX_PAGES (setting both is an error).
#
# PAGES: explicit page_count values; a value K puts pages 1..K in context,
#        i.e. K * 100 PII masks the model must keep straight.
# MAX_PAGES: run every page_count from 1 up to K.
# Leave both as None to run every page_count from 1 to 10.
#
# Use cases:
#   PAGES = [1, 5, 10]  -> probe short, medium, and long contexts only
#   MAX_PAGES = 3       -> a quick scaling curve without the expensive long contexts
PAGES = None
MAX_PAGES = None

# Run only the first N questions of each cell; 0 runs all of them.
#
# Use case: LIMIT = 5 keeps a smoke test down to a handful of API calls
# per cell while still exercising loading, scoring, and summarizing.
LIMIT = 0

# How many questions of a cell are sent to the API in parallel.
#
# Raise it to finish large grids faster if your rate limits allow; lower it
# if requests start failing and showing up as api_error records.
CONCURRENCY = 8

# Re-run cells whose raw output file already exists under results/raw/.
#
# Leave False to resume an interrupted run without repaying for finished
# cells. Set True after changing the prompt, questions, or scoring, so the
# summary is not mixed from stale and fresh records.
OVERWRITE = False
