"""Optional local paths used by legacy data/model utilities.

The public runtime never assumes a cluster filesystem. Set these variables only
when using the corresponding legacy utilities.
"""

import os


DATA_PATH = os.getenv("AGENQA_DATA_PATH", "")
QWEN_DIR = os.getenv("AGENQA_QWEN_DIR", "")
Qwen3_235B_A22B_Thinking_2507 = os.getenv("AGENQA_QWEN3_235B_PATH", "")
