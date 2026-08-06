"""Make Take Judge V2 delivery features use the shared EN/ES tokenizer."""
from pathlib import Path

path = Path("worker/take_judge_v2.py")
text = path.read_text(encoding="utf-8")

if "from worker.text_normalization import unicode_word_tokens" in text:
    print("Take Judge Spanish tokenization patch already applied")
    raise SystemExit(0)

old_import = "from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple\n"
new_import = old_import + "\nfrom worker.text_normalization import unicode_word_tokens\n"
if text.count(old_import) != 1:
    raise SystemExit("Unexpected take_judge_v2 import baseline")
text = text.replace(old_import, new_import, 1)

old_tokens = '    tokens = re.findall(r"[A-Za-z0-9\']+", text.lower())\n'
new_tokens = '    tokens = unicode_word_tokens(text)\n'
if text.count(old_tokens) != 1:
    raise SystemExit("Unexpected take_judge_v2 tokenization baseline")
text = text.replace(old_tokens, new_tokens, 1)

path.write_text(text, encoding="utf-8")
print("Applied shared English/Spanish tokenization to Take Judge V2")
