"""Run an LLM on the VNHSGE item bank (official exams 2019-2023).

Supports any OpenAI-compatible endpoint (OpenAI, OpenRouter, Google Gemini,
DeepSeek, local Ollama / LM Studio / vLLM) and the Anthropic API.

Examples
--------
# Ollama local (free; install from ollama.com, then `ollama pull qwen2.5:7b`)
python 02_run_llm.py --provider openai --base-url http://localhost:11434/v1 \
       --api-key ollama --model qwen2.5:7b

# Google Gemini (requires your own credentials and provider access)
python 02_run_llm.py --provider openai \
       --base-url https://generativelanguage.googleapis.com/v1beta/openai/ \
       --api-key $GEMINI_API_KEY --model gemini-2.5-flash

# OpenAI / Anthropic
python 02_run_llm.py --provider openai --model gpt-4o-mini            # uses $OPENAI_API_KEY
python 02_run_llm.py --provider anthropic --model claude-haiku-5-5    # uses $ANTHROPIC_API_KEY

Conditions
----------
--condition original : options in the published order
--condition shuffled : options permuted with a fixed seed per item (robustness /
                       option-order probe; not proof of contamination)
--mode direct        : answer with one letter
--mode cot           : brief reasoning, then "Đáp án: X"
--vision             : send question images to multimodal models

Output: outputs/runs/<model>__<condition>__<mode>[__vision].jsonl (resumable).
"""
import argparse, base64, json, os, random, re, sys, time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ITEMBANK = ROOT / "outputs/itembank_eval.csv"
IMGROOT = ROOT / "data/raw/vnhsge/VNHSGE-main/Dataset/VNHSGE-V/JSON format/eval"
RUNS = ROOT / "outputs/runs"

SYSTEM = ("Bạn là một thí sinh đang làm bài thi tốt nghiệp trung học phổ thông quốc gia "
          "của Việt Nam. Mỗi câu hỏi có đúng một phương án đúng trong A, B, C, D.")
SYSTEM_OPEN = ("Bạn là một thí sinh đang làm bài thi tốt nghiệp trung học phổ thông quốc gia "
               "của Việt Nam. Hãy giải bài toán và trả lời bằng giá trị số.")
INSTR_OPEN = ("Hãy giải ngắn gọn (tối đa 200 từ), sau đó kết thúc bằng một dòng duy nhất có dạng: "
              "'Đáp án: <giá trị số>' (chỉ ghi số, không ghi đơn vị).")
INSTR = {
    "direct": "Chỉ trả lời bằng một chữ cái duy nhất (A, B, C hoặc D), không giải thích.",
    "cot": ("Hãy suy luận ngắn gọn từng bước (tối đa 150 từ), sau đó kết thúc bằng một "
            "dòng duy nhất có dạng: 'Đáp án: X' với X là A, B, C hoặc D."),
}
OPT_RE = re.compile(r"(?m)^\s*([A-D])[\.\)]\s*")


def split_options(q):
    """Return (stem, {letter: text}) or (q, None) if options cannot be parsed."""
    marks = list(OPT_RE.finditer(q))
    letters = [m.group(1) for m in marks]
    if letters[-4:] != ["A", "B", "C", "D"]:
        return q, None
    marks = marks[-4:]
    stem = q[: marks[0].start()].rstrip()
    opts = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i < 3 else len(q)
        opts[m.group(1)] = q[m.end():end].strip()
    return stem, opts


def build_item(row, condition, seed=2026):
    """Return (prompt_text, gold_letter, mapping) for the given condition."""
    q, gold = row.question, row.key
    if condition == "original":
        return q, gold, None
    stem, opts = split_options(q)
    if opts is None:  # cannot shuffle safely -> keep original, flag it
        return q, gold, "unshuffled"
    rng = random.Random(f"{seed}-{row.id}")
    order = list("ABCD")
    rng.shuffle(order)  # new position i shows old option order[i]
    new_gold = "ABCD"[order.index(gold)]
    body = "\n".join(f"{'ABCD'[i]}. {opts[o]}" for i, o in enumerate(order))
    return f"{stem}\n{body}", new_gold, "".join(order)


def parse_letter(text, mode):
    if not text:
        return None
    t = text.strip()
    m = re.search(r"Đáp án\s*[:：]?\s*\**\s*([A-D])\b", t, re.I)
    if m:
        return m.group(1).upper()
    if mode == "direct":
        m = re.match(r"\W*([A-D])\b", t)
        if m:
            return m.group(1)
    m = re.findall(r"\b([A-D])\b", t)
    return m[-1] if m else None


def parse_number(text):
    if not text:
        return None
    # Require the requested final answer; never grade a number from the reasoning,
    # a numerator of a fraction, or the first component of a vector as a scalar.
    lines = re.findall(r'Đáp án\s*[:：]\s*([^\n]+)', text, re.I)
    if not lines:
        return None
    value = lines[-1].strip().strip('* $').strip()
    if value.startswith('\\boxed{') and value.endswith('}'):
        value = value[7:-1].strip()
    if re.fullmatch(r'-?\d{1,3}(?:,\d{3}){2,}', value):
        value = value.replace(',', '')
    m = re.fullmatch(r'[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?', value)
    try:
        return float(value.replace(',', '.')) if m else None
    except ValueError:
        return None


def item_images(row):
    """Image files attached to a question (field Image_Question, may list several)."""
    if not isinstance(row.image_question, str) or not row.image_question.strip():
        return []
    refs = re.split(r"[;,\s]+", row.image_question.strip())
    return [IMGROOT / row.subject / r for r in refs if r and (IMGROOT / row.subject / r).exists()]


def call_openai(client, model, system, user, images, max_tokens, temperature):
    content = [{"type": "text", "text": user}]
    for f in images:
        b64 = base64.b64encode(f.read_bytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    r = client.chat.completions.create(
        model=model, temperature=temperature, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": content if images else user}])
    u = getattr(r, "usage", None)
    return r.choices[0].message.content, (u.prompt_tokens if u else None), (u.completion_tokens if u else None)


def call_anthropic(client, model, system, user, images, max_tokens, temperature):
    content = []
    for f in images:
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                        "data": base64.b64encode(f.read_bytes()).decode()}})
    content.append({"type": "text", "text": user})
    r = client.messages.create(model=model, system=system, max_tokens=max_tokens,
                               temperature=temperature,
                               messages=[{"role": "user", "content": content}])
    text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    return text, r.usage.input_tokens, r.usage.output_tokens


def call_ollama(base_url, model, system, user, images, max_tokens, temperature):
    """Use the same OpenAI-compatible local endpoint without an SDK dependency."""
    if images:
        raise ValueError('Native local provider is text-only; use openai for vision')
    payload = dict(model=model, temperature=temperature, max_tokens=max_tokens,
                   messages=[dict(role='system', content=system),
                             dict(role='user', content=user)], stream=False)
    request = urllib.request.Request(base_url.rstrip('/') + '/chat/completions',
                                     data=json.dumps(payload).encode('utf-8'),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    usage = result.get('usage', {})
    return result['choices'][0]['message']['content'], usage.get('prompt_tokens'), usage.get('completion_tokens')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["openai", "ollama", "anthropic", "dummy"], required=True,
                    help="dummy = random guesser, for testing the pipeline / chance baseline")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--itembank", default=str(ITEMBANK),
                    help="item bank CSV (default: VNHSGE eval; or outputs/itembank_hard.csv)")
    ap.add_argument("--condition", choices=["original", "shuffled"], default="original")
    ap.add_argument("--mode", choices=["direct", "cot"], default="direct")
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="pilot: first N items")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between calls (rate limits)")
    a = ap.parse_args()

    df = pd.read_csv(a.itembank)
    if "format" not in df:
        df["format"] = "mcq"
    df["image_question"] = df.get("image_question", "").fillna("") if "image_question" in df else ""
    if a.subjects:
        df = df[df.subject.isin(a.subjects)]
    if a.limit:
        df = df.head(a.limit)

    RUNS.mkdir(parents=True, exist_ok=True)
    tag = re.sub(r"[^\w.-]+", "-", a.model)
    bank = "" if Path(a.itembank).resolve() == ITEMBANK.resolve() else "__" + Path(a.itembank).stem
    out = RUNS / f"{tag}__{a.condition}__{a.mode}{'__vision' if a.vision else ''}{bank}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}

    if a.provider == "openai":
        from openai import OpenAI
        client = OpenAI(base_url=a.base_url, api_key=a.api_key or os.environ.get("OPENAI_API_KEY"))
        call = call_openai
    elif a.provider == "ollama":
        client, call = a.base_url or 'http://localhost:11434/v1', call_ollama
    elif a.provider == "dummy":
        client, rng = None, random.Random(0)
        call = lambda *args: ("Đáp án: 1", 0, 0) if args[2] == SYSTEM_OPEN else (f"Đáp án: {rng.choice('ABCD')}", 0, 0)
    else:
        import anthropic
        client = anthropic.Anthropic(api_key=a.api_key or os.environ.get("ANTHROPIC_API_KEY"))
        call = call_anthropic
    max_tokens = 8 if a.mode == "direct" else 600
    if (df["format"] == "open").any():
        max_tokens = max(max_tokens, 900)

    todo = df[~df.id.isin(done)]
    print(f"{len(todo)} items to run -> {out.name}")
    with out.open("a", encoding="utf-8") as fh:
        for k, row in enumerate(todo.itertuples(index=False), 1):
            is_open = row.format == "open"
            if is_open:
                prompt, gold, mapping = row.question, row.gold_value, None
            else:
                prompt, gold, mapping = build_item(row, a.condition)
            images = item_images(row) if a.vision else []
            user = f"{prompt}\n\n{INSTR_OPEN if is_open else INSTR[a.mode]}"
            system = SYSTEM_OPEN if is_open else SYSTEM
            t0 = time.time()
            for attempt in range(5):
                try:
                    t0 = time.time()
                    text, tin, tout = call(client, a.model, system, user, images,
                                           max_tokens, a.temperature)
                    break
                except Exception as e:  # rate limits / transient errors
                    wait = 2 ** attempt * 5
                    print(f"  ! {row.id}: {type(e).__name__}: {e} -> retry in {wait}s", flush=True)
                    time.sleep(wait)
            else:
                # give up on this item for now WITHOUT writing a record, so that re-running resumes it
                print(f"  ! {row.id}: skipped after 5 failed attempts (re-run to retry)", flush=True)
                continue
            try:
                if is_open:
                    pred = parse_number(text)
                    ok = pred is not None and abs(pred - gold) <= 1e-6 + 0.01 * abs(gold)
                else:
                    pred = parse_letter(text, a.mode)
                    ok = pred == gold
            except Exception as e:  # never let one malformed answer stop the whole run
                print(f"  ! {row.id}: could not score answer ({e}); recorded as incorrect", flush=True)
                pred, ok = None, False
            rec = dict(id=row.id, model=a.model, condition=a.condition, mode=a.mode, format=row.format,
                       vision=a.vision, n_images=len(images), gold=gold, pred=pred,
                       correct=int(ok), mapping=mapping, raw=text,
                       tokens_in=tin, tokens_out=tout, latency=round(time.time() - t0, 2))
            rec.update(system_prompt=system, user_prompt=user, temperature=a.temperature,
                       max_tokens=max_tokens, provider=a.provider,
                       timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if k % 25 == 0:
                print(f"  {k}/{len(todo)}")
            if a.sleep:
                time.sleep(a.sleep)
    print("done")


if __name__ == "__main__":
    sys.exit(main())
