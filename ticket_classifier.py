from transformers import pipeline
import json
import re

# ── LOAD MODEL ───────────────────────────────────────────────
print("⏳ Loading TinyLlama...")
pipe = pipeline("text-generation",
                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                torch_dtype="auto", device_map="auto")
print("✅ Ready!\n")


# ── FEW-SHOT + CoT PROMPT ────────────────────────────────────
def build_prompt(ticket):
    return f"""You are a customer support classifier.
Classify the ticket into ONLY one of these: Billing, Technical Issue, Account Access, Refund, Other.
Think step by step, then return ONLY a JSON object.

JSON format:
{{"thinking": "your reasoning here", "category": "one of the 5 categories", "reason": "one sentence"}}

--- EXAMPLE 1 ---
Ticket: "I was charged twice for my subscription this month."
{{"thinking": "The user is talking about a payment/charge problem. This is a money issue.", "category": "Billing", "reason": "User was charged twice which is a billing error."}}

--- EXAMPLE 2 ---
Ticket: "The app keeps crashing every time I open it."
{{"thinking": "The app is crashing. This is a software/technical problem.", "category": "Technical Issue", "reason": "App crashing is a technical malfunction."}}

--- EXAMPLE 3 ---
Ticket: "I forgot my password and cannot login to my account."
{{"thinking": "User cannot access their account due to forgotten password.", "category": "Account Access", "reason": "Login failure due to forgotten password is an account access issue."}}

--- NOW CLASSIFY ---
Ticket: "{ticket}"
"""


# ── SAFE JSON PARSER ─────────────────────────────────────────
# LLMs sometimes add extra text around JSON — this handles it safely
def safe_parse(text):
    try:
        return json.loads(text.strip())
    except:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return {"thinking": "parse failed", "category": "Other", "reason": "Could not parse response"}


# ── CLASSIFIER FUNCTION ──────────────────────────────────────
def classify(ticket):
    messages = [
        {"role": "system", "content": "You are a support ticket classifier. Always reply with a valid JSON object only."},
        {"role": "user",   "content": build_prompt(ticket)}
    ]
    output = pipe(messages, max_new_tokens=200, temperature=0.2,
                  top_p=0.9, top_k=40, do_sample=True)
    raw = output[0]["generated_text"][-1]["content"]
    return safe_parse(raw)


# ── TEST WITH 5 TICKETS ──────────────────────────────────────
tickets = [
    "I want a refund for my last order, it never arrived.",
    "My internet keeps disconnecting every 10 minutes.",
    "How do I reset my password? I am locked out.",
    "You charged me $99 but my plan is only $49.",
    "I just wanted to say your service is amazing!"
]

print("=" * 55)
print("   CUSTOMER SUPPORT TICKET CLASSIFIER — RESULTS")
print("=" * 55)

for i, ticket in enumerate(tickets, 1):
    print(f"\n Ticket {i}: {ticket}")
    result = classify(ticket)
    print(f"Thinking  : {result.get('thinking', 'N/A')}")
    print(f"Category  : {result.get('category', 'N/A')}")
    print(f" eason    : {result.get('reason', 'N/A')}")
    print(f" Full JSON : {json.dumps(result, indent=2)}")
    print("-" * 55)

print("\n✅ Done! Screenshot this for submission 📸")