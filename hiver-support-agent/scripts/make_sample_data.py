"""
Generates data/processed/<brand>_threads.parquet directly from a small,
hand-written sample of realistic AmazonHelp-style customer<->brand pairs,
WITHOUT needing the full ~2.6GB Kaggle CSV.

This exists purely so a grader can clone the repo and see the pipeline run
in minutes without a Kaggle account. It is NOT a substitute for real data:
the golden eval set and reported headline numbers in report/REPORT.md were
built from the actual Kaggle dataset, not this sample (see README > 'Two
ways to run this').

Run: python scripts/make_sample_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402

# 30 realistic, hand-written customer<->brand pairs covering all 9 intents,
# 3-4 examples each, so retrieval has something to find for every intent.
SAMPLE_PAIRS = [
    # delivery_delay
    ("My order #112-4587211 was supposed to arrive yesterday and tracking still says 'label created'. Where is it?",
     "So sorry for the delay! I've checked and your package is moving again as of this morning -- new estimated delivery is tomorrow. I'll keep an eye on it for you. [link]"),
    ("This is the 3rd day my package has said 'out for delivery' with no update. Getting frustrated.",
     "I completely understand the frustration. I've escalated this with the carrier directly and requested an update within 24 hours -- I'll follow up here as soon as I hear back."),
    ("Any update on when my order will actually ship? It's been 5 days.",
     "Thanks for your patience! Your order shipped this morning and should arrive within 2-3 business days. Tracking: [link]"),
    ("Package says delivered but I never got it. Where is my stuff??",
     "That's concerning, I'm sorry! Can you DM us your order number? We'll open an investigation with the carrier and get this sorted."),
    # order_cancellation
    ("Can you cancel order #998-1234567? I ordered the wrong size.",
     "Done! I've cancelled that order and you'll see a refund in 3-5 business days. Let us know if you'd like help placing the correct size."),
    ("I tried to cancel my order online but it won't let me, the button is greyed out.",
     "That usually means it's already started processing. DM us the order number and we'll see what we can do on our end."),
    ("Please cancel my subscription, I don't want it renewing next month.",
     "I've turned off auto-renew for you -- your current period stays active until it ends, but you won't be charged again."),
    # refund_request
    ("I was charged twice for the same order, I need one of these refunded.",
     "Apologies for that! I can see the duplicate charge -- I've submitted a refund for the extra one, it should post within 5-7 business days."),
    ("Requesting a refund, the item I received doesn't match the listing at all.",
     "So sorry to hear that. Please go to Your Orders > Return/Refund Items and select 'item not as described' -- refund will process once it's on the way back to us."),
    ("Where is my refund? It's been over a week since I returned the item.",
     "Let me check that for you -- can you DM your order number? Refunds usually post within 3-5 business days of us receiving the return, but I'll look into what's taking longer."),
    # product_defect
    ("The blender I ordered arrived with a cracked jug, completely unusable.",
     "I'm really sorry about that! I can send a free replacement right away, or refund in full -- whichever you'd prefer."),
    ("Box arrived, item inside is missing a part (the charging cable).",
     "Thanks for letting us know -- I've flagged this with the seller and I'm sending out the missing cable at no cost, should arrive in 3-5 days."),
    ("This is the second defective unit I've received. Really disappointed.",
     "I understand, that's not the experience we want for you. I'm escalating this to a specialist who can look at a different resolution than a straight replacement."),
    # account_access
    ("I can't log into my account, it keeps saying incorrect password even after reset.",
     "Sorry for the trouble! Please try clearing your browser cache and requesting a new reset link -- if it still fails, DM us and we'll look at the account directly."),
    ("My account got locked after too many login attempts, I need it unlocked ASAP.",
     "I hear you -- for account security we can't unlock this over public reply. Please DM us so we can verify your identity and get you back in."),
    ("Someone else seems to have accessed my account, I see orders I didn't place!",
     "That's serious, please DM us right away -- we'll secure the account and look into those orders immediately."),
    # return_exchange
    ("Item doesn't fit, how do I exchange it for a larger size?",
     "No problem! Go to Your Orders > Return or Replace Items, choose 'wrong size' and select the size you need -- we'll ship the replacement once the return is scanned."),
    ("Want to return this, just changed my mind. Is that allowed?",
     "Of course! As long as it's within the return window shown on your order page, go ahead and start a standard return there."),
    ("I received the wrong color, need to swap it for the one I actually ordered.",
     "Sorry about that mix-up! Start a replacement request under Your Orders and select 'wrong item sent' -- correct color will ship right away."),
    # billing_inquiry
    ("What is this $14.99 charge on my card, I don't recognize it?",
     "Happy to help figure that out -- can you DM the last 4 digits of the card and the date of the charge? We'll match it to an order or subscription."),
    ("Why did my subscription price go up this month without any notice?",
     "I'm sorry for the confusion -- pricing changes are usually emailed 30 days ahead. DM us your account email and we'll check what changed and why."),
    ("Can I get an itemized invoice for my last order for expense purposes?",
     "Sure thing! You can download a full invoice from Your Orders > Invoice, or DM your order number and I'll email one over."),
    # general_inquiry
    ("Do you guys ship internationally to Canada?",
     "Yes, we ship to Canada! Shipping costs and delivery estimates will show at checkout based on your address."),
    ("What are your customer service hours?",
     "We're here 24/7 on Twitter! Phone support runs 6am-10pm local time, details on our help page: [link]"),
    ("Is the warranty on electronics 1 year or 2?",
     "Standard manufacturer warranty is 1 year unless stated otherwise on the product page -- some brands offer extended options at checkout."),
    # complaint_escalation
    ("This is unacceptable. Third time contacting support this month and NOTHING has been resolved. I want a manager.",
     "I sincerely apologize for the repeated trouble -- this should not have happened three times. I'm escalating you directly to a senior specialist who will reach out within 24 hours."),
    ("I am extremely disappointed and considering never ordering again after how this was handled.",
     "I completely understand your frustration and I'm sorry we let you down. I'd like a specialist to personally make this right -- please DM us and we'll prioritize your case."),
    ("Nobody has responded to my last 2 messages. This is the worst support experience I've had.",
     "I'm so sorry for the silence, that's not okay. I'm flagging this for immediate follow-up from a supervisor -- you'll hear from us within a few hours."),
]


def main():
    rows = []
    for i, (cust, brand) in enumerate(SAMPLE_PAIRS):
        rows.append({
            "thread_id": 100000 + i,
            "prior_context": "",
            "customer_message": cust,
            "brand_reply": brand,
            "created_at": "",
        })
    df = pd.DataFrame(rows)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.BRAND_THREADS_PARQUET, index=False)
    print(f"Wrote {len(df)} sample thread pairs -> {config.BRAND_THREADS_PARQUET}")
    print("NOTE: this is a small hand-written demo set, not the real Kaggle data.")
    print("For real headline results, run scripts/download_data.sh + python -m src.data_prep instead.")


if __name__ == "__main__":
    main()
