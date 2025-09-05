# Feedback Analyzer — README

## Problem framing & assumptions

**Goal of the project:** Use and manipulate the two raw data files to ger clear and actionable insights for the product team.


**Assumptions:**

## Assumptions:
1) **Data Quality:** Feedback has reliable timestamps and readable text.
2) **Scale:** Data is small enoguh so that a simple for loop can process it.
3) **Labeling simplification:** Each message gets one primary theme. Each message is assigned a theme even if it doesn't fully match.
4) **Actionable granularity:** Weekly trends are the default and nomalies are investigation only, not causal claims.
5) **Analysis Schedule:** This analysis is ran on a regular schedule with a set time interval.



**Tasks perfomed in the python notebook and python scripts:**

1. **Theme mining:** assign a concise (≤5 words) theme + sentiment to each message.
2. **Consolidation:** identify 3–5 recurring themes.
3. **Trends:** weekly volumes + a short summary.
4. **Anomalies:** detect unusual spikes in negative sentiment.
5. **Segmentation:** compare impact by tier, spend cohort, and approval ratio bands.
6. **Recommendations:** 5 prioritized concrete actions with rationale and business impact.
7. **Exdplainability:** Display examples of what type of messages belong to each theme.


**Assumptions:**

* Feedback timestamps are reliable to at least day granularity.
* JSONL is the canonical format for feedback (large-file friendly, streamable).
* Usage features are pre-aggregated (e.g., 30-day spend, approval ratio) or derivable via a straightforward join.
* Light PII handling: outputs show example comments but avoid sensitive identifiers.

---

## Setup instructions

> Works in 3 ways: Notebook, Python scripts (CLI), or website (https://feedbacl-genie.netlify.app/).

### Prereqs

* Python 3.10+ (tested on 3.13)

### Create and activate a virtual environment

```bash
# from repo root
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## How to run
### Easy Option to Analyze and Visualize Results

1. Go to: https://feedbacl-genie.netlify.app/
2. **NOTE:** the graphs displaying 

### Option A — Notebook

1. Open `Notebook/Feedback-Analysis.ipynb`.
2. Update paths to `data/feedback.jsonl` and `data/product_usage.csv` if needed.
3. Add a .env file with your Gemini (Other LLM API)
4. Run all cells.
5. Analyze the outputs.

### Option B — Python scripts (preferred for repeatability)

```bash
* .env file with Gemini API needed. Example
* GOOGLE_API_KEY="actual key"

# 1) Ingest + analyze + generate artifacts
python "Python Scripts/analyze.py" \
  --feedback "data/feedback.jsonl" \
  --usage "data/product_usage.csv" \
  --out "Outputs"

# 2) (Optional) Render an HTML analysis report from artifacts
python "Python Scripts/build_report.py" \
  --inputs "Outputs" \
  --out "Outputs" \
  --title "Customer Feedback Analysis"
```

**Expected outputs in `Outputs/`:**

* **Themes:** `themes_consolidated_feedback.csv`, `theme_consolidation.json`
* **Trends:** `trend_weekly_volume.csv`, `trend_monthly_volume.csv`, `trend_weekly_chart.png`, `trend_weekly_chart_with_anomalies.png`, `trend_summary.json` (and/or `.txt`)
* **Anomalies:** `anomaly_report.json`
* **Segmentation:** `segmentation_by_tier.csv`, `segmentation_by_spend.csv`, `segmentation_by_approval_ratio.csv`
* **Explainability:** `explainability_samples.json`, `explainability_samples.html`
* **Recommendations:** `strategic_recommendations.json`

### Option C — Local static HTML dashboard

The dashboard reads directly from `./Outputs`.

```bash
# simplest way to serve the static site from repo root:
python -m http.server 8080

# then open:
# http://localhost:8080/index.html
```

**Notes**

* Paths are **relative**; keep `index.html`, `data-binder.js`, and the `Outputs/` folder together.
* If hosting (e.g., Netlify), ensure `Outputs/*.csv|*.json|*.png|*.html` are deployed (not ignored by `.gitignore`).
* If you previously ignored all `*.csv`, whitelist `Outputs/` patterns.

---

## Decisions & trade-offs

**JSONL for feedback**

* ✅ Just input the data using pandas.
* ❗At a larger scale, there needs to be a proper data ingestion pipeline so that there can be more scale


**Consolidation → exactly 5 top themes**

* ✅ Forces to have 5 themes where in reality, there might be one off cases.
* ❗ Have slightly less accuarate classification, but ensure all data is labeled.


**Anomaly detection via robust stats (IQR + z-score hybrid)**

* ✅ Simple and explainable, focusing on negative spikes.
* ❗ Anomalies don't have causal inference, they are merely investigation trigggers.

**Segmentation by tier/spend/approval ratio**

* ✅ Aligns with business levers (pricing, underwriting, messaging).
* ❗ Might not be accurated because we are not taking length of customer life into account. This might cause biased inference.

**Simple static dashboard**

* ✅ Zero backend dependency; easy to host/share.
* ❗ There are no itneractive dashboards where users can filter for date ranges, subscription tiers, feedback themes, etc.

---

##  What I’d do next (with more time)

* **Build a full stack app:** build a prototype using Python. Django, JavaScript, and React.
* **Add a Vector Database:** store messages in a vector database based on similarity. This will improve classification.
**Add a knowledge based and previous examples for RAG:** This will also improve message classifation, providing better recommendations
**Use unsupervised machine learning techniques and NLP:** Over time, as more data comes in, classifications methods will become more reliable.
* **Richer anomaly methods:** STL + residual control charts; change-point detection (PELT/BOCPD).
* **Attribution analysis:** link anomalies to new features, updates, policy changes, cloud outages, etc.
* **Ops:** scheduled pipeline to ingest new data and run this analysis on a predefined timeframe.
---

## LLM prompts & approach (actual prompts + reasoning)

Below are the **exact prompts (verbatim)** I used for each step, followed by the **reasoning** behind how each prompt was designed.

---

### Per-message theme & sentiment tagging

# Theme extraction prompt for individual feedback messages
THEME_EXTRACTION_PROMPT = """
* [Role/Persona]
You are a highly precise AI analyst specializing in customer feedback analytics for fintech applications. You process user messages with a focus on accuracy and strict adherence to formatting rules.

* [Context]
You will be given a single customer feedback message from a user of a cash advance app.

* [Task]
Your goal is to perform two actions: first, precisely categorize the user's sentiment, and second, distill the core topic of the message into a concise summary.

[Sentiment Analysis Rules]
1.  The "sentiment" value MUST be one of these three exact strings: "Positive" or "Negative". No other values are permitted.
2.  **Positive**: Use for messages expressing satisfaction, good customer service, and overall experience.
3.  **Negative**: Use for messages expressing frustration, confusion, disappointment, bugs, or problems.

[Theme Generation Rules]
1.  The "theme" value MUST be a concise summary of the message's core topic.
2.  The theme MUST be 5 words or less.
3.  Focus on the root cause or primary subject. For example, for "Why can't I link my Chime account?", the theme should be "Chime account linking issue", not "User has a question".

[Examples]
-   **Input**: "The approval rules are unclear—got rejected again."
    **Output**: {{"theme": "Unclear advance approval rules", "sentiment": "Negative"}}

-   **Input**: "Loving the instant access to cash—thanks!"
    **Output**: {{"theme": "Appreciation for instant cash", "sentiment": "Positive"}}

[Output Format]
Your response MUST be a single, valid JSON object and nothing else. Do not wrap it in markdown or add any introductory text. The JSON object must contain these two keys:
1.  "theme": A string summarizing the core topic.
2.  "sentiment": A string indicating the sentiment.

[User Message]
{message}
"""

---
# Theme consolidation prompt for grouping themes into categories
THEME_CONSOLIDATION_PROMPT = """
* [Role/Persona]
You are a meticulous and expert data analyst AI. Your primary strength is identifying meaningful, high-level patterns from raw text data and grouping them with extreme precision.

* [Context]
You will be given a JSON list of raw, specific themes that were extracted from customer feedback messages in a fintech app. Many of these themes are semantically similar or refer to the same core user issue, just phrased differently.

* [Task]
Your task is to analyze the entire list of themes and consolidate them into exactly 5 distinct, high-level categories. The goal is to create meaningful groups that represent the most significant recurring issues or topics in the feedback.

[Input Data]
Here is the list of themes you must categorize:
{theme_list}

[CRITICAL RULES]
1.  **Exactly 5 Categories**: The final output MUST contain exactly 5 theme categories. No more, no less.
2.  **Complete Coverage**: EVERY single theme from the input list must be placed into one of the 5 categories. Do not omit any themes.
3.  **No Vague Categories**: You are strictly forbidden from creating generic, catch-all categories like "Miscellaneous," "Other," "General Feedback," or "User Issues." Each category name must be specific and actionable.
4.  **Concise Naming**: Each category name must be concise and descriptive, with a maximum of 4 words.

[Good vs. Bad Example]
-   **GOOD Category Name**: "Advance Approval & Rejection" (This is specific and actionable).
-   **BAD Category Name**: "User Problems" (This is too generic and not useful).

[Output Format]
Your response MUST be a single, valid JSON object and nothing else. Do not wrap it in markdown backticks or add any introductory text like "Here is the JSON object:".

The JSON object must have a single key "theme_categories".
The value of this key must be a list of 5 JSON objects.
Each object in the list must contain two keys:
1.  "category_name": A string for the high-level category name you created (e.g., "Technical & Performance Problems").
2.  "included_themes": A list of all the original theme strings that belong to this category.

[Example of Final Output Structure]
{
  "theme_categories": [
    {
      "category_name": "Example Category Name 1",
      "included_themes": ["raw theme a", "raw theme b", "raw theme c"]
    },
    {
      "category_name": "Example Category Name 2",
      "included_themes": ["raw theme d", "raw theme e"]
    }
  ]
}
"""

# Trend summary prompt for analyzing feedback trends
TREND_SUMMARY_PROMPT = """
* [Role/Persona]
You are a senior data analyst at a financial technology company.

* [Context]
You have been given two data tables summarizing customer feedback trends for our app. The first table shows the number of feedback messages received each week for our top 5 recurring themes. The second table shows the same data aggregated by month. The data covers the period from mid-June to late-August 2025.

* [Task]
Your goal is to write a short, professional summary of the key insights from this data. Focus on the most significant trends. Mention which themes are most frequent overall and if any themes show a noticeable increase or decrease over the period.

[Data]
Weekly Feedback Volume:
{weekly_data}

Monthly Feedback Volume:
{monthly_data}

[Output Format]
Provide the summary as a professional paragraph, consisting of 2-3 distinct sentences. Do not add any extra titles or formatting.
"""

---
# Recommendations prompt for strategic analysis
RECOMMENDATION_PROMPT = """
* [Role/Persona]
You are a Senior Product Manager at a high-growth fintech company. You are data-obsessed and your primary goal is to translate complex analytical findings into a prioritized, actionable product strategy.

* [Context]
You have been presented with a "Data Dossier" summarizing a recent analysis of customer feedback, product usage, and sentiment trends. Your task is to use ONLY the information in this dossier to propose a strategic plan.

{data_dossier}

* [Task]
Based ONLY on the data provided in the dossier, generate exactly 5 concrete, prioritized recommendations to address the identified customer issues. For each recommendation, you must provide:
1. A clear rationale that directly references a specific finding from the dossier.
2. A tangible business outcome that the recommendation aims to achieve.

* [CRITICAL RULES]
- Generate EXACTLY 5 recommendations.
- Base every part of your rationale on the specific data points provided in the dossier. Do not invent information.
- Prioritize the recommendations from 1 (most urgent/impactful) to 5.

[Output Format]
Your response MUST be a single, valid JSON object and nothing else.
The JSON object should contain a single key "recommendations", which is a list of 5 JSON objects.
Each object in the list must have the following four keys:
- "priority": An integer (1-5).
- "recommendation_title": A concise, actionable title for the initiative.
- "rationale": A detailed explanation of why this action is needed, directly citing the data from the dossier.
- "business_outcome": The specific business goal this action will impact (e.g., "Reduce churn in the 'High Spender' cohort," "Improve advance approval rates for new users," "Increase user engagement on the 'free' tier.").
"""

---
### Why I wrote each prompt this way

**Theme extraction (per-message):** I constrained the output to two short fields—`theme` (≤5 words) and `sentiment` (Positive/Negative)—so parsing is trivial and readable. The analyst persona reduces creative drift, and the examples anchor tone and polarity. Binary sentiment avoids ambiguity and makes negative-spike detection easier.

**Theme consolidation (exactly 5 categories):** Forcing exactly five, specifically named categories creates a reliable mapping with no data loss. Concise, non-vague names keep categories actionable and allows time-series comparisons.

**Trend summary (2–3 sentences):** I limited the response to a short paragraph to pair with charts without duplicating them. Referencing both weekly and monthly tables nudges the model to reconcile short-term fluctuations with longer trends. Asking for “most frequent” and “increase/decrease” keeps the narrative specific and decision-oriented instead of generic commentary.

**Recommendations (exactly 5, prioritized):** The structure forces each recommendation to tie back to concrete evidence and to state the business outcome, turning insights into actions. Requiring exactly five, ranked by priority, yields a ready-to-use plan for stakeholders.


## Product thinking (how insights translate to action)

**Identity and verification checks:** We see weekly volume rising for identity and verification issues, with negative sentiment above the baseline. The hypotheis is that are failing verification because instructions are unclear or there is data mismatch. A viable solution might be to simplify the steps and add on-screen text, an explicit fallback path (manual review or a clearer document upload option), and show understandable error messages. Success will reuslt in higher approval rate, faster time to verify, and a decline in negative sentiment.

**Bank account linking with Chime by tier:** Complaints about linking a Chime bank account are concentrated in specific subscription tiers. The hypothesis is that there are partner outages, expiring tokens, or multi-factor authentication edge cases affect these cohorts more. A possible solution is to add a “bank-link health” widget, show a proactive status banner when the partner is degraded, and guide the retry path. This will result in higher link-successes on retry, shorter time to relink, and better conversion to the next key steps.

**Clarity on approval criteria:** The hypothesize here is that people do not understand why they were declined. The action is to add a lightweight pre-check that gives eligibility hints. Success will result in fewer support contacts and a higher approval rate after the pre-check,  leading to an improved customer satisfaction.

**Notifications and customer support:** There seems to be noisy and mistimed alerts, which cann be resolved by batching and summarizing the notifications and adding context-aware links inside the app. This will mjght reduce notification opt-outs, increase higher first-contact resolution, and accelerate time to answee.


**User interface and performance:** Persistent complaints reference slow or freezing screens. The hypothesis is that a small number of heavy screens (large assets, slow queries, or cold starts) create most of the pain. The solution is to profile those screens, implement lazy loading and caching, and optimize the slowest queries. This will reduce negative sentiment and improve user experience.

---
## Repo structure (high level)

```
.
├─ data/                         # (local inputs; not committed)
├─ Outputs/                      # generated artifacts consumed by the dashboard
├─ Notebook/
│  └─ Feedback-Analysis.ipynb
├─ Python Scripts/
│  ├─ analyze.py                 # end-to-end pipeline
│  ├─ build_report.py            # compiles artifacts into HTML report
│  └─ prompts.py                 # LLM prompts (unchanged)
├─ index.html                    # static dashboard
├─ data-binder.js                # fetches & binds Outputs/* into the page
├─ requirements.txt
└─ README.md                     # (this file)
```

---


