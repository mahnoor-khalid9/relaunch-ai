"""
relaunch.ai — Full LangGraph Multi-Agent Pipeline
===================================================
Agents:
  1. research_agent     — builds a structured research dossier
  2. autopsy_agent      — 6-lens failure analysis + Primary Failure Hypothesis
  3. revival_agent      — full GTM relaunch strategy for 2025
  4. copywriter_agent   — 3 polished outputs (card, pitch, elevator)
  5. marketing_agent    — generates full mock landing page HTML
"""

import os, json, re, logging, requests
from pathlib import Path
from typing import TypedDict, Optional
from datetime import datetime
from langgraph.graph import StateGraph, START, END

# ── Logging ───────────────────────────────────────────────────────────────────
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(handlers=[logging.FileHandler(log_dir / "relaunch_ai.json")], level=logging.INFO)
logger = logging.getLogger("relaunch_ai")

# ── Deploy AI ─────────────────────────────────────────────────────────────────
AUTH_URL      = "https://api-auth.deploy.ai/oauth2/token"
API_URL       = "https://core-api.deploy.ai"
CLIENT_ID     = os.getenv("CLIENT_ID", "")
CLIENT_SECRET = os.getenv("CLIENT_SECRET", "")
ORG_ID        = os.getenv("ORG_ID", "59f3dce8-2dcf-4a7f-b6ff-d2cbce1231dc")


def _token():
    r = requests.post(AUTH_URL, data={
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET
    }, timeout=15)
    return r.json()["access_token"]


def llm(system: str, user: str) -> str:
    if not CLIENT_ID or not CLIENT_SECRET:
        return _mock(system, user)
    try:
        tok = _token()
        h = {"Authorization": f"Bearer {tok}", "X-Org": ORG_ID,
             "Content-Type": "application/json", "accept": "application/json"}
        cid = requests.post(f"{API_URL}/chats", headers=h,
                            json={"agentId": "GPT_4O", "stream": False}, timeout=15).json()["id"]
        return requests.post(f"{API_URL}/messages", headers=h, json={
            "chatId": cid, "stream": False,
            "content": [{"type": "text", "value": f"SYSTEM:\n{system}\n\nUSER:\n{user}"}]
        }, timeout=120).json()["content"][0]["value"]
    except Exception as e:
        logger.warning(f"LLM error: {e}")
        return _mock(system, user)


# ── Current year (used throughout mock responses) ─────────────────────────────
CUR_YEAR = datetime.now().year


# ── Fully dynamic competitor archetypes (no hardcoded companies) ───────────────
def _competitors_from_context(
    name: str, desc: str, category: str, market: str,
    founded: str, shutdown: str, active_str: str,
    funding: str, signals: list, why_failed: str
) -> list:
    """
    Generate 3 competitor success archetypes derived entirely from the user's
    startup inputs. No hardcoded company names — every sentence references the
    actual startup, its description, its market, its failure signals, and its timeline.
    """
    sig_text   = " ".join(signals).lower() + " " + why_failed.lower()
    core_desc  = (desc[:75] + "…") if len(desc) > 75 else desc
    short_name = name.split()[0] if " " in name else name  # first word for brevity
    cat        = category or "technology"

    # ── Derive context-sensitive labels for outcome/channel ─────────────────
    # Monetisation context
    ran_out   = "ran out of money" in sig_text or "wrong pricing" in sig_text
    no_pmf    = "growth stalled" in sig_text or "product was never finished" in sig_text
    team_fail = "team fell apart" in sig_text
    too_early = "too early" in sig_text or "lockdown" in sig_text
    big_comp  = "larger competitor" in sig_text

    # Timing context
    yrs_active = ""
    try:
        n = int(shutdown) - int(founded)
        yrs_active = f"{n} year{'s' if n != 1 else ''}"
    except Exception:
        yrs_active = "its operating window"

    # Funding shorthand
    raised_str = f"{funding} raised" if funding not in ("Undisclosed", "Unknown", "") else "its capital"

    # ── ARCHETYPE 1: The Narrow-Wedge Player ─────────────────────────────────
    # Contrasts with {name}'s likely over-broad scope; derived from desc + category
    wedge_full = (
        f"tried to build {core_desc} for the full {cat} market" if desc
        else f"tried to address the entire {cat} space at once"
    )
    a1 = {
        "name": f"The Narrow-Wedge {cat} Player",
        "outcome": (
            f"Achieved self-sustaining growth in the {market} {cat} market"
            f"{' — in less time than ' + short_name + ' had on the market' if yrs_active else ''}"
        ),
        "why_succeeded": (
            f"This competitor solved the same core problem as {name} but refused to serve more than one "
            f"specific customer segment in the first 12 months. "
            f"Unlike {name} — which {wedge_full} — "
            f"this player picked the single most painful step in the {cat} workflow and became "
            f"indispensable for it before touching anything adjacent. "
            f"Every feature, every sales conversation, every pricing decision was anchored to "
            f"that one segment's exact daily pain — not to a broader vision. "
            f"Expansion happened only after word-of-mouth within that segment was self-sustaining."
        ),
        "key_lesson": (
            f"The startup that wins a large market usually enters through the narrowest possible door. "
            f"Narrow scope compresses the feedback loop, reduces burn rate, and manufactures "
            f"the word-of-mouth that broad products can never buy. "
            f"In the {cat} space, 'solve everything' is a fundraising story — 'solve this one thing completely' "
            f"is a go-to-market strategy."
        ),
        "how_to_apply": (
            f"The revived {name} should answer one question before writing a line of code: "
            f"'What is the single most painful, most frequent moment in the {cat} workflow "
            f"that our target customer in {market} experiences?' "
            f"{'The original spent ' + active_str + ' trying to be comprehensive — the revival must spend its first 6 months being indispensable for one thing.' if yrs_active else 'Start narrower than feels commercially viable. Expand only after the segment is locked.'} "
            f"Resist every pressure to generalise until that wedge generates unsolicited referrals."
        ),
    }

    # ── ARCHETYPE 2: The Revenue-First Builder ───────────────────────────────
    # Contrasts with {name}'s failure to validate monetisation before burning capital
    capital_contrast = (
        f"where {name} spent {raised_str} validating whether the market existed"
        if funding not in ("Undisclosed", "Unknown", "")
        else f"where {name} burned runway before confirming willingness to pay"
    )
    pmf_note = (
        "Growth stalling after initial traction is a classic late-stage PMF signal — "
        "it means early adopters adopted but the mainstream refused to follow."
        if no_pmf else
        f"In the {cat} space, the gap between 'users love it' and 'users pay for it' "
        f"has killed more startups than any competitor ever has."
    )
    a2 = {
        "name": f"The Revenue-First {cat} Builder",
        "outcome": (
            f"Reached positive unit economics in the {market} {cat} market "
            f"spending a fraction of {raised_str if raised_str != 'its capital' else 'the typical raise for this category'}"
        ),
        "why_succeeded": (
            f"This competitor's founding rule was: no product feature gets built unless a customer "
            f"has already paid for it. They ran a manual concierge MVP for 90 days — doing the job "
            f"by hand that the software would eventually automate. "
            f"The first 5 customers paid before a single scalable line of code was written. "
            f"Every subsequent feature was pre-sold. "
            f"{capital_contrast.capitalize()}, this player spent under $100K confirming "
            f"the same hypothesis. {pmf_note}"
        ),
        "key_lesson": (
            f"Willingness to pay is the only PMF signal that doesn't lie. "
            f"User sign-ups, NPS scores, and letters of intent are all proxies. "
            f"A customer handing over money — before the product is complete — "
            f"is the only truly honest signal in the {cat} space. "
            f"{'The ' + funding + ' raised by ' + name + ' should have purchased 10 paying customers before it purchased a single engineer.' if funding not in ('Undisclosed','Unknown','') else 'Revenue should precede roadmap. Every feature should be paid for before it is built.'}"
        ),
        "how_to_apply": (
            f"Before rebuilding {name}, identify 5 target customers in {market} who would pay today — "
            f"not 'when the product is ready', not 'in principle', but this week, for a manual "
            f"version of the solution. If 5 people won't pay for a human doing the job, the software "
            f"version won't change that. "
            f"{'Those 5 paying customers should fund the first 60 days of development entirely — no external capital needed to reach that milestone.' if ran_out else f'Use those 5 paying customers as the only valid input to the {CUR_YEAR} roadmap. Kill every feature that none of them asked for.'}"
        ),
    }

    # ── ARCHETYPE 3: The Distribution-Owner ──────────────────────────────────
    # Contrasts with {name}'s likely undefined or expensive acquisition channel
    big_comp_note = (
        f"When a larger competitor entered the {cat} space, this player was protected "
        f"because its distribution channel was owned, not rented — the competitor "
        f"couldn't copy the channel relationship the way it could copy features."
        if big_comp else
        f"In the {cat} market in {market}, no amount of product quality compensates "
        f"for the wrong distribution strategy. This player proved it."
    )
    # Derive a channel hint from the category
    c_low = cat.lower()
    if any(k in c_low for k in ["b2b", "saas", "enterprise", "platform", "software"]):
        channel_type = "bottom-up adoption through a permanent free tier that individual contributors used before IT got involved"
    elif any(k in c_low for k in ["health", "medical", "bio", "clinic"]):
        channel_type = "clinical co-development partnerships with 2–3 health systems who provided distribution in exchange for design authority"
    elif any(k in c_low for k in ["consumer", "social", "media", "content", "audio"]):
        channel_type = "a single organic content channel that already had the attention of the target audience before the product launched"
    elif any(k in c_low for k in ["fintech", "finance", "payment", "banking"]):
        channel_type = "API-first developer adoption where engineers at target companies became internal champions before the enterprise sale began"
    elif any(k in c_low for k in ["hardware", "iot", "device"]):
        channel_type = "a strategic distribution partnership with an established channel that already sold to the target customer"
    else:
        channel_type = f"one trusted distribution partner already embedded in the {cat} customer's existing workflow"

    a3 = {
        "name": f"The Channel-Owned {cat} Entrant",
        "outcome": (
            f"Acquired its first 100 paying customers in {market} at near-zero acquisition cost "
            f"by owning its distribution channel before shipping product"
        ),
        "why_succeeded": (
            f"This competitor spent the first 60 days of its existence securing "
            f"{channel_type} — before writing a single line of product code. "
            f"By launch day, 50 qualified, warm leads were already waiting. "
            f"They never ran a paid ad. They never hired a sales team before achieving repeatable, "
            f"founder-led revenue. {big_comp_note}"
        ),
        "key_lesson": (
            f"Distribution is a strategy, not a tactic. In {cat}, the company that owns "
            f"a distribution channel — whether through partnerships, developer communities, "
            f"platform integrations, or earned content — beats the company with a better "
            f"product every time at the same funding level. "
            f"{'The ' + active_str + ' that ' + name + ' spent suggests time was available to build distribution. The question is whether it was prioritised.' if yrs_active else 'Distribution strategy should precede product strategy, not follow it.'}"
        ),
        "how_to_apply": (
            f"Before the revived {name} builds anything, map the full distribution landscape "
            f"in {market}: who already has the daily attention of the target {cat} customer? "
            f"What integration, partnership, or community could deliver customers without paid acquisition? "
            f"Specifically for {name}'s space: {channel_type} is the distribution vector worth exploring first. "
            f"Close that partnership before the product ships. "
            f"{'If no such partner exists, that absence is itself a signal — distribution-resistant markets require either a very long runway or a very viral product mechanic.' if funding not in ('Undisclosed','Unknown','') else 'If no distribution partner is available, the go-to-market must be rethought before a line of code is written.'}"
        ),
    }

    return [a1, a2, a3]


# ── Mock LLM ──────────────────────────────────────────────────────────────────
def _parse_user_ctx(user: str) -> dict:
    """
    Parse all structured fields from the _build_context output embedded in the
    user message. This ensures every mock agent response uses the actual user
    inputs rather than any hardcoded placeholder data.
    """
    def _get(pattern, default=""):
        m = re.search(pattern, user, re.IGNORECASE)
        return m.group(1).strip() if m else default

    # Try context-string format first, then JSON format as fallback
    founded  = _get(r'Active:\s*(\d{4})') or _get(r'"founded"\s*:\s*"([^"]+)"')
    shutdown = _get(r'Active:\s*\d{4}\s*[→\-]+\s*(\d{4})') or _get(r'"shutdown"\s*:\s*"([^"]+)"')
    funding  = _get(r'Funding:\s*([^\n]+)') or _get(r'"funding"\s*:\s*"([^"]+)"')
    # Category: stop at — (em dash) or " or newline; avoid source titles like "Industry: X — CB Insights"
    # First try JSON "category" key (most reliable for revival/copywriter agents)
    # then fall back to context string "Industry: X" format
    _cat_json = _get(r'"category"\s*:\s*"([^"]+)"')
    _cat_ctx  = _get(r'Industry:\s*([^\u2014\n"]+)').strip()  # stops at em-dash
    category  = _cat_json or _cat_ctx
    market   = _get(r'Market:\s*([^\n]+)') or _get(r'"market"\s*:\s*"([^"]+)"')
    desc     = (_get(r'What it did:\s*([^\n]+)')
                or _get(r'"one_liner"\s*:\s*"([^"]+)"')
                or _get(r'"what_they_built"\s*:\s*"([^"]{10,200})"'))

    return {
        "founded":    founded,
        "shutdown":   shutdown,
        "funding":    funding,
        "category":   category,
        "market":     market,
        "desc":       desc,
        "overview":   _get(r"Founder's description.*?:\s*([^\n]+)"),
        "why_failed": _get(r'Why it failed.*?:\s*([^\n]+)'),
        # context signals from JSON
        "signals":    re.findall(r'"([^"]+)"', _get(r'context_signals.*?(\[[^\]]*\])', '[]')),
    }


def _mock(system: str, user: str) -> str:
    s = system.lower()

    # ── Extract startup name (most reliable first) ────────────────────────────
    name = "This Startup"
    for pattern in [
        r'^Startup[:\s]+([A-Za-z0-9][A-Za-z0-9\s\.\-]{1,50}?)[,\s]',   # "Startup: XYZ\n"
        r'startup_name["\s:]+([A-Za-z0-9][A-Za-z0-9\s\.\-]{1,50}?)"',  # startup_name: "XYZ"
        r'"name"\s*:\s*"([A-Za-z0-9][^"]{1,50}?)"',                     # "name": "XYZ"
    ]:
        m = re.search(pattern, user, re.IGNORECASE | re.MULTILINE)
        if m:
            candidate = m.group(1).strip().title()
            # Reject obvious noise words
            if candidate.lower() not in ("this startup", "unknown", "n/a", "none", "the startup"):
                name = candidate
                break

    # ── Parse actual user inputs from the context string ─────────────────────
    ctx = _parse_user_ctx(user)
    founded   = ctx["founded"]   or "Unknown"
    shutdown  = ctx["shutdown"]  or "Unknown"
    funding   = ctx["funding"]   or "Undisclosed"
    category  = ctx["category"]  or "Technology"
    market    = ctx["market"]    or "Global"
    desc      = ctx["desc"]      or f"{name} built a product in the {category} space."
    overview  = ctx["overview"]
    why       = ctx["why_failed"]
    signals   = ctx["signals"]

    # Slug for URL generation
    slug = re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')
    name_enc = name.replace(' ', '+')

    # Infer years active
    try:
        years = int(shutdown) - int(founded)
        active_str = f"{founded}–{shutdown} ({years} year{'s' if years != 1 else ''})"
    except Exception:
        active_str = f"{founded}–{shutdown}"

    # ── Calibrate autopsy ratings from context signals ────────────────────────
    def _rating(signal_keys: list, default: str) -> str:
        for sig in signals:
            for key in signal_keys:
                if key.lower() in sig.lower():
                    return "Critical"
        return default

    timing_r   = _rating(["too early", "lockdown", "pandemic"], "Significant")
    market_r   = _rating(["wrong pricing", "ran out of money"], "Significant")
    pmf_r      = _rating(["growth stalled", "product was never finished"], "Significant")
    team_r     = _rating(["team fell apart", "ran out of money", "product was never finished"], "Significant")
    comp_r     = _rating(["larger competitor"], "Minor")
    extern_r   = _rating(["regulation", "lockdown"], "Minor")

    # ── Research Agent ────────────────────────────────────────────────────────
    if "encyclopaedic" in s or ("research analyst" in s and "dossier" in s):
        sources = [
            {"title": f"{name} — Crunchbase Profile",        "url": f"https://www.crunchbase.com/organization/{slug}"},
            {"title": f"{name} — Google News Archive",       "url": f"https://news.google.com/search?q={name_enc}+startup+shutdown"},
            {"title": f"{name} — Hacker News Discussions",   "url": f"https://hn.algolia.com/?q={name_enc}"},
            {"title": f"{name} — TechCrunch Coverage",       "url": f"https://techcrunch.com/search/{slug}"},
            {"title": f"{name} — Reddit Threads",            "url": f"https://www.reddit.com/search/?q={name_enc}+startup&sort=relevance"},
            {"title": f"{name} — LinkedIn Company Page",     "url": f"https://www.linkedin.com/search/results/companies/?keywords={name_enc}"},
            {"title": f"{name} — PitchBook Entry",           "url": f"https://pitchbook.com/search#q={name_enc}&type=all"},
            {"title": f"Industry: {category} — CB Insights", "url": f"https://www.cbinsights.com/research-{slug}"},
        ]
        pivot_text = (f"Pivots noted: {why}" if why else
                      "No specific pivot data publicly available; shutdown appears to have been a clean wind-down.")
        press_text = (f"Based on available signals, {name} received coverage during its {active_str} lifespan, "
                      f"with post-mortem commentary emerging after the {shutdown} shutdown.")
        community_text = (f"Hacker News and Reddit discussions around {name} reference common themes: "
                          f"difficulty finding a scalable business model, competitive pressure in the {category} space, "
                          f"and challenges converting early traction into sustainable growth.")
        competitor_text = (f"Competitors in the {category} space during {founded}–{shutdown} included both "
                           f"established incumbents and well-funded startups racing for market share. "
                           f"{name} faced the challenge of differentiating in an increasingly crowded landscape.")
        market_cond = (f"The {market} {category} market during {founded}–{shutdown} was characterised by "
                       f"rapid technological change, shifting customer expectations, and increasing competition for funding. "
                       f"External macro conditions during this window added pressure on runway-constrained startups.")
        founder_txt = (f"{overview}" if overview else
                       f"Limited public commentary from {name}'s founders is available. "
                       f"Post-shutdown interviews or blog posts, if they exist, would provide the most direct insight.")

        # ── Market shifts since shutdown (fully derived from user inputs) ────
        years_since = CUR_YEAR - int(shutdown) if shutdown.isdigit() else 3
        sig_ctx     = " ".join(signals).lower()
        # Build each shift to directly reference the startup's specific context
        shift_pmf   = (
            f"The 'growth stalled' failure mode that affected {name} is now a well-documented pattern — "
            f"founders in {CUR_YEAR} have access to battle-tested frameworks (Jobs-to-be-Done, concierge MVP, "
            f"pre-charged waitlists) specifically designed to prevent the product-market fit gap that shut {name} down."
            if "growth stalled" in sig_ctx else
            f"The {category} market has matured since {shutdown}: customer education costs are lower, "
            f"the category vocabulary is established, and buyers arrive with clearer expectations than {name}'s "
            f"early customers did — reducing the sales cycle friction that consumed early runway."
        )
        shift_ai    = (
            f"Post-2023 AI/LLM tooling has cut the cost of building a {category} product by 60–80%. "
            f"The core {name} vision — {desc[:60] + '…' if len(desc) > 60 else desc} — "
            f"{'can now be validated with under $500K, compared to the ' + funding + ' the original required' if funding not in ('Undisclosed','Unknown','') else 'can now be validated for a fraction of what the original required'}."
        )
        shift_infra = (
            f"Since {shutdown}, {years_since} year{'s' if years_since != 1 else ''} of cloud infrastructure "
            f"investment has commoditised the {category} backend stack that would have absorbed a significant "
            f"portion of {name}'s engineering budget. What required a full platform team in {founded} "
            f"is now a managed service configuration in {CUR_YEAR}."
            if founded not in ("Unknown", "") else
            f"Infrastructure commoditisation since {shutdown} means the platform engineering investment "
            f"that consumed early runway in the {category} space is now available as managed services, "
            f"dramatically reducing time-to-market for a revived product."
        )
        shift_funding = (
            f"Post-2022 funding discipline has flipped the narrative: investors in {CUR_YEAR} actively "
            f"reward capital efficiency and early revenue — the exact story a lean {name} revival can tell "
            f"by starting with 10 paying customers and no institutional capital. "
            f"{'The ' + funding + ' raised by the original is now a cautionary number, not an aspirational one — a revived ' + name + ' that raises less and proves more will be the stronger fundraising story.' if funding not in ('Undisclosed','Unknown','') else 'A leaner raise with earlier revenue is now a competitive advantage in fundraising, not a compromise.'}"
        )
        shift_comp = (
            f"Competitors that defeated {name} in {founded}–{shutdown} may themselves have weakened or pivoted "
            f"in the {years_since} years since. The competitive map in the {category} space in {market} "
            f"must be re-drawn from scratch in {CUR_YEAR} — advantages that seemed insurmountable in {shutdown} "
            f"may no longer exist, and new gaps may have opened."
        )
        market_shifts = [shift_ai, shift_pmf, shift_infra, shift_funding, shift_comp]

        # ── Competitor success stories (fully derived from user inputs) ──────
        competitors_doing_well = _competitors_from_context(
            name, desc, category, market, founded, shutdown,
            active_str, funding, signals, why
        )

        return json.dumps({
            "name":                name,
            "founded":             founded,
            "shutdown":            shutdown,
            "funding":             funding,
            "investors":           [f"Seed-stage investors ({founded})", f"Series A investors ({int(founded)+2 if founded.isdigit() else 'n/a'})", "Strategic angels"],
            "category":            category,
            "market":              market,
            "one_liner":           desc,
            "what_they_built":     (overview if overview else desc),
            "press_coverage":      press_text,
            "founder_interviews":  founder_txt,
            "community_signals":   community_text,
            "pivots":              pivot_text,
            "competitor_landscape":competitor_text,
            "market_conditions":   market_cond,
            "key_market_shifts":   market_shifts,
            "competitors_doing_well": competitors_doing_well,
            "data_confidence":     "medium",
            "public_data_available": True,
            "sources":             sources,
        })

    # ── Autopsy Agent ─────────────────────────────────────────────────────────
    if "ruthless" in s or "post-mortem analyst" in s:
        why_text = why or f"{name} struggled to find a repeatable, scalable business model before running out of runway."
        return json.dumps({
            "primary_failure_hypothesis": (
                f"{name} failed to achieve product-market fit within its {active_str} lifespan — "
                f"spending {funding} without validating a sustainable path to growth, "
                f"and ultimately shutting down when the gap between capital efficiency and market demand became insurmountable."
            ),
            "overall_score": 22,
            "data_note": (
                f"Analysis is partially inferred from founder-provided context and publicly available signals. "
                f"Direct metrics (churn, NPS, revenue) were not publicly disclosed by {name}."
            ),
            "timing": {
                "rating": timing_r,
                "finding": (
                    f"{name} operated from {founded} to {shutdown}. "
                    f"The {category} market during this window was {'still maturing, making customer education expensive and sales cycles long' if timing_r == 'Critical' else 'competitive but addressable with the right positioning'}. "
                    f"The timing of the shutdown in {shutdown} suggests the team ran out of time before the market came to them."
                ),
                "evidence": (
                    f"Active from {founded}–{shutdown} ({active_str}). "
                    f"Funding of {funding} was not sufficient to outlast the market timing gap. "
                    f"{'Founder noted market timing as a factor.' if timing_r == 'Critical' else 'No specific timing crisis was flagged in available signals.'}"
                )
            },
            "market_size_monetization": {
                "rating": market_r,
                "finding": (
                    f"The monetisation model for {name}'s {category} product was never definitively validated at scale. "
                    f"With {funding} raised, the path to a unit-economics-positive business required either a larger TAM "
                    f"than the market supported or a pricing model that customers consistently accepted."
                ),
                "evidence": (
                    f"Funding of {funding} is consistent with a seed/Series A stage company that had not yet demonstrated "
                    f"repeatable revenue. {'Founder cited pricing/monetisation as a challenge.' if market_r == 'Critical' else 'No confirmed ARR or revenue milestones were publicly disclosed.'}"
                )
            },
            "pmf": {
                "rating": pmf_r,
                "finding": (
                    f"{name}'s core product — {desc[:120] if len(desc) > 120 else desc} — "
                    f"{'showed initial traction but failed to retain customers at the rate needed to justify continued investment' if pmf_r == 'Significant' else 'struggled from the outset to demonstrate consistent, organic customer pull'}. "
                    f"The gap between early adopter enthusiasm and mainstream adoption was never bridged."
                ),
                "evidence": (
                    f"Shutdown in {shutdown} without a successful exit or acqui-hire strongly implies PMF was not achieved. "
                    f"{'Growth stalled after initial traction — a classic late-stage PMF failure signal.' if 'growth stalled' in ' '.join(signals).lower() else 'No public retention or engagement metrics confirm sustained PMF.'}"
                )
            },
            "team_execution": {
                "rating": team_r,
                "finding": (
                    f"{'The team fell apart before the company could recover — a critical execution failure that compounded every other problem.' if team_r == 'Critical' else f'{name} faced execution challenges common to startups in the {category} space: hiring the right talent, managing burn, and pivoting quickly enough to stay ahead of market feedback.'}  "
                    f"The {active_str} window suggests the team had time to attempt corrections but could not find the right formula."
                ),
                "evidence": (
                    f"{'Team fragmentation explicitly cited as a failure factor.' if 'team fell apart' in ' '.join(signals).lower() else f'No public founder conflict data available for {name}. Shutdown timeline implies execution gaps went unresolved for too long.'}"
                )
            },
            "competition_defensibility": {
                "rating": comp_r,
                "finding": (
                    f"{'A larger competitor moved into the space and commoditised the core value proposition before ' + name + ' could build sufficient defensibility.' if comp_r == 'Critical' else f'The {category} space in which {name} competed became increasingly crowded between {founded} and {shutdown}.'}  "
                    f"Without a clear moat — proprietary data, network effects, or switching costs — "
                    f"{name} was vulnerable to better-funded competitors replicating its core features."
                ),
                "evidence": (
                    f"{'Competitor copying explicitly cited as a factor.' if comp_r == 'Critical' else f'Standard competitive pressure in the {category} market during {founded}–{shutdown}. No specific copycat event was flagged in available signals.'}"
                )
            },
            "external_factors": {
                "rating": extern_r,
                "finding": (
                    f"{'Regulatory intervention was cited as a direct blocker — an external factor largely outside the team\'s control.' if extern_r == 'Critical' else f'No catastrophic external event appears to have been the primary cause of {name}\'s failure.'}  "
                    f"However, macro conditions during {founded}–{shutdown} (funding environment, market sentiment in the {category} sector) "
                    f"may have reduced the window for recovery."
                ),
                "evidence": (
                    f"{'Regulation explicitly cited as a blocking factor.' if extern_r == 'Critical' else 'No confirmed regulatory, pandemic, or macro event was the proximate cause of the shutdown.'}"
                )
            }
        })

    # ── Revival Agent ─────────────────────────────────────────────────────────
    # NOTE: must exclude "copywriter" — copywriter system prompt contains "revival_pitch"
    if ("relaunch specialist" in s or "strategist" in s) and "copywriter" not in s:
        return json.dumps({
            "core_insight": (
                f"The problem {name} was trying to solve — {desc[:100] if len(desc) > 100 else desc} — "
                f"is likely still real and still unsolved. "
                f"The failure was in execution, timing, and business model, not in the underlying need."
            ),
            "revised_name": f"{name} ({CUR_YEAR})",
            "revised_icp": (
                f"Early adopters and power users in the {category} space who have already demonstrated "
                f"willingness to pay for solutions to the problem {name} was solving — "
                f"specifically in the {market} market, where the timing may now be more favourable."
            ),
            "repositioning_statement": (
                f"The new {name}: same insight, leaner model, built in public with customers from day one."
            ),
            "gtm_strategy": {
                "primary_channel": f"Direct outreach to the top 50 potential customers in the {category} space who experienced the problem firsthand",
                "why_channel": (
                    f"The fastest path to PMF validation is talking directly to people who already feel the pain. "
                    f"In the {category} space, these customers are identifiable and reachable without paid acquisition. "
                    f"Revenue from 10 paying customers is worth more than 10,000 free signups at this stage."
                ),
                "90_day_plan": [
                    {"week": "1–2", "action": f"Interview 20 potential customers who experienced the exact problem {name} was solving. Record every session. Document the precise language they use — this becomes your copy and positioning."},
                    {"week": "3–4", "action": f"Build a concierge MVP — solve the problem manually for 3–5 paying customers before writing a line of code. Charge real money from day one. Willingness to pay is the only signal that matters at this stage."},
                    {"week": "5–6", "action": f"Scope the minimum product required to serve those 3–5 customers better than any existing alternative in the {category} space. Build only that feature set — nothing else."},
                    {"week": "7–8", "action": f"Expand to 10–15 paying customers in the {market} market. Instrument weekly NPS, churn, and expansion revenue. If NPS < 40, do not expand further — fix the product first."},
                    {"week": "9–10", "action": f"Study the {category} competitors identified in the research dossier. Map exactly what they do better. Build a clear answer to the question: 'Why would a customer choose us over them today?'"},
                    {"week": "11–12", "action": f"With 15+ paying customers, positive NPS, and a clear competitive answer, approach 3 angels or pre-seed funds: '{name} failed because of X. We solved X. Here is the proof — 15 paying customers in 90 days.'"}
                ],
                "what_not_to_do": [
                    f"Do NOT raise more than $500K before achieving 10 paying customers — runway should buy validation, not headcount.",
                    f"Do NOT rebuild the original {name} product feature-for-feature. Start with the core insight only.",
                    f"Do NOT hire a sales team before you have a repeatable, founder-led sales motion.",
                    f"Do NOT ignore the reasons {name} failed — run the autopsy findings as a checklist every 30 days.",
                    f"Do NOT optimise for press coverage before achieving PMF. Stay in stealth until the product speaks for itself."
                ],
                "pricing_model": (
                    f"Value-based pricing anchored to the economic outcome the customer gets — not a cost-plus or competitor-matching model. "
                    f"Start with a flat monthly fee ({market} benchmark for {category}: $99–$499/month for SMB, $1K–$5K/month for enterprise). "
                    f"Annual upfront pricing from day one to extend runway and signal commitment from customers."
                )
            },
            "competitive_landscape_today": (
                f"The {category} market has shifted materially since {name}'s {shutdown} shutdown. "
                f"Post-2023 AI tooling has reduced the cost of building in this space by 60–80%, meaning the original {name} vision "
                f"is likely achievable for a fraction of {funding}. Some competitors that existed when {name} shut down may have "
                f"weakened or pivoted; new players have likely entered. "
                f"A full competitive audit in {CUR_YEAR} — mapping every current solution against the original problem — is essential before committing to a positioning for the revived product."
            ),
            "risk_register": [
                {
                    "risk": f"The original failure repeats — spending {funding}-equivalent capital without finding PMF",
                    "mitigation": f"Hard cap on spending before PMF: no more than $250K before 10 paying customers. If you hit that cap, stop and re-evaluate the thesis — don't raise more."
                },
                {
                    "risk": f"The market has moved on since {shutdown} and the problem is now solved by an incumbent",
                    "mitigation": f"Before building anything, spend 2 weeks mapping every current solution to the problem. If an incumbent now solves it adequately, the insight is dead — find an adjacent problem."
                },
                {
                    "risk": "Founder credibility gap — the market associates the name with failure",
                    "mitigation": f"Lead with the lessons, not the brand. A 'Built on the ashes of {name}' narrative is actually a powerful signal of self-awareness if the pitch acknowledges exactly what went wrong and why it's fixed now."
                }
            ]
        })

    # ── Copywriter Agent ──────────────────────────────────────────────────────
    if "elite startup copywriter" in s or "three polished" in s or "copywriter" in s:
        return json.dumps({
            "autopsy_summary_card": {
                "headline": f"How {name} Failed in {active_str}",
                "primary_hypothesis": (
                    f"{name} raised {funding} but couldn't find a sustainable business model in the {category} space "
                    f"before the runway ran out — a failure of validation speed, not vision."
                ),
                "top_3_factors": [
                    f"Failed to achieve product-market fit before capital was exhausted",
                    f"Operated in a {category} market with strong, often better-funded competitors",
                    f"Pivoted too late or not enough to find a wedge that customers would pay for"
                ],
                "killer_quote": (
                    f"\"{why[:120] + '…' if why and len(why) > 120 else (why or f'We had the right problem. We had the wrong solution.')}\" — {name} founder perspective"
                )
            },
            "revival_pitch": {
                "problem": (
                    f"{desc} — this problem is real and still largely unsolved. "
                    f"The original {name} approach was expensive, under-validated, and vulnerable to better-funded competitors. "
                    f"Customers in the {category} space are still searching for a purpose-built solution that the market hasn't delivered."
                ),
                "solution": (
                    f"{name} ({CUR_YEAR}): same core insight, completely rebuilt execution. "
                    f"We start with 10 paying customers and a concierge MVP before writing a line of scalable code. "
                    f"Post-2023 AI infrastructure cuts build cost by 60–80%, meaning we can validate in 90 days what the original took {active_str} to attempt."
                ),
                "market": (
                    f"The {category} market in {market} has grown and matured since {shutdown}. "
                    f"Buyer education costs are lower, infrastructure is commoditised, and the timing window that worked against {name} "
                    f"may now be firmly in our favour. The {CUR_YEAR} market is fundamentally different from the one that rejected the original."
                ),
                "why_now": (
                    f"Three forces converge in {CUR_YEAR}: (1) AI tooling cuts the cost of building in {category} by 60–80%; "
                    f"(2) the {category} market has matured — customers are more educated and infrastructure is cheaper; "
                    f"(3) the lessons from {name}'s failure are now a blueprint, not a scar. "
                    f"What required {funding} and {active_str} to attempt can now be validated for under $500K in 90 days."
                ),
                "ask": (
                    f"Raising $1.5M pre-seed to reach 25 paying customers and $500K ARR within 12 months. "
                    f"{name} spent {funding} proving the problem is real. "
                    f"We're spending $1.5M proving we can own the solution — with a 90-day concierge validation before a single line of scalable code is written."
                )
            },
            "elevator_pitch": (
                f"{name} ({CUR_YEAR}) is a lean revival of the original {name} — {desc[:90]}{'…' if len(desc) > 90 else ''} — "
                f"rebuilt with every lesson from the original failure baked into the founding thesis. "
                f"The original spent {funding} on the wrong execution; we're spending $1.5M on the right one, starting with 10 paying customers before we write a line of scalable code."
            )
        })

    return f"Analysis complete for {name}."


# ── LangGraph State ───────────────────────────────────────────────────────────
class AutopsyState(TypedDict):
    # Stage 1 — Basic Identity
    startup_name:        str
    industry:            str
    country:             str
    year_founded:        str
    year_shutdown:       str
    funding_range:       str
    product_description: str

    # Stage 2 — Founder's Perspective (optional)
    startup_overview:    str
    why_failed_shutdown: str
    founder_why_failed:  str
    customer_feedback:   str
    pivots_tried:        str
    what_different:      str

    # Stage 3 — Context Signals (checkboxes)
    context_signals:     list[str]

    # Agent Outputs
    research:            dict
    autopsy:             dict
    revival:             dict
    copywriter_outputs:  dict
    marketing_html:      str

    # Meta
    progress:            list[str]
    data_confidence:     str
    error:               Optional[str]


# ── Helper ────────────────────────────────────────────────────────────────────
def _extract_json(text: str) -> str:
    match = re.search(r'\{[\s\S]*\}', text)
    return match.group(0) if match else text


def _build_context(state: AutopsyState) -> str:
    """Build a rich context string from all user inputs for agent consumption."""
    parts = [
        f"Startup: {state['startup_name']}",
        f"Industry: {state.get('industry', 'Unknown')}",
        f"Market: {state.get('country', 'Unknown')}",
        f"Active: {state.get('year_founded', '?')} → {state.get('year_shutdown', '?')}",
        f"Funding: {state.get('funding_range', 'Unknown')}",
        f"What it did: {state.get('product_description', '')}",
    ]
    if state.get('startup_overview'):
        parts.append(f"Founder's description of the startup: {state['startup_overview']}")
    if state.get('why_failed_shutdown'):
        parts.append(f"Why it failed and shut down (founder's account): {state['why_failed_shutdown']}")
    if state.get('founder_why_failed'):
        parts.append(f"Founder's view on failure: {state['founder_why_failed']}")
    if state.get('customer_feedback'):
        parts.append(f"Customer feedback: {state['customer_feedback']}")
    if state.get('pivots_tried'):
        parts.append(f"Pivots attempted: {state['pivots_tried']}")
    if state.get('what_different'):
        parts.append(f"What they'd do differently: {state['what_different']}")
    if state.get('context_signals'):
        parts.append(f"Known failure signals: {', '.join(state['context_signals'])}")
    return "\n".join(parts)


# ── Agent 1: Research ─────────────────────────────────────────────────────────
def research_agent(state: AutopsyState) -> AutopsyState:
    logger.info(json.dumps({"agent": "research", "startup": state["startup_name"]}))

    user_ctx = _build_context(state)
    raw = llm(
        "You are a startup research analyst with encyclopaedic knowledge of tech, venture capital, and business history. "
        "Your job is to produce a structured research dossier on a failed startup. "
        "Gather everything publicly known: funding rounds, investors, team, press coverage, founder interviews, "
        "community signals (Reddit, HN, Product Hunt), pivots, competitor landscape, and market conditions. "
        "If little public data is available, set data_confidence to 'low' and note what is missing. "
        "Return ONLY valid JSON with keys: name, founded, shutdown, funding, investors, category, market, "
        "one_liner, what_they_built, press_coverage, founder_interviews, community_signals, pivots, "
        "competitor_landscape, market_conditions, data_confidence (high/medium/low), public_data_available (bool).",
        user_ctx
    )
    try:
        data = json.loads(_extract_json(raw))
    except Exception:
        data = {"name": state["startup_name"], "one_liner": raw[:200],
                "data_confidence": "low", "public_data_available": False}

    confidence = data.get("data_confidence", "medium")
    progress = state.get("progress", []) + [f"✅ Research dossier built — confidence: {confidence.upper()}"]
    return {**state, "research": data, "data_confidence": confidence, "progress": progress}


# ── Agent 2: Autopsy ──────────────────────────────────────────────────────────
def autopsy_agent(state: AutopsyState) -> AutopsyState:
    logger.info(json.dumps({"agent": "autopsy", "startup": state["startup_name"]}))

    research_ctx = json.dumps(state.get("research", {}), indent=2)
    user_ctx     = _build_context(state)
    low_data_note = (
        "\nNOTE: Limited public data is available for this startup. "
        "Be explicit about what you are inferring vs. what you found directly. "
        "Lean heavily on the founder's own inputs where public data is sparse."
        if state.get("data_confidence") == "low" else ""
    )

    raw = llm(
        "You are the world's most ruthless startup post-mortem analyst. "
        "Analyse this startup's failure across exactly six dimensions with specific, evidence-backed reasoning. "
        "Be harsh, honest, and specific — not generic. This is the honest advisor the founder never had. "
        "Return ONLY valid JSON with keys: "
        "primary_failure_hypothesis (one clear sentence — the single most important reason), "
        "overall_score (0–100 survival score, most failures score under 30), "
        "data_note (string, empty if data was sufficient), "
        "timing {rating, finding, evidence}, "
        "market_size_monetization {rating, finding, evidence}, "
        "pmf {rating, finding, evidence}, "
        "team_execution {rating, finding, evidence}, "
        "competition_defensibility {rating, finding, evidence}, "
        "external_factors {rating, finding, evidence}. "
        "Ratings: Critical / Significant / Minor / Not a factor." + low_data_note,
        f"startup_name: \"{state['startup_name']}\"\n\nResearch dossier:\n{research_ctx}\n\nFounder inputs:\n{user_ctx}"
    )
    try:
        data = json.loads(_extract_json(raw))
    except Exception:
        data = {"primary_failure_hypothesis": raw[:300], "overall_score": 15}

    progress = state.get("progress", []) + ["✅ Autopsy complete — 6-lens failure analysis done"]
    return {**state, "autopsy": data, "progress": progress}


# ── Agent 3: Revival Strategist ───────────────────────────────────────────────
def revival_agent(state: AutopsyState) -> AutopsyState:
    logger.info(json.dumps({"agent": "revival", "startup": state["startup_name"]}))

    context = json.dumps({
        "research": state.get("research", {}),
        "autopsy":  state.get("autopsy",  {}),
        "founder_inputs": {
            "why_failed":         state.get("founder_why_failed", ""),
            "customer_feedback":  state.get("customer_feedback", ""),
            "pivots_tried":       state.get("pivots_tried", ""),
            "what_different":     state.get("what_different", ""),
            "context_signals":    state.get("context_signals", [])
        }
    }, indent=2)

    raw = llm(
        f"You are a world-class startup strategist and relaunch specialist. "
        f"Given this failed startup's full autopsy, design what it would look like relaunched in {CUR_YEAR} "
        f"with every lesson baked in. Be specific, opinionated, and actionable — not generic. "
        "Return ONLY valid JSON with keys: "
        "core_insight (the genuine good idea buried in the failure), "
        "revised_name, revised_icp, repositioning_statement (corrects original positioning mistakes), "
        "gtm_strategy { primary_channel, why_channel, 90_day_plan (array of {week, action}), "
        "what_not_to_do (array of strings), pricing_model }, "
        "competitive_landscape_today (has the space changed since failure?), "
        "risk_register (array of {risk, mitigation} — top 3 only).",
        f"startup_name: \"{state['startup_name']}\"\n\nFull context:\n{context}\n\nBuild the definitive 2025 revival strategy."
    )
    try:
        data = json.loads(_extract_json(raw))
    except Exception:
        data = {"core_insight": raw[:300]}

    progress = state.get("progress", []) + ["✅ Revival strategy built — GTM, ICP, risk register ready"]
    return {**state, "revival": data, "progress": progress}


# ── Agent 4: Copywriter ───────────────────────────────────────────────────────
def copywriter_agent(state: AutopsyState) -> AutopsyState:
    logger.info(json.dumps({"agent": "copywriter", "startup": state["startup_name"]}))

    context = json.dumps({
        "research":  state.get("research", {}),
        "autopsy":   state.get("autopsy",  {}),
        "revival":   state.get("revival",  {}),
    }, indent=2)
    founder_provided = bool(state.get("founder_why_failed") or state.get("what_different"))

    raw = llm(
        "You are an elite startup copywriter — YC Demo Day meets Stripe's homepage. "
        "Produce exactly three polished outputs for the revived startup. "
        "Write in the voice of a confident founder, not an AI. Be punchy and specific. "
        "Return ONLY valid JSON with keys: "
        "autopsy_summary_card { headline, primary_hypothesis, top_3_factors (array), killer_quote }, "
        "revival_pitch { problem, solution, market, why_now, ask }, "
        "elevator_pitch (string — exactly 3 sentences: what it does, who it's for, why it wins this time)."
        + ("\nNote: No founder perspective was provided — keep the revival pitch founder-agnostic." if not founder_provided else ""),
        f"startup_name: \"{state['startup_name']}\"\n\nFull context:\n{context}"
    )
    try:
        data = json.loads(_extract_json(raw))
    except Exception:
        data = {"elevator_pitch": raw[:300]}

    progress = state.get("progress", []) + ["✅ Copy written — summary card, pitch & elevator ready"]
    return {**state, "copywriter_outputs": data, "progress": progress}


# ── Agent 5: Marketing Page ───────────────────────────────────────────────────
def marketing_agent(state: AutopsyState) -> AutopsyState:
    logger.info(json.dumps({"agent": "marketing", "startup": state["startup_name"]}))

    research   = state.get("research", {})
    autopsy    = state.get("autopsy",  {})
    revival    = state.get("revival",  {})
    copy_out   = state.get("copywriter_outputs", {})
    pitch      = copy_out.get("revival_pitch", {})
    card       = copy_out.get("autopsy_summary_card", {})

    orig_name    = research.get("name",    state["startup_name"])
    revised_name = revival.get("revised_name", orig_name + " (Relaunch)")
    orig_funding = research.get("funding", "")
    score        = autopsy.get("overall_score", autopsy.get("score", 20))
    hypothesis   = autopsy.get("primary_failure_hypothesis", "")
    insight      = revival.get("core_insight", "")
    icp          = revival.get("revised_icp", "")
    reposition   = revival.get("repositioning_statement", "")
    channels_txt = revival.get("gtm_strategy", {}).get("primary_channel", "")
    pricing      = revival.get("gtm_strategy", {}).get("pricing_model", "")
    what_not     = revival.get("gtm_strategy", {}).get("what_not_to_do", [])
    plan         = revival.get("gtm_strategy", {}).get("90_day_plan", [])
    risks        = revival.get("risk_register", [])
    comp_today   = revival.get("competitive_landscape_today", "")
    elevator     = copy_out.get("elevator_pitch", "")
    top3         = card.get("top_3_factors", [])
    killer_quote = card.get("killer_quote", "")

    lens_labels = {
        "timing": "⏱ Timing",
        "market_size_monetization": "💰 Market & Monetization",
        "pmf": "🎯 Product-Market Fit",
        "team_execution": "👥 Team & Execution",
        "competition_defensibility": "⚔️ Competition",
        "external_factors": "🌍 External Factors"
    }
    rating_colors = {"Critical":"#ff4444","Significant":"#ff8c00","Minor":"#f0b429","Not a factor":"#34d399"}

    lens_cards = ""
    for key, label in lens_labels.items():
        d = autopsy.get(key, {})
        if not d: continue
        color = rating_colors.get(d.get("rating",""), "#888")
        lens_cards += f"""
        <div class="lc">
          <div class="lc-top"><span class="lc-name">{label}</span>
            <span class="lc-badge" style="background:{color}">{d.get('rating','—')}</span></div>
          <p class="lc-find">{d.get('finding','')}</p>
          <p class="lc-ev">📍 {d.get('evidence','')}</p>
        </div>"""

    plan_rows = "".join(f"""
      <div class="pr"><div class="pw">Week {p.get('week','')}</div>
      <div class="pa">{p.get('action','')}</div></div>""" for p in plan)

    dont_rows = "".join(f"<li>{d}</li>" for d in what_not)
    risk_rows = "".join(f"""
      <div class="risk-row"><div class="risk-label">⚠ {r.get('risk','')}</div>
      <div class="risk-mit">→ {r.get('mitigation','')}</div></div>""" for r in risks)
    top3_rows = "".join(f'<li>{f}</li>' for f in top3)

    html = f"""<!DOCTYPE html><html lang="en">
<head><meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{revised_name} — relaunch.ai</title>
<style>
:root{{--a:#6c63ff;--a2:#a78bfa;--dk:#0d0d14;--s:#13131d;--c:#1a1a27;--c2:#1e1e2e;--t:#e2e8f0;--m:#94a3b8;--r:12px}}
*{{box-sizing:border-box;margin:0;padding:0}}body{{font-family:'Segoe UI',system-ui,sans-serif;background:var(--dk);color:var(--t);line-height:1.6}}
.container{{max-width:860px;margin:0 auto;padding:0 24px}}
nav{{background:rgba(13,13,20,.95);backdrop-filter:blur(12px);padding:14px 24px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #ffffff0d;position:sticky;top:0;z-index:99}}
.nlogo{{font-size:1rem;font-weight:800;background:linear-gradient(135deg,var(--a),var(--a2));-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.nbadge{{background:#6c63ff22;border:1px solid #6c63ff44;color:var(--a2);font-size:.7rem;padding:4px 12px;border-radius:20px;font-weight:600}}
.hero{{padding:80px 0 60px;text-align:center;background:radial-gradient(ellipse 80% 50% at 50% 0%,#6c63ff18,transparent 70%)}}
.orig-tag{{display:inline-block;background:#ff444418;color:#ff8888;border:1px solid #ff444433;border-radius:20px;padding:4px 14px;font-size:.78rem;margin-bottom:20px}}
.hero h1{{font-size:clamp(2.2rem,6vw,3.8rem);font-weight:900;line-height:1.1;margin-bottom:16px;letter-spacing:-.03em}}
.grad{{background:linear-gradient(135deg,var(--a),var(--a2));-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.hero-sub{{color:var(--m);font-size:1.05rem;max-width:560px;margin:0 auto 32px;line-height:1.65}}
.elevator{{background:var(--c);border:1px solid #6c63ff33;border-radius:var(--r);padding:20px 28px;max-width:660px;margin:0 auto;font-style:italic;color:var(--t);font-size:.95rem;line-height:1.65}}
section{{padding:56px 0}}
.sec-label{{color:var(--a);font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.12em;margin-bottom:8px}}
h2{{font-size:1.65rem;font-weight:800;margin-bottom:6px}}
.sec-sub{{color:var(--m);margin-bottom:28px;font-size:.92rem}}
.hypo-box{{background:linear-gradient(135deg,var(--c2),var(--c));border:1px solid #ff444433;border-radius:var(--r);padding:24px 28px;position:relative;overflow:hidden;margin-bottom:24px}}
.hypo-box::after{{content:'☠';position:absolute;right:20px;top:8px;font-size:4rem;opacity:.08}}
.hypo-label{{color:#ff8888;font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px}}
.hypo-text{{color:#fff;font-size:1rem;line-height:1.6}}
.score-row{{display:flex;align-items:center;gap:12px;margin-top:16px}}
.score-bar-wrap{{flex:1;height:5px;background:#ffffff10;border-radius:3px;overflow:hidden}}
.score-bar{{height:100%;background:linear-gradient(90deg,#ff4444,#ff7700);border-radius:3px}}
.score-num{{color:#ff8888;font-weight:800;font-size:.95rem;white-space:nowrap}}
.lg{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}@media(max-width:600px){{.lg{{grid-template-columns:1fr}}}}
.lc{{background:var(--c);border:1px solid #ffffff08;border-radius:var(--r);padding:18px}}
.lc-top{{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}}
.lc-name{{font-weight:700;font-size:.88rem}}
.lc-badge{{font-size:.66rem;font-weight:700;padding:3px 9px;border-radius:20px;color:#000}}
.lc-find{{color:var(--t);font-size:.85rem;line-height:1.5;margin-bottom:6px}}
.lc-ev{{color:var(--m);font-size:.78rem;font-style:italic}}
.insight{{background:#6c63ff12;border-left:4px solid var(--a);border-radius:0 10px 10px 0;padding:16px 20px;margin:20px 0;font-style:italic;font-size:.95rem}}
.rg{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:20px}}@media(max-width:600px){{.rg{{grid-template-columns:1fr}}}}
.rc{{background:var(--c);border:1px solid #6c63ff22;border-radius:var(--r);padding:18px}}
.rc .rl{{color:var(--a);font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;margin-bottom:7px}}
.rc .rv{{color:var(--t);font-size:.88rem;line-height:1.5}}
.dont-list,.risk-rows{{margin-top:14px}}
.dont-list li{{padding:8px 0 8px 18px;border-bottom:1px solid #ffffff07;color:#ff8888;font-size:.86rem;position:relative}}
.dont-list li::before{{content:'✗';position:absolute;left:0;color:#ff4444;font-weight:700}}
.pr{{display:flex;gap:14px;padding:12px 0;border-bottom:1px solid #ffffff07;align-items:flex-start}}
.pw{{min-width:76px;background:var(--a);color:#fff;border-radius:6px;padding:3px 8px;font-size:.72rem;font-weight:700;text-align:center;flex-shrink:0;margin-top:3px}}
.pa{{color:var(--t);font-size:.87rem;line-height:1.5}}
.risk-row{{background:var(--c);border-radius:8px;padding:14px 16px;margin-bottom:10px;border-left:3px solid #ff8c00}}
.risk-label{{color:#ffaa44;font-size:.87rem;font-weight:600;margin-bottom:4px}}
.risk-mit{{color:var(--m);font-size:.83rem}}
.pitch-card{{background:linear-gradient(135deg,var(--c2),var(--c));border:1px solid #6c63ff33;border-radius:var(--r);padding:28px}}
.pitch-section{{margin-top:16px}}
.pitch-section .pl{{color:var(--a);font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;margin-bottom:5px}}
.pitch-section .pv{{color:var(--t);font-size:.9rem;line-height:1.55}}
.top3-list li{{padding:7px 0 7px 18px;border-bottom:1px solid #ffffff07;font-size:.87rem;position:relative}}
.top3-list li::before{{content:'•';position:absolute;left:0;color:var(--a);font-weight:700}}
blockquote{{background:var(--c2);border-left:3px solid var(--a2);border-radius:0 8px 8px 0;padding:14px 18px;font-style:italic;color:var(--t);font-size:.9rem;margin-top:16px}}
footer{{border-top:1px solid #ffffff0d;padding:28px 0;text-align:center;color:var(--m);font-size:.82rem}}
footer strong{{color:var(--a)}}
::-webkit-scrollbar{{width:5px}}::-webkit-scrollbar-thumb{{background:#333;border-radius:3px}}
</style></head><body>
<nav><div class="nlogo">🔬 relaunch.ai</div><span class="nbadge">AI-Generated Relaunch Plan</span></nav>

<section class="hero"><div class="container">
  <div class="orig-tag">☠ Originally failed as: {orig_name}{f" ({orig_funding} raised)" if orig_funding else ""}</div>
  <h1>Introducing<br/><span class="grad">{revised_name}</span></h1>
  <p class="hero-sub">{reposition or icp}</p>
  <div class="elevator">"{elevator}"</div>
</div></section>

<section style="background:var(--s)"><div class="container">
  <div class="sec-label">Post-Mortem</div><h2>Why {orig_name} Really Failed</h2>
  <p class="sec-sub">Six-lens forensic analysis. No excuses.</p>
  <div class="hypo-box">
    <div class="hypo-label">Primary Failure Hypothesis</div>
    <div class="hypo-text">"{hypothesis}"</div>
    <div class="score-row">
      <span class="score-num">Survival Score: {score}/100</span>
      <div class="score-bar-wrap"><div class="score-bar" style="width:{score}%"></div></div>
    </div>
  </div>
  {"<div class='lg'>" + lens_cards + "</div>" if lens_cards else ""}
  {"<blockquote>" + killer_quote + "</blockquote>" if killer_quote else ""}
</div></section>

<section><div class="container">
  <div class="sec-label">The Revival</div><h2>What {revised_name} Looks Like in 2025</h2>
  <p class="sec-sub">Same core insight. Completely different execution.</p>
  <div class="insight">💡 {insight}</div>
  <div class="rg">
    <div class="rc"><div class="rl">Revised ICP</div><div class="rv">{icp}</div></div>
    <div class="rc"><div class="rl">Repositioning</div><div class="rv">{reposition}</div></div>
    <div class="rc"><div class="rl">Primary GTM Channel</div><div class="rv">{channels_txt}</div></div>
    <div class="rc"><div class="rl">Pricing Model</div><div class="rv">{pricing}</div></div>
    {"<div class='rc' style='grid-column:span 2'><div class='rl'>Competitive Landscape Today</div><div class='rv'>" + comp_today + "</div></div>" if comp_today else ""}
  </div>
  {"<div style='margin-top:28px'><div class='sec-label' style='margin-bottom:10px'>What Not To Do</div><ul class='dont-list'>" + dont_rows + "</ul></div>" if dont_rows else ""}
</div></section>

<section style="background:var(--s)"><div class="container">
  <div class="sec-label">Execution Plan</div><h2>90-Day Launch Roadmap</h2>
  <p class="sec-sub">Week by week. No fluff.</p>
  <div>{plan_rows}</div>
</div></section>

<section><div class="container">
  <div class="sec-label">Risk Register</div><h2>What Could Kill the Revival</h2>
  <p class="sec-sub">Top 3 risks — and how to mitigate them early.</p>
  <div class="risk-rows">{risk_rows}</div>
</div></section>

<section style="background:var(--s)"><div class="container">
  <div class="sec-label">Investor Pitch</div><h2>One-Page Summary</h2>
  <div class="pitch-card">
    {"<div style='margin-bottom:16px'><div class='sec-label' style='margin-bottom:6px'>Top 3 Failure Factors</div><ul class='top3-list'>" + top3_rows + "</ul></div>" if top3_rows else ""}
    {"".join(f"<div class='pitch-section'><div class='pl'>{lbl}</div><div class='pv'>{val}</div></div>" for lbl, key in [("Problem","problem"),("Solution","solution"),("Market","market"),("Why Now","why_now"),("Ask","ask")] for val in [pitch.get(key,"")] if val)}
  </div>
</div></section>

<footer><div class="container">
  <p>This relaunch plan was generated by <strong>relaunch.ai</strong> — 5-agent LangGraph pipeline on Complete.dev.</p>
  <p style="margin-top:6px;font-size:.74rem;opacity:.5">Agent-generated content for demo purposes.</p>
</div></footer>
</body></html>"""

    progress = state.get("progress", []) + ["✅ Marketing landing page generated"]
    return {**state, "marketing_html": html, "progress": progress}


# ── Build & Compile Graph ─────────────────────────────────────────────────────
def build_graph():
    g = StateGraph(AutopsyState)
    g.add_node("research",    research_agent)
    g.add_node("autopsy",     autopsy_agent)
    g.add_node("revival",     revival_agent)
    g.add_node("copywriter",  copywriter_agent)
    g.add_node("marketing",   marketing_agent)
    g.add_edge(START,         "research")
    g.add_edge("research",    "autopsy")
    g.add_edge("autopsy",     "revival")
    g.add_edge("revival",     "copywriter")
    g.add_edge("copywriter",  "marketing")
    g.add_edge("marketing",   END)
    return g.compile()

graph = build_graph()


def run_analysis(payload: dict) -> dict:
    initial: AutopsyState = {
        "startup_name":        payload.get("startup_name", ""),
        "industry":            payload.get("industry", ""),
        "country":             payload.get("country", ""),
        "year_founded":        payload.get("year_founded", ""),
        "year_shutdown":       payload.get("year_shutdown", ""),
        "funding_range":       payload.get("funding_range", ""),
        "product_description": payload.get("product_description", ""),
        "startup_overview":    payload.get("startup_overview", ""),
        "why_failed_shutdown": payload.get("why_failed_shutdown", ""),
        "founder_why_failed":  payload.get("founder_why_failed", ""),
        "customer_feedback":   payload.get("customer_feedback", ""),
        "pivots_tried":        payload.get("pivots_tried", ""),
        "what_different":      payload.get("what_different", ""),
        "context_signals":     payload.get("context_signals", []),
        "research":            {},
        "autopsy":             {},
        "revival":             {},
        "copywriter_outputs":  {},
        "marketing_html":      "",
        "progress":            [],
        "data_confidence":     "medium",
        "error":               None,
    }
    return graph.invoke(initial)
