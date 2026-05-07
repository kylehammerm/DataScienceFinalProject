import requests
import json
import time
import re
import os
import pickle
import pandas as pd
from difflib import SequenceMatcher

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

CHECKPOINT_FILE = "polymarket_checkpoint.pkl"


# =========================
# Basic Helpers
# =========================

def safe_json_loads(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except:
            return None
    return x


def extract_token_ids(market):
    return safe_json_loads(market.get("clobTokenIds")) or []


def extract_outcomes(market):
    return safe_json_loads(market.get("outcomes")) or []


def extract_outcome_prices(market):
    prices = safe_json_loads(market.get("outcomePrices")) or []

    clean_prices = []
    for p in prices:
        try:
            clean_prices.append(float(p))
        except:
            clean_prices.append(None)

    return clean_prices


def get_winning_outcome_from_final_price(market, min_winner_price=0.95):
    outcomes = extract_outcomes(market)
    prices = extract_outcome_prices(market)

    if not outcomes or not prices:
        return None

    if len(outcomes) != len(prices):
        return None

    valid = [(i, p) for i, p in enumerate(prices) if p is not None]

    if not valid:
        return None

    winner_idx, winner_price = max(valid, key=lambda x: x[1])

    if winner_price < min_winner_price:
        return None

    return outcomes[winner_idx]


def classify_winner_type(winning_outcome):
    w = str(winning_outcome).strip().lower()

    if w == "yes":
        return "yes"
    if w == "no":
        return "no"

    return "other"


def market_is_closed_or_resolved(market):
    return (
        market.get("closed") is True
        or market.get("resolved") is True
        or market.get("active") is False
    )


# =========================
# Similarity Filtering
# =========================

def normalize_question(q):
    q = str(q).lower()
    q = re.sub(r"\$?\d+(\.\d+)?%?", " NUMBER ", q)
    q = re.sub(r"[^a-z\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def question_similarity(q1, q2):
    return SequenceMatcher(
        None,
        normalize_question(q1),
        normalize_question(q2),
    ).ratio()


def find_similar_group(market, groups, threshold=0.65):
    q = market.get("question", "")

    for group_id, group in groups.items():
        sim = question_similarity(q, group["rep_question"])
        if sim >= threshold:
            return group_id

    return None


# =========================
# Checkpoint Helpers
# =========================

def save_checkpoint(
    offset,
    scanned_events,
    groups,
    completed_combo_groups,
    seen_market_slugs,
    price_rows,
    skipped_rows,
    accepted_market_summaries,
    n,
):
    checkpoint = {
        "offset": offset,
        "scanned_events": scanned_events,
        "groups": groups,
        "completed_combo_groups": completed_combo_groups,
        "seen_market_slugs": seen_market_slugs,
        "price_rows": price_rows,
        "skipped_rows": skipped_rows,
        "accepted_market_summaries": accepted_market_summaries,
        "n": n,
    }

    with open(CHECKPOINT_FILE, "wb") as f:
        pickle.dump(checkpoint, f)

    save_partial_csvs(
        price_rows,
        skipped_rows,
        accepted_market_summaries,
        n,
    )


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return None

    with open(CHECKPOINT_FILE, "rb") as f:
        return pickle.load(f)


def clear_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print(f"Deleted checkpoint file: {CHECKPOINT_FILE}")


def save_partial_csvs(price_rows, skipped_rows, accepted_market_summaries, n):
    partial_price_filename = f"partial_polymarket_daily_prices_{n}_yes_no_combos.csv"
    partial_summary_filename = f"partial_polymarket_market_summary_{n}_yes_no_combos.csv"
    partial_skipped_filename = f"partial_polymarket_skipped_{n}_yes_no_combos.csv"

    pd.DataFrame(price_rows).to_csv(partial_price_filename, index=False)
    pd.DataFrame(accepted_market_summaries).to_csv(partial_summary_filename, index=False)
    pd.DataFrame(skipped_rows).to_csv(partial_skipped_filename, index=False)

    print(f"Partial save complete: {partial_price_filename}")


# =========================
# API Calls
# =========================

def get_recent_closed_events(limit=100, offset=0):
    r = requests.get(
        f"{GAMMA}/events",
        params={
            "limit": limit,
            "offset": offset,
            "closed": "true",
            "order": "endDate",
            "ascending": "false",
        },
        timeout=15,
    )

    if r.status_code == 422:
        print(f"Reached Gamma API pagination limit at offset={offset}.")
        return []

    r.raise_for_status()
    return r.json()


def get_daily_price_history_for_month(token_id):
    r = requests.get(
        f"{CLOB}/prices-history",
        params={
            "market": token_id,
            "interval": "1m",
            "fidelity": 1440,
        },
        timeout=15,
    )

    if r.status_code != 200:
        return pd.DataFrame()

    history = r.json().get("history", [])
    df = pd.DataFrame(history)

    if df.empty:
        return df

    df = df.rename(columns={"t": "timestamp", "p": "price"})

    df["datetime"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        errors="coerce",
        utc=True,
    )

    df["date"] = df["datetime"].dt.date
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = df.dropna(subset=["datetime", "price"])
    df = df[(df["price"] >= 0) & (df["price"] <= 1)]

    return df[["date", "datetime", "price"]]


# =========================
# Logging + Metadata
# =========================

def add_skipped(skipped_rows, event, market, reason):
    skipped_rows.append({
        "reason": reason,
        "event_title": event.get("title"),
        "event_slug": event.get("slug"),
        "market_question": market.get("question"),
        "market_slug": market.get("slug"),
        "condition_id": market.get("conditionId"),
        "outcomes": market.get("outcomes"),
        "outcomePrices": market.get("outcomePrices"),
        "inferred_winning_outcome": get_winning_outcome_from_final_price(market),
        "volume": market.get("volume"),
        "liquidity": market.get("liquidity"),
        "endDate": market.get("endDate"),
    })


def get_market_metadata(event, market):
    return {
        "event_title": event.get("title"),
        "event_slug": event.get("slug"),
        "market_question": market.get("question"),
        "market_slug": market.get("slug"),
        "condition_id": market.get("conditionId"),
        "category": (
            market.get("category")
            or event.get("category")
            or event.get("categorySlug")
        ),
        "startDate": market.get("startDate"),
        "endDate": market.get("endDate"),
        "closedTime": market.get("closedTime"),
        "volume": market.get("volume"),
        "liquidity": market.get("liquidity"),
        "lastTradePrice": market.get("lastTradePrice"),
        "bestBid": market.get("bestBid"),
        "bestAsk": market.get("bestAsk"),
        "closed": market.get("closed"),
        "resolved": market.get("resolved"),
        "active": market.get("active"),
        "outcomes_raw": market.get("outcomes"),
        "outcomePrices_raw": market.get("outcomePrices"),
    }


# =========================
# Dataset Builder
# =========================

def build_dataset(
    n=100,
    max_events_to_scan=20000,
    event_page_size=100,
    sleep_seconds=0.05,
    similarity_threshold=0.65,
    min_price_points=10,
    min_volume=1000,
    min_hours_to_resolution=24,
    min_winner_price=0.95,
    resume=True,
    checkpoint_every_accepted=10,
):
    checkpoint = load_checkpoint() if resume else None

    if checkpoint is not None:
        print("Resuming from checkpoint...")

        offset = checkpoint["offset"]
        scanned_events = checkpoint["scanned_events"]
        groups = checkpoint["groups"]
        completed_combo_groups = checkpoint["completed_combo_groups"]
        seen_market_slugs = checkpoint["seen_market_slugs"]
        price_rows = checkpoint["price_rows"]
        skipped_rows = checkpoint["skipped_rows"]
        accepted_market_summaries = checkpoint["accepted_market_summaries"]

        if checkpoint.get("n") != n:
            print(
                f"Warning: checkpoint was created for n={checkpoint.get('n')}, "
                f"but current run uses n={n}."
            )

        next_group_id = max(groups.keys()) + 1 if groups else 0

        print(f"Starting from offset={offset}")
        print(f"Already completed combos={len(completed_combo_groups)}")
        print(f"Already accepted markets={len(accepted_market_summaries)}")

    else:
        price_rows = []
        skipped_rows = []
        accepted_market_summaries = []

        groups = {}
        completed_combo_groups = set()
        seen_market_slugs = set()

        next_group_id = 0
        offset = 0
        scanned_events = 0

    accepted_since_checkpoint = 0

    while len(completed_combo_groups) < n and scanned_events < max_events_to_scan:
        events = get_recent_closed_events(
            limit=event_page_size,
            offset=offset,
        )

        if not events:
            print("No more events returned or API pagination limit reached.")
            break

        for event in events:
            scanned_events += 1

            for market in event.get("markets", []):
                if len(completed_combo_groups) >= n:
                    break

                market_slug = market.get("slug")

                if market_slug in seen_market_slugs:
                    continue

                seen_market_slugs.add(market_slug)

                if not market_is_closed_or_resolved(market):
                    add_skipped(skipped_rows, event, market, "not_closed_or_resolved")
                    continue

                token_ids = extract_token_ids(market)
                if not token_ids:
                    add_skipped(skipped_rows, event, market, "no_token_ids")
                    continue

                outcomes = extract_outcomes(market)
                if not outcomes:
                    add_skipped(skipped_rows, event, market, "no_outcomes")
                    continue

                winning_outcome = get_winning_outcome_from_final_price(
                    market,
                    min_winner_price=min_winner_price,
                )

                if winning_outcome is None:
                    add_skipped(
                        skipped_rows,
                        event,
                        market,
                        "could_not_infer_winner_from_final_price",
                    )
                    continue

                winner_type = classify_winner_type(winning_outcome)

                if winner_type not in ["yes", "no"]:
                    add_skipped(skipped_rows, event, market, "winner_not_yes_no")
                    continue

                volume = pd.to_numeric(market.get("volume"), errors="coerce")

                if pd.isna(volume) or volume < min_volume:
                    add_skipped(skipped_rows, event, market, "low_or_missing_volume")
                    continue

                metadata = get_market_metadata(event, market)

                end_dt = pd.to_datetime(
                    metadata["endDate"],
                    errors="coerce",
                    utc=True,
                )

                if pd.isna(end_dt):
                    add_skipped(skipped_rows, event, market, "missing_endDate")
                    continue

                group_id = find_similar_group(
                    market,
                    groups,
                    threshold=similarity_threshold,
                )

                if group_id is None:
                    group_id = next_group_id
                    next_group_id += 1

                    groups[group_id] = {
                        "rep_question": market.get("question", ""),
                        "yes_market_slug": None,
                        "no_market_slug": None,
                    }

                group = groups[group_id]

                if group[f"{winner_type}_market_slug"] is not None:
                    add_skipped(
                        skipped_rows,
                        event,
                        market,
                        f"similar_{winner_type}_already_saved",
                    )
                    continue

                market_rows = []

                for j, token_id in enumerate(token_ids):
                    outcome = outcomes[j] if j < len(outcomes) else f"Outcome {j}"

                    hist = get_daily_price_history_for_month(token_id)
                    time.sleep(sleep_seconds)

                    if hist.empty:
                        continue

                    hist["datetime"] = pd.to_datetime(
                        hist["datetime"],
                        errors="coerce",
                        utc=True,
                    )

                    hist = hist.dropna(subset=["datetime"])

                    hist["hours_to_resolution"] = (
                        end_dt - hist["datetime"]
                    ).dt.total_seconds() / 3600

                    hist = hist[
                        hist["hours_to_resolution"] >= min_hours_to_resolution
                    ]

                    if hist.empty:
                        continue

                    label = 1 if str(outcome) == str(winning_outcome) else 0

                    for _, row in hist.iterrows():
                        market_rows.append({
                            **metadata,
                            "similarity_group": group_id,
                            "outcome": outcome,
                            "token_id": token_id,
                            "date": row["date"],
                            "datetime": row["datetime"],
                            "price": row["price"],
                            "hours_to_resolution": row["hours_to_resolution"],
                            "label": label,
                            "winning_outcome": winning_outcome,
                            "winner_type": winner_type,
                            "winner_inference_method": "final_outcomePrices",
                        })

                if len(market_rows) < min_price_points:
                    add_skipped(skipped_rows, event, market, "too_few_price_points")
                    continue

                market_df = pd.DataFrame(market_rows)

                if len(token_ids) == 2:
                    unique_outcomes_with_history = market_df["outcome"].nunique()

                    if unique_outcomes_with_history < 2:
                        add_skipped(
                            skipped_rows,
                            event,
                            market,
                            "binary_market_missing_one_side_history",
                        )
                        continue

                price_rows.extend(market_rows)
                group[f"{winner_type}_market_slug"] = market_slug

                summary = {
                    **metadata,
                    "similarity_group": group_id,
                    "winning_outcome": winning_outcome,
                    "winner_type": winner_type,
                    "winner_inference_method": "final_outcomePrices",
                    "num_price_rows": len(market_df),
                    "num_unique_dates": market_df["date"].nunique(),
                    "first_datetime": market_df["datetime"].min(),
                    "last_datetime": market_df["datetime"].max(),
                    "first_price_mean": market_df.groupby("outcome")["price"].first().mean(),
                    "last_price_mean": market_df.groupby("outcome")["price"].last().mean(),
                    "min_price": market_df["price"].min(),
                    "max_price": market_df["price"].max(),
                    "mean_price": market_df["price"].mean(),
                }

                accepted_market_summaries.append(summary)

                if (
                    group["yes_market_slug"] is not None
                    and group["no_market_slug"] is not None
                ):
                    completed_combo_groups.add(group_id)

                accepted_since_checkpoint += 1

                print(
                    f"Accepted market | combos {len(completed_combo_groups)}/{n} | "
                    f"group {group_id} | {winner_type.upper()} winner | "
                    f"{market.get('question')}"
                )

                if accepted_since_checkpoint >= checkpoint_every_accepted:
                    save_checkpoint(
                        offset=offset,
                        scanned_events=scanned_events,
                        groups=groups,
                        completed_combo_groups=completed_combo_groups,
                        seen_market_slugs=seen_market_slugs,
                        price_rows=price_rows,
                        skipped_rows=skipped_rows,
                        accepted_market_summaries=accepted_market_summaries,
                        n=n,
                    )

                    accepted_since_checkpoint = 0
                    print("Checkpoint saved.")

            if len(completed_combo_groups) >= n:
                break

        offset += event_page_size

        save_checkpoint(
            offset=offset,
            scanned_events=scanned_events,
            groups=groups,
            completed_combo_groups=completed_combo_groups,
            seen_market_slugs=seen_market_slugs,
            price_rows=price_rows,
            skipped_rows=skipped_rows,
            accepted_market_summaries=accepted_market_summaries,
            n=n,
        )

    price_df = pd.DataFrame(price_rows)
    market_summary_df = pd.DataFrame(accepted_market_summaries)
    skipped_df = pd.DataFrame(skipped_rows)

    completed_groups = set(completed_combo_groups)

    if not price_df.empty:
        price_df = price_df[price_df["similarity_group"].isin(completed_groups)]

    if not market_summary_df.empty:
        market_summary_df = market_summary_df[
            market_summary_df["similarity_group"].isin(completed_groups)
        ]

    return price_df, market_summary_df, skipped_df


# =========================
# Main
# =========================

def main(n=100, resume=True, clear_existing_checkpoint=False):
    if clear_existing_checkpoint:
        clear_checkpoint()

    price_df, market_summary_df, skipped_df = build_dataset(
        n=n,
        max_events_to_scan=150000,
        event_page_size=100,
        sleep_seconds=0.05,
        similarity_threshold=0.65,
        min_price_points=10,
        min_volume=1000,
        min_hours_to_resolution=24,
        min_winner_price=0.95,
        resume=resume,
        checkpoint_every_accepted=10,
    )

    price_filename = f"polymarket_daily_prices_{n}_yes_no_combos.csv"
    summary_filename = f"polymarket_market_summary_{n}_yes_no_combos.csv"
    skipped_filename = f"polymarket_skipped_{n}_yes_no_combos.csv"

    price_df.to_csv(price_filename, index=False)
    market_summary_df.to_csv(summary_filename, index=False)
    skipped_df.to_csv(skipped_filename, index=False)

    print("\nDone.")
    print("Price rows:", price_df.shape)
    print("Market summary rows:", market_summary_df.shape)
    print("Skipped rows:", skipped_df.shape)

    if not skipped_df.empty:
        print("\nSkipped reason counts:")
        print(skipped_df["reason"].value_counts())

    print(f"\nSaved price data to: {price_filename}")
    print(f"Saved market summary to: {summary_filename}")
    print(f"Saved skipped markets to: {skipped_filename}")

    if len(market_summary_df) > 0:
        print("\nFinal dataset saved successfully.")

    return price_df, market_summary_df, skipped_df


if __name__ == "__main__":
    price_df, market_summary_df, skipped_df = main(
        n=1000,
        resume=True,
        clear_existing_checkpoint=False,
    )