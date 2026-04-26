import json
import os


def load_jsons(output_dir: str = "outputs") -> list:
    results, flagged = [], []

    for fname in os.listdir(output_dir):
        if not fname.endswith(".json") or fname == "summary.json":
            continue

        path = os.path.join(output_dir, fname)
        try:
            with open(path) as f:
                data = json.load(f)
            data["_source"] = fname
            results.append(data)
        except Exception as e:
            print(f"[SKIP] Could not read {fname}: {e}")
            flagged.append(fname)

    return results, flagged


def safe_float(value) -> float:
    try:
        return float(str(value).replace(",", "").replace("$", ""))
    except:
        return 0.0


def get_total(receipt: dict) -> float:
    """Get total from total_amount field, fallback to summing items."""
    total_val = receipt.get("total_amount", {}).get("value")

    if total_val:
        return safe_float(total_val)

    # Fallback — sum item prices
    items = receipt.get("items", [])
    if items:
        print(f"[FALLBACK] No total in {receipt['_source']} — summing items")
        return round(sum(safe_float(i.get("price", 0)) for i in items), 2)

    return 0.0


def get_store(receipt: dict) -> str:
    return receipt.get("store_name", {}).get("value") or "Unknown"


def generate_summary(output_dir: str = "outputs") -> dict:
    receipts, flagged = load_jsons(output_dir)

    if not receipts:
        print("[WARN] No valid receipts found.")
        return {}

    spend_per_store = {}
    all_items       = []
    transactions    = []

    for r in receipts:
        store  = get_store(r)
        total  = get_total(r)
        source = r["_source"]

        # Flag receipts with missing critical fields
        missing = r.get("flags", {}).get("missing_fields", [])
        if missing:
            print(f"[FLAG] {source} missing: {missing}")
            flagged.append(source)

        spend_per_store[store] = round(spend_per_store.get(store, 0) + total, 2)
        transactions.append({"source": source, "store": store, "total": total})

        for item in r.get("items", []):
            all_items.append({
                "store" : store,
                "name"  : item.get("name"),
                "price" : item.get("price"),
            })

    total_spend  = round(sum(t["total"] for t in transactions), 2)
    num_tx       = len(transactions)
    avg_tx       = round(total_spend / num_tx, 2) if num_tx else 0.0
    most_exp     = max(transactions, key=lambda t: t["total"])

    summary = {
        "total_spend"      : total_spend,
        "num_transactions" : num_tx,
        "avg_transaction"  : avg_tx,
        "most_expensive"   : {"store": most_exp["store"], "amount": most_exp["total"]},
        "spend_per_store"  : spend_per_store,
        "all_items"        : all_items,
        "flagged_receipts" : list(set(flagged)),
    }

    out_path = os.path.join(output_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print_summary(summary)
    print(f"\nSaved -> {out_path}")
    return summary


def print_summary(s: dict):
    print("\n" + "="*50)
    print("FINANCIAL SUMMARY")
    print("="*50)
    print(f"Total Spend      : ${s['total_spend']:.2f}")
    print(f"Transactions     : {s['num_transactions']}")
    print(f"Avg Transaction  : ${s['avg_transaction']:.2f}")
    print(f"Most Expensive   : {s['most_expensive']['store']} (${s['most_expensive']['amount']:.2f})")
    print("\nSpend per Store:")
    for store, amt in s["spend_per_store"].items():
        print(f"  {store:<30} ${amt:.2f}")
    if s["flagged_receipts"]:
        print(f"\nFlagged Receipts : {s['flagged_receipts']}")
    print("="*50)


if __name__ == "__main__":
    generate_summary()