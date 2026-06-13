"""Offline RAG retrieval acceptance check.

Requires a Redis Stack server on REDIS_URL (default localhost:6379). No Google
key is required; ingest will build a BM25/structured index and skip live
embedding unless GOOGLE_API_KEY is set.

Run from the repo root:
    python cs_agent/offline_retrieval_test.py
"""

from ingest import build_index
from rag_tools import kb_search


CASES = [
    {
        "query": "overdraft fee Blue Account",
        "text_needles": ["blue account"],
        "tool_needles": ["Blue Account"],
    },
    {
        "query": "open a fee-free account",
        "text_needles": ["green fee-free account"],
        "tool_needles": ["open_bank_account_4821"],
    },
    {
        "query": "refer a friend",
        "text_needles": ["refer", "referral"],
        "tool_needles": ["submit_referral", "get_referral_link"],
        "match_any_text": True,
        "match_any_tool": True,
    },
]


def _haystack(result: dict) -> str:
    return " ".join(
        str(result.get(field, ""))
        for field in ("doc_id", "title", "content", "tool_names", "keywords")
    ).lower()


def _has_expected_result(result: dict, case: dict) -> bool:
    haystack = _haystack(result)
    tool_names = str(result.get("tool_names", ""))
    text_needles = [needle.lower() for needle in case["text_needles"]]
    tool_needles = case["tool_needles"]
    text_ok = (
        any(needle in haystack for needle in text_needles)
        if case.get("match_any_text")
        else all(needle in haystack for needle in text_needles)
    )
    tool_ok = (
        any(needle in tool_names for needle in tool_needles)
        if case.get("match_any_tool")
        else all(needle in tool_names for needle in tool_needles)
    )
    return text_ok and tool_ok


def main() -> None:
    build_index()
    for case in CASES:
        query = case["query"]
        results = kb_search(query)
        print(f"\nQUERY: {query}")
        for rank, result in enumerate(results[:5], start=1):
            snippet = " ".join(str(result.get("content", "")).split())[:220]
            print(
                f"{rank}. {result.get('doc_id')} | {result.get('title')} | "
                f"tool_names={result.get('tool_names', '')} | snippet={snippet}"
            )
        assert any(_has_expected_result(result, case) for result in results), (
            f"Expected doc/tool_names evidence not found for query: {query}"
        )
    print("\nAll retrieval assertions passed.")


if __name__ == "__main__":
    main()
