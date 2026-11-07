"""Neighbours of the top hits, and joining adjacent chunks into one source."""
import json

import pytest

from app.chat import find_sources
from app.config import Settings
from app.db.connection import connect
from app.ingest.embed import serialize
from app.llm.prompts import format_sources
from app.retrieval.context import carried_hits, expand_hits
from app.retrieval.search import Hit

PATH = '["Ch. 1"]'
PREFIX = "Ch. 1 ›\n\n"


@pytest.fixture
def conn(tmp_path):
    db = connect(tmp_path / "a.sqlite")
    for doc_id in (1, 2):
        db.execute("INSERT INTO documents (id, path, sha256, title, kind) VALUES (?, ?, ?, ?, 'pdf')",
                   (doc_id, f"{doc_id}.pdf", str(doc_id), f"Book {doc_id}"))
    return db


def add(db, seq, body=None, doc_id=1, path=PATH, page=None, meta=None, text=None):
    page = page or seq + 1
    row = db.execute(
        "INSERT INTO chunks (doc_id, kind, seq, heading_path, text, token_count, block_ids, "
        "page_start, page_end, meta) VALUES (?, 'text', ?, ?, ?, 10, ?, ?, ?, ?)",
        (doc_id, seq, path, text or PREFIX + (body or f"Chunk {seq}."), json.dumps([100 * doc_id + seq]),
         page, page, json.dumps(meta) if meta else None),
    ).lastrowid
    return row


def hit(db, seq, doc_id=1):
    row = db.execute("SELECT * FROM chunks WHERE doc_id = ? AND seq = ?", (doc_id, seq)).fetchone()
    return Hit(chunk_id=row["id"], doc_id=doc_id, doc_path=f"{doc_id}.pdf", doc_title=f"Book {doc_id}",
               heading_path=["Ch. 1"], text=row["text"], page_start=row["page_start"], page_end=row["page_end"],
               block_ids=json.loads(row["block_ids"]), meta=json.loads(row["meta"] or "{}"), score=1.0,
               seq=seq, chunk_ids=[row["id"]], focus_block_ids=json.loads(row["block_ids"]))


def fill(db, count, doc_id=1):
    for seq in range(count):
        add(db, seq, doc_id=doc_id)


def test_a_top_hit_brings_in_the_chunks_on_both_sides(conn):
    fill(conn, 5)
    [source] = expand_hits(conn, [hit(conn, 2)], top=3)
    assert source.text == f"{PREFIX}Chunk 1.\n\nChunk 2.\n\nChunk 3."
    assert len(source.chunk_ids) == 3


def test_only_the_top_hits_bring_neighbours(conn):
    fill(conn, 30)
    hits = [hit(conn, seq) for seq in (2, 7, 12, 17, 22, 27)]
    sources = expand_hits(conn, hits, top=3)
    assert [len(source.chunk_ids) for source in sources] == [3, 3, 3, 1, 1, 1]


def test_the_first_chunk_of_a_document_has_no_chunk_before_it(conn):
    fill(conn, 3)
    assert len(expand_hits(conn, [hit(conn, 0)], top=3)[0].chunk_ids) == 2


def test_the_last_chunk_of_a_document_has_no_chunk_after_it(conn):
    fill(conn, 3)
    assert len(expand_hits(conn, [hit(conn, 2)], top=3)[0].chunk_ids) == 2


def test_adjacent_hits_become_one_source(conn):
    fill(conn, 6)
    sources = expand_hits(conn, [hit(conn, 2), hit(conn, 3)], top=3)
    assert len(sources) == 1
    assert len(sources[0].chunk_ids) == 4


def test_a_shared_neighbour_joins_two_hits(conn):
    fill(conn, 8)
    [source] = expand_hits(conn, [hit(conn, 2), hit(conn, 4)], top=3)
    assert source.chunk_ids == sorted(source.chunk_ids) and len(source.chunk_ids) == 5


def test_hits_after_the_top_three_still_merge_when_adjacent(conn):
    fill(conn, 40)
    hits = [hit(conn, seq) for seq in (2, 10, 20, 30, 31)]
    assert [len(source.chunk_ids) for source in expand_hits(conn, hits, top=3)] == [3, 3, 3, 2]


def test_a_merged_source_keeps_the_rank_of_its_best_hit(conn):
    fill(conn, 30)
    sources = expand_hits(conn, [hit(conn, 20), hit(conn, 5), hit(conn, 6)], top=0)
    assert sources[0].seq == 20 and sources[1].seq == 5


def test_chunks_from_two_documents_never_merge(conn):
    fill(conn, 3, doc_id=1)
    fill(conn, 3, doc_id=2)
    assert len(expand_hits(conn, [hit(conn, 1, doc_id=1), hit(conn, 1, doc_id=2)], top=3)) == 2


def test_the_overlap_sentence_is_read_once(conn):
    add(conn, 0, "First part. The overlap sentence is here.")
    add(conn, 1, "The overlap sentence is here. Second part.")
    source = expand_hits(conn, [hit(conn, 0)], top=3)[0]
    assert source.text.count("The overlap sentence") == 1
    assert source.text.endswith("Second part.")


def test_a_chunk_from_another_section_keeps_its_heading(conn):
    add(conn, 0)
    add(conn, 1, path='["Ch. 2"]', text="Ch. 2 ›\n\nNew section.")
    source = expand_hits(conn, [hit(conn, 0)], top=3)[0]
    assert "Ch. 2 ›\n\nNew section." in source.text


def test_pages_cover_the_whole_merged_source(conn):
    add(conn, 0, page=4)
    add(conn, 1, page=5)
    add(conn, 2, page=6)
    source = expand_hits(conn, [hit(conn, 1)], top=3)[0]
    assert (source.page_start, source.page_end) == (4, 6)


def test_the_focus_stays_on_the_hit_not_its_neighbours(conn):
    fill(conn, 3)
    source = expand_hits(conn, [hit(conn, 1)], top=3)[0]
    assert source.focus_block_ids == [101]
    assert source.block_ids == [100, 101, 102]


def test_flagged_formulas_stay_aligned_after_merging(conn):
    note = "[unverified formula, see page {}]\n$$ wrong{} $$"
    add(conn, 0, note.format(1, 1), meta={"unverified_formulas": [{"page": 1, "crop": "a", "text_layer": "first"}]})
    add(conn, 1, "Plain text in between.")
    add(conn, 2, note.format(3, 2), meta={"unverified_formulas": [{"page": 3, "crop": "b", "text_layer": "second"}]})
    source = expand_hits(conn, [hit(conn, 1)], top=3)[0]
    prompt = format_sources([source])
    assert prompt.index("first") < prompt.index("Plain text") < prompt.index("second")
    assert "wrong" not in prompt


def test_no_hits_no_sources(conn):
    assert expand_hits(conn, [], top=3) == []


def test_worst_case_prompt_and_answer_fit_the_context():
    settings = Settings()
    rules = 500
    assert (settings.search_source_tokens + rules + settings.chat_history_tokens
            + settings.followup_carry_tokens + settings.chat_num_predict) <= settings.chat_num_ctx


def test_six_of_the_largest_chunks_always_fit_the_source_budget():
    largest_chunk = 1137        # measured across the library
    settings = Settings()
    assert settings.search_top_k * largest_chunk <= settings.search_source_tokens


def test_the_chat_pipeline_widens_the_sources(tmp_path):
    settings = Settings(data_dir=tmp_path, library_dir=tmp_path)
    db = connect(settings.db_path)
    db.execute("INSERT INTO documents (id, path, sha256, title, kind) VALUES (1, 'a.pdf', 'x', 'A', 'pdf')")
    for seq in range(3):
        db.execute('INSERT INTO blocks (id, doc_id, "order", kind, page, bbox, text) '
                   "VALUES (?, 1, ?, 'paragraph', ?, '[1, 2, 3, 4]', 't')", (seq + 1, seq, seq + 5))
        chunk = db.execute(
            "INSERT INTO chunks (doc_id, kind, seq, heading_path, text, token_count, block_ids, page_start, "
            "page_end) VALUES (1, 'text', ?, '[]', ?, 5, ?, ?, ?)",
            (seq, f"Unique{seq} words", json.dumps([seq + 1]), seq + 5, seq + 5)).lastrowid
        vector = [0.0] * 1024
        vector[seq] = 1.0
        db.execute("INSERT INTO chunk_vec (chunk_id, embedding) VALUES (?, ?)", (chunk, serialize(vector, 1024)))
    db.commit()
    vector = [0.0] * 1024
    vector[1] = 1.0
    settings.search_top_k = 1
    [source] = find_sources(settings, "Unique1", None, lambda texts, _: [vector for _ in texts])
    assert (source.page_start, source.page_end, source.focus_page) == (5, 7, 6)
    assert [region["page"] for region in source.regions] == [5, 6, 7]


def cited(*groups):
    """Each group is a source's chunk ids, with its main chunk first."""
    return [{"chunk_id": group[0], "chunk_ids": sorted(group)} for group in groups]


def ids(hits):
    return [hit.chunk_ids for hit in hits]


def test_the_previous_answers_sources_are_carried_with_their_text(conn):
    fill(conn, 3)
    [hit_one] = carried_hits(conn, cited([2]), set(), 100)
    assert hit_one.chunk_ids == [2] and "Chunk 1." in hit_one.text
    assert (hit_one.doc_title, hit_one.page_start) == ("Book 1", 2)
    assert hit_one.focus_block_ids == [101]


def test_a_chunk_the_search_found_again_is_not_repeated(conn):
    fill(conn, 3)
    assert ids(carried_hits(conn, cited([1], [3]), {1}, 100)) == [[3]]


def test_adjacent_chunks_join_into_one_source(conn):
    fill(conn, 4)
    hits = carried_hits(conn, cited([2, 1, 3]), set(), 100)
    assert ids(hits) == [[1, 2, 3]]
    assert hits[0].text.count("Ch. 1 ›") == 1 and hits[0].focus_block_ids == [101]


def test_a_tight_budget_keeps_each_sources_main_chunk(conn):
    fill(conn, 6)
    hits = carried_hits(conn, cited([2, 1, 3], [5, 4, 6]), set(), 20)
    assert ids(hits) == [[2], [5]]


def test_nothing_is_carried_past_the_budget(conn):
    fill(conn, 3)
    assert carried_hits(conn, cited([1], [3]), set(), 5) == []


def test_sources_keep_the_order_the_answer_cited_them(conn):
    fill(conn, 5)
    assert ids(carried_hits(conn, cited([5], [1]), set(), 100)) == [[5], [1]]


def test_a_chunk_outside_the_current_scope_stays_out(conn):
    fill(conn, 2)
    other = add(conn, 0, doc_id=2)
    assert ids(carried_hits(conn, cited([1], [other]), set(), 100, doc_ids=[2])) == [[other]]


def test_a_chunk_gone_after_re_indexing_is_skipped(conn):
    fill(conn, 2)
    assert ids(carried_hits(conn, cited([99], [1]), set(), 100)) == [[1]]


def test_a_cited_chunk_whose_file_is_gone_is_not_carried(conn):
    fill(conn, 2)
    other = add(conn, 0, doc_id=2)
    conn.execute("UPDATE documents SET status = 'missing' WHERE id = 1")
    assert ids(carried_hits(conn, cited([1], [other]), set(), 100)) == [[other]]


def test_flagged_formulas_come_along_with_a_carried_chunk(conn):
    add(conn, 0, meta={"unverified_formulas": [{"page": 1, "crop": "c.png", "text_layer": "x"}]})
    assert carried_hits(conn, cited([1]), set(), 100)[0].meta["unverified_formulas"][0]["crop"] == "c.png"


def seqs(hits):
    return [sorted(row - 1 for row in hit.chunk_ids) for hit in hits]


def test_the_radius_reaches_further_each_way(conn):
    fill(conn, 9)
    assert seqs(expand_hits(conn, [hit(conn, 4)], 1, radius=2)) == [[2, 3, 4, 5, 6]]


def test_only_the_top_hits_get_the_wider_window(conn):
    fill(conn, 20)
    hits = expand_hits(conn, [hit(conn, 5), hit(conn, 15)], 1, radius=2)
    assert seqs(hits) == [[3, 4, 5, 6, 7], [15]]


def test_the_window_stops_at_the_edge_of_the_document(conn):
    fill(conn, 3)
    assert seqs(expand_hits(conn, [hit(conn, 0)], 1, radius=2)) == [[0, 1, 2]]


def test_nearest_neighbours_fill_the_budget_first(conn):
    fill(conn, 9)
    assert seqs(expand_hits(conn, [hit(conn, 4)], 1, radius=2, budget=30)) == [[3, 4, 5]]


def test_every_hit_is_read_even_past_the_budget(conn):
    fill(conn, 9)
    hits = expand_hits(conn, [hit(conn, 1), hit(conn, 7)], 2, radius=2, budget=5)
    assert seqs(hits) == [[1], [7]]


def test_a_chunk_in_two_windows_is_paid_for_once(conn):
    fill(conn, 9)
    hits = expand_hits(conn, [hit(conn, 3), hit(conn, 4)], 2, radius=1, budget=40)
    assert seqs(hits) == [[2, 3, 4, 5]]


# --- Figure and summary hits (Phase 5b) ------------------------------------------

from dataclasses import replace  # noqa: E402

from app.retrieval.search import with_slots  # noqa: E402


def _kind(conn, row_id, kind):
    conn.execute("UPDATE chunks SET kind = ? WHERE id = ?", (kind, row_id))


def test_text_keeps_its_places_and_slotted_kinds_come_on_top():
    hits = [Hit(chunk_id=i, doc_id=1, doc_path="", doc_title="", heading_path=[], text="", page_start=1,
                page_end=1, block_ids=[], meta={}, score=0, kind=kind)
            for i, kind in enumerate(["figure", "text", "summary", "figure", "figure", "text", "summary",
                                      "summary", "text", "text", "text", "text", "figure"])]
    chosen = with_slots(hits, 6, {"figure": 2, "summary": 2})
    assert [h.chunk_id for h in chosen if h.kind == "text"] == [1, 5, 8, 9, 10, 11]
    assert [h.chunk_id for h in chosen if h.kind == "figure"] == [0, 3]     # from the top 6, at most 2
    assert [h.chunk_id for h in chosen if h.kind == "summary"] == [2]       # 6 and 7 rank below the top 6
    assert [h.chunk_id for h in chosen] == sorted(h.chunk_id for h in chosen)


def test_a_figure_hit_brings_no_neighbours_and_takes_no_text_hits_window(conn):
    fill(conn, 30)
    figure = hit(conn, 10)
    _kind(conn, figure.chunk_id, "figure")
    figure = replace(figure, kind="figure")
    sources = expand_hits(conn, [figure, hit(conn, 2), hit(conn, 20), hit(conn, 25)], top=3)
    by_hit = {s.chunk_id: s for s in sources}
    assert len(by_hit[figure.chunk_id].chunk_ids) == 1
    assert len(by_hit[hit(conn, 25).chunk_id].chunk_ids) == 3      # the third text hit has its window


def _summary(conn, text, children=(), tokens=10):
    return conn.execute(
        "INSERT INTO chunks (doc_id, kind, level, seq, heading_path, text, token_count, block_ids, page_start, "
        "page_end, meta) VALUES (1, 'summary', 1, ?, '[]', ?, ?, '[]', 1, 9, ?)",
        (100_000_000 + conn.execute("SELECT count(*) FROM chunks").fetchone()[0], text, tokens,
         json.dumps({"children": list(children)})),
    ).lastrowid


def _summary_hit(conn, row_id):
    row = conn.execute("SELECT * FROM chunks WHERE id = ?", (row_id,)).fetchone()
    return Hit(chunk_id=row_id, doc_id=1, doc_path="1.pdf", doc_title="Book 1", heading_path=[], text=row["text"],
               page_start=1, page_end=9, block_ids=[], meta=json.loads(row["meta"]), score=1.0, seq=row["seq"],
               chunk_ids=[row_id], kind="summary")


def test_a_summary_hit_comes_with_its_parts_summaries_up_to_the_cap(conn):
    parts = [_summary(conn, f"Part {i}.", tokens=400) for i in range(3)]
    chapter = _summary(conn, "Chapter.", parts)
    [source] = expand_hits(conn, [_summary_hit(conn, chapter)], top=3, summary_tokens=1000)
    assert source.chunk_ids == [chapter, parts[0], parts[1]]
    assert source.text == "Chapter.\n\nPart 0.\n\nPart 1."


def test_a_summary_stands_alone_and_comes_out_of_the_budget_first(conn):
    fill(conn, 10)
    from app.ingest.chunker import estimate_tokens
    chapter = _summary(conn, "Chapter. " * 20)
    budget = estimate_tokens("Chapter. " * 20) + 10 + 2 * 10 + 5    # the summary, the hit, two neighbours
    sources = expand_hits(conn, [_summary_hit(conn, chapter), hit(conn, 5)], top=3, radius=2, budget=budget)
    assert [s.kind for s in sources] == ["summary", "text"]
    assert len(sources[1].chunk_ids) == 3      # the hit and its nearest neighbours; the next two don't fit


def test_parts_summaries_only_fill_what_the_text_neighbours_leave(conn):
    fill(conn, 10)
    parts = [_summary(conn, f"Part {i}.", tokens=10) for i in range(3)]
    chapter = _summary(conn, "Chapter.", parts)
    from app.ingest.chunker import estimate_tokens
    budget = estimate_tokens("Chapter.") + 5 * 10 + 15     # the summary, the hit, 4 neighbours, one part
    sources = expand_hits(conn, [_summary_hit(conn, chapter), hit(conn, 5)], top=3, radius=2, budget=budget,
                          summary_tokens=1000)
    assert len(sources[1].chunk_ids) == 5       # the text keeps its whole window
    assert sources[0].chunk_ids == [chapter, parts[0]]


def test_a_carried_chunk_from_a_hidden_collection_stays_out_of_whole_library_search(conn):
    fill(conn, 2)
    other = add(conn, 0, doc_id=2)
    conn.execute("UPDATE documents SET collection = 'test' WHERE id = 1")
    assert ids(carried_hits(conn, cited([1], [other]), set(), 100, hidden=("test",))) == [[other]]
    assert ids(carried_hits(conn, cited([1]), set(), 100, doc_ids=[1], hidden=("test",))) == [[1]]


def test_a_neighbour_past_one_too_large_to_read_isnt_read_alone(conn):
    """Review R2-C3: a big table next to the hit didn't fit, the chunk past it did, and the
    run it made had no hit: min() of nothing failed the whole question."""
    fill(conn, 9)
    conn.execute("UPDATE chunks SET token_count = 500 WHERE seq = 3")
    assert seqs(expand_hits(conn, [hit(conn, 4)], 1, radius=2, budget=40)) == [[4, 5, 6]]


def test_a_neighbour_past_a_missing_chunk_isnt_read_alone(conn):
    fill(conn, 9)
    conn.execute("DELETE FROM chunks WHERE seq = 3")
    assert seqs(expand_hits(conn, [hit(conn, 4)], 1, radius=2)) == [[4, 5, 6]]
