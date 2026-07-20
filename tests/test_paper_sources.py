import datetime
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import daily_arxiv


class PaperSourceTests(unittest.TestCase):
    def make_paper(self, source, source_id, title, published_date, **kwargs):
        return daily_arxiv.Paper(
            source=source,
            source_id=source_id,
            title=title,
            authors=["Alice"],
            abstract="A complete abstract.",
            published_date=published_date,
            paper_url=f"https://example.org/{source_id}",
            **kwargs,
        )

    def test_deduplicate_papers_keeps_source_priority(self):
        published = datetime.date(2026, 7, 1)
        arxiv_paper = daily_arxiv.Paper(
            source="arxiv",
            source_id="2607.00001",
            title="A SLAM Paper",
            authors=["Alice"],
            abstract="Abstract",
            published_date=published,
            paper_url="https://arxiv.org/pdf/2607.00001",
            arxiv_id="2607.00001",
        )
        semantic_paper = daily_arxiv.Paper(
            source="semantic_scholar",
            source_id="s2-id",
            title="A-SLAM Paper",
            authors=["Alice"],
            abstract="Abstract",
            published_date=published,
            paper_url="https://www.semanticscholar.org/paper/s2-id",
            arxiv_id="2607.00001",
        )

        result = daily_arxiv.deduplicate_papers([arxiv_paper, semantic_paper])

        self.assertEqual(result, [arxiv_paper])

    def test_sort_papers_uses_entry_date_for_all_sources(self):
        entries = {
            "openreview:old": "|2026-06-01|Old|A|[OpenReview](url)|无|Summary|\n",
            "2607.00001": "|2026-07-01|New|A|[2607.00001](url)|无|Summary|\n",
        }

        result = daily_arxiv.sort_papers(entries)

        self.assertEqual(list(result), ["2607.00001", "openreview:old"])

    def test_render_summary_html_truncates_long_text(self):
        summary = "A" * 700

        rendered = daily_arxiv.render_summary_html(summary)

        self.assertEqual(rendered, f"<strong>摘要：</strong> {'A' * 600}")
        self.assertNotIn("<details>", rendered)

    def test_json_to_md_renders_link_and_summary_in_same_cell(self):
        long_summary = "Long summary sentence. " * 30
        data = {
            "SLAM": {
                "2607.00001": (
                    "|2026-07-19|Paper title|Alice|[Paper](url)|无|"
                    f"{long_summary}|\n"
                )
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "papers.json"
            markdown_path = Path(temp_dir) / "README.md"
            json_path.write_text(json.dumps(data), encoding="utf-8")
            with patch(
                "daily_arxiv.current_date",
                return_value=datetime.date(2026, 7, 20),
            ):
                daily_arxiv.json_to_md(str(json_path), str(markdown_path))
            rendered = markdown_path.read_text(encoding="utf-8")

        self.assertIn("<th>论文与摘要</th></tr>", rendered)
        self.assertIn("<td>[Paper](url)<br><strong>摘要：</strong>", rendered)
        self.assertNotIn("colspan", rendered)
        self.assertNotIn("<details><summary><strong>摘要", rendered)
        rendered_summary = rendered.split("<strong>摘要：</strong> ", 1)[1].split(
            "</td>", 1
        )[0]
        self.assertLessEqual(len(rendered_summary), daily_arxiv.SUMMARY_MAX_CHARS)

    def test_filter_old_papers_removes_future_dates(self):
        entries = {
            "boundary": "|2026-05-20|Boundary|A|[Paper](url)|无|Summary|\n",
            "current": "|2026-07-19|Current|A|[Paper](url)|无|Summary|\n",
            "future": "|2026-08-01|Future|A|[Paper](url)|无|Summary|\n",
            "old": "|2026-05-19|Old|A|[Paper](url)|无|Summary|\n",
        }

        with patch(
            "daily_arxiv.current_date",
            return_value=datetime.date(2026, 7, 20),
        ):
            result = daily_arxiv.filter_old_papers(entries)

        self.assertEqual(list(result), ["boundary", "current"])

    def test_retention_start_date_handles_month_end(self):
        self.assertEqual(
            daily_arxiv.retention_start_date(datetime.date(2026, 4, 30)),
            datetime.date(2026, 2, 28),
        )

    def test_update_json_file_deletes_expired_entries(self):
        data = {
            "SLAM": {
                "expired": "|2026-05-19|Old|A|[Paper](url)|无|Summary|\n",
                "retained": "|2026-05-20|New|A|[Paper](url)|无|Summary|\n",
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "papers.json"
            json_path.write_text(json.dumps(data), encoding="utf-8")
            with patch(
                "daily_arxiv.current_date",
                return_value=datetime.date(2026, 7, 20),
            ):
                daily_arxiv.update_json_file(str(json_path), [])
            updated = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertEqual(list(updated["SLAM"]), ["retained"])

    @patch("daily_arxiv.fetch_arxiv_results")
    def test_fetch_arxiv_papers_uses_initial_publication_date(self, mock_fetch):
        result = Mock()
        result.get_short_id.return_value = "2607.00001v2"
        result.title = "Revised SLAM Paper"
        result.authors = ["Alice"]
        result.summary = "Abstract"
        result.published = datetime.datetime(2026, 7, 1)
        result.updated = datetime.datetime(2026, 7, 19)
        result.doi = None
        mock_fetch.return_value = [result]

        papers = daily_arxiv.fetch_arxiv_papers("SLAM", 1)

        self.assertEqual(papers[0].published_date, datetime.date(2026, 7, 1))

    def test_get_daily_papers_serializes_each_source(self):
        arxiv_paper = self.make_paper(
            "arxiv",
            "2607.00001",
            "Arxiv Paper",
            datetime.date(2026, 7, 1),
            arxiv_id="2607.00001",
        )
        openreview_paper = self.make_paper(
            "openreview",
            "openreview-id",
            "OpenReview Paper",
            datetime.date(2026, 7, 2),
        )
        semantic_paper = self.make_paper(
            "semantic_scholar",
            "semantic-id",
            "Semantic Paper",
            datetime.date(2026, 7, 3),
        )

        with (
            patch("daily_arxiv.fetch_arxiv_papers", return_value=[arxiv_paper]),
            patch(
                "daily_arxiv.fetch_openreview_papers",
                return_value=[openreview_paper],
            ),
            patch(
                "daily_arxiv.fetch_semantic_scholar_papers",
                return_value=[semantic_paper],
            ),
            patch("daily_arxiv.get_paper_summary", return_value="Summary"),
            patch("daily_arxiv.get_official_code_link", return_value=None),
            patch(
                "daily_arxiv.current_date",
                return_value=datetime.date(2026, 7, 20),
            ),
        ):
            data, _ = daily_arxiv.get_daily_papers(
                "SLAM",
                ["SLAM"],
                max_results=10,
            )

        entries = data["SLAM"]
        self.assertIn("[2607.00001]", entries["2607.00001"])
        self.assertIn("[OpenReview]", entries["openreview:openreview-id"])
        self.assertIn(
            "[Semantic Scholar]",
            entries["semantic_scholar:semantic-id"],
        )

    @patch("daily_arxiv.requests.get")
    def test_fetch_openreview_papers_normalizes_public_note(self, mock_get):
        response = Mock(status_code=200)
        response.json.return_value = {
            "notes": [{
                "id": "note-id",
                "forum": "forum-id",
                "pdate": 1782864000000,
                "content": {
                    "title": {"value": "Recent Visual SLAM"},
                    "abstract": {"value": "A recent abstract."},
                    "authors": {"value": ["Alice", "Bob"]},
                    "venue": {"value": "Robotics Conference 2026"},
                    "pdf": {"value": "https://arxiv.org/pdf/2607.00001"},
                },
            }]
        }
        mock_get.return_value = response

        with patch(
            "daily_arxiv.current_date",
            return_value=datetime.date(2026, 7, 20),
        ):
            papers = daily_arxiv.fetch_openreview_papers("Visual SLAM", 10)

        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0].source, "openreview")
        self.assertEqual(papers[0].arxiv_id, "2607.00001")
        self.assertEqual(
            papers[0].paper_url,
            "https://openreview.net/forum?id=forum-id",
        )

    @patch("daily_arxiv.requests.get")
    def test_fetch_semantic_scholar_papers_uses_key_and_filters(self, mock_get):
        response = Mock(status_code=200, headers={})
        response.json.return_value = {
            "data": [{
                "paperId": "s2-id",
                "title": "Recent LiDAR SLAM",
                "abstract": "A recent abstract.",
                "authors": [{"name": "Alice"}],
                "publicationDate": "2026-07-01",
                "year": 2026,
                "url": "https://www.semanticscholar.org/paper/s2-id",
                "externalIds": {
                    "ArXiv": "2607.00002v1",
                    "DOI": "10.1000/example",
                },
                "openAccessPdf": {
                    "url": "https://example.org/paper.pdf",
                },
            }]
        }
        mock_get.return_value = response

        with (
            patch.object(daily_arxiv, "SEMANTIC_SCHOLAR_API_KEY", "test-key"),
            patch(
                "daily_arxiv.current_date",
                return_value=datetime.date(2026, 7, 20),
            ),
        ):
            daily_arxiv._semantic_scholar_disabled = False
            papers = daily_arxiv.fetch_semantic_scholar_papers(
                ["SLAM", "LiDAR Odometry"],
                10,
            )

        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0].storage_key, "2607.00002")
        request = mock_get.call_args
        self.assertEqual(request.kwargs["headers"]["x-api-key"], "test-key")
        self.assertEqual(
            request.kwargs["params"]["publicationDateOrYear"],
            "2026-05-20:2026-07-20",
        )
        self.assertIn(" | ", request.kwargs["params"]["query"])

    @patch("daily_arxiv.requests.get")
    def test_fetch_semantic_scholar_papers_rejects_future_date(self, mock_get):
        response = Mock(status_code=200, headers={})
        response.json.return_value = {
            "data": [{
                "paperId": "future-id",
                "title": "Future Issue Paper",
                "abstract": "Already indexed with a future issue date.",
                "authors": [{"name": "Alice"}],
                "publicationDate": "2026-08-01",
                "year": 2026,
                "url": "https://www.semanticscholar.org/paper/future-id",
                "externalIds": {},
                "openAccessPdf": None,
            }]
        }
        mock_get.return_value = response

        with (
            patch.object(daily_arxiv, "SEMANTIC_SCHOLAR_API_KEY", "test-key"),
            patch(
                "daily_arxiv.current_date",
                return_value=datetime.date(2026, 7, 20),
            ),
        ):
            daily_arxiv._semantic_scholar_disabled = False
            papers = daily_arxiv.fetch_semantic_scholar_papers(["SLAM"], 10)

        self.assertEqual(papers, [])


if __name__ == "__main__":
    unittest.main()
