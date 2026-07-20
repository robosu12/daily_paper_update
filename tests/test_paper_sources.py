import datetime
import unittest
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

        with patch.object(daily_arxiv, "SEMANTIC_SCHOLAR_API_KEY", "test-key"):
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
            "2026-06-01:",
        )
        self.assertIn(" | ", request.kwargs["params"]["query"])


if __name__ == "__main__":
    unittest.main()
