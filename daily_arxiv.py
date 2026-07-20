import os
import re
import json
import arxiv
import yaml
import logging
import argparse
import datetime
import requests
import time
import html
import feedparser
import random
from dataclasses import dataclass

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

base_url = "https://arxiv.paperswithcode.com/api/v0/papers/"
github_url = "https://api.github.com/search/repositories"
arxiv_url = "https://arxiv.org/"
openreview_search_url = "https://api2.openreview.net/notes/search"
semantic_scholar_search_url = (
    "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
)
http_headers = {
    "User-Agent": (
        "daily_paper_update/1.0 "
        "(+https://github.com/robosu12/daily_paper_update)"
    )
}

# DeepSeek API配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

# 全局过滤日期 - 修改这里调整过滤条件
MIN_DATE = datetime.date(2026, 6, 1)
SUMMARY_PREVIEW_CHARS = 360

_semantic_scholar_disabled = False


@dataclass
class Paper:
    source: str
    source_id: str
    title: str
    authors: list[str]
    abstract: str
    published_date: datetime.date
    paper_url: str
    arxiv_id: str = ""
    doi: str = ""

    @property
    def storage_key(self) -> str:
        if self.arxiv_id:
            return self.arxiv_id.split("v")[0]
        if self.doi:
            return f"doi:{self.doi.lower()}"
        return f"{self.source}:{self.source_id}"

    @property
    def link_label(self) -> str:
        if self.source == "arxiv":
            return self.arxiv_id or self.source_id
        if self.source == "openreview":
            return "OpenReview"
        return "Semantic Scholar"


def normalize_title(title: str) -> str:
    """Normalize titles for cross-source duplicate detection."""
    return re.sub(r"\W+", "", title.casefold())


def paper_identities(paper: Paper) -> set[str]:
    identities = {f"title:{normalize_title(paper.title)}"}
    if paper.arxiv_id:
        identities.add(f"arxiv:{paper.arxiv_id.split('v')[0].lower()}")
    if paper.doi:
        identities.add(f"doi:{paper.doi.lower()}")
    return identities


def deduplicate_papers(papers: list[Paper]) -> list[Paper]:
    """Keep source priority while removing duplicate records."""
    unique = []
    seen = set()
    for paper in papers:
        identities = paper_identities(paper)
        if seen.intersection(identities):
            logging.info(f"跳过跨来源重复论文: {paper.title} ({paper.source})")
            continue
        seen.update(identities)
        unique.append(paper)
    return unique


def extract_arxiv_id(*values) -> str:
    for value in values:
        if not value:
            continue
        match = re.search(
            r"(?:arxiv(?:\.org/(?:abs|pdf)/|:))([0-9]{4}\.[0-9]{4,5})(?:v\d+)?",
            str(value),
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1)
    return ""


def openreview_value(content: dict, key: str, default=None):
    value = content.get(key, default)
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def timestamp_to_date(timestamp) -> datetime.date | None:
    if not timestamp:
        return None
    try:
        return datetime.datetime.fromtimestamp(
            int(timestamp) / 1000,
            tz=datetime.timezone.utc,
        ).date()
    except (TypeError, ValueError, OSError):
        return None


def parse_iso_date(date_str: str | None, year=None) -> datetime.date | None:
    if date_str:
        try:
            return datetime.date.fromisoformat(date_str)
        except ValueError:
            pass
    try:
        return datetime.date(int(year), 1, 1) if year else None
    except (TypeError, ValueError):
        return None


def sanitize_entry_text(value: str) -> str:
    return str(value).replace("|", "｜")


def normalize_summary_text(summary: str) -> str:
    return re.sub(r"\s+", " ", str(summary)).strip()


def summary_preview(summary: str, max_chars=SUMMARY_PREVIEW_CHARS) -> tuple[str, bool]:
    normalized = normalize_summary_text(summary)
    if len(normalized) <= max_chars:
        return normalized, False

    preview = normalized[:max_chars].rstrip()
    last_space = preview.rfind(" ")
    if last_space >= int(max_chars * 0.75):
        preview = preview[:last_space]
    return preview.rstrip(" ,;:"), True


def render_summary_html(summary: str) -> str:
    normalized = normalize_summary_text(summary)
    preview, truncated = summary_preview(normalized)
    preview_html = html.escape(preview)
    if not truncated:
        return f"<strong>摘要：</strong> {preview_html}"

    return (
        "<details>"
        f"<summary><strong>摘要：</strong> {preview_html}...</summary>"
        f"<div>{html.escape(normalized)}</div>"
        "</details>"
    )


def current_date() -> datetime.date:
    return datetime.date.today()


def is_date_in_range(date_value: datetime.date) -> bool:
    return MIN_DATE <= date_value <= current_date()

def load_config(config_file: str) -> dict:
    '''
    config_file: 配置文件路径
    return: 配置字典
    '''
    def pretty_filters(**config) -> dict:
        keywords = {}
        for k, v in config['keywords'].items():
            filters = v['filters']
            formatted = ' OR '.join(f'"{f}"' if ' ' in f else f for f in filters)
            keywords[k] = formatted
        return keywords
    
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config['kv'] = pretty_filters(**config)
        logging.info(f'加载配置: {config}')
    return config

def filter_old_papers(papers: dict) -> dict:
    """
    过滤掉旧论文的核心函数
    """
    filtered = {}
    count = 0
    for paper_id, paper_entry in papers.items():
        try:
            # 提取论文日期
            parts = paper_entry.split('|')
            if len(parts) > 1:
                date_str = parts[1].strip()
                
                if is_date_above_min(date_str):
                    filtered[paper_id] = paper_entry
                else:
                    count += 1
            else:
                # 格式错误时保留
                filtered[paper_id] = paper_entry
        except ValueError:
            # 日期解析错误时保留
            filtered[paper_id] = paper_entry
    
    if count > 0:
        logging.info(
            f"过滤掉 {count} 篇日期不在 {MIN_DATE} 至 {current_date()} 范围内的论文"
        )
    
    return filtered

def get_authors(authors, first_author=False):
    """优化作者显示格式"""
    if not authors:
        return "未知作者"
    if not first_author:
        return ", ".join(str(author) for author in authors)
    elif len(authors) > 1:
        return f"{str(authors[0])}等"
    return str(authors[0])

def sort_papers(papers):
    """按日期排序论文"""
    return dict(sorted(
        papers.items(),
        key=lambda item: item[1].split("|")[1],
        reverse=True,
    ))

def get_official_code_link(paper_id: str, title: str, authors: list) -> str:
    """获取论文官方开源代码链接（带多重验证）"""
    # 1. 优先使用paperswithcode API获取官方链接
    if paper_id:
        try:
            code_response = requests.get(base_url + paper_id, timeout=10)
            if code_response.status_code == 200:
                data = code_response.json()
                if data.get("official") and data["official"].get("url"):
                    return data["official"]["url"]
        except Exception:
            pass
    
    # 2. 从作者名构建查询（修复last属性不存在的问题）
    author_query = ""
    if authors:
        try:
            # 尝试从全名提取姓氏（西方姓名规范）
            full_name = str(authors[0])
            if full_name:
                # 分割全名并取最后一部分作为姓氏
                author_query = full_name.split()[-1]
        except Exception as e:
            logging.warning(f"作者名解析失败: {str(e)}")
    
    # 3. 构建GitHub搜索查询
    query = f"{title} {author_query} arxiv in:name,description"
    params = {"q": query, "sort": "stars", "order": "desc"}
    
    try:
        response = requests.get(github_url, params=params, timeout=15)
        if response.status_code == 200:
            repos = response.json().get('items', [])
            
            # 验证仓库是否确实包含论文相关代码
            for repo in repos:
                repo_description = (repo.get('description') or '').lower()
                repo_name = (repo.get('name') or '').lower()
                
                # 验证关键词匹配
                if ('arxiv' in repo_description or
                    (paper_id and paper_id.split('v')[0] in repo_description) or
                    any(keyword in repo_description for keyword in ['paper', 'implementation'])):
                    return repo['html_url']
    except Exception:
        pass
    
    return None

def get_paper_summary(title: str, abstract: str) -> str:
    """确保摘要永不空白的生成函数"""
    MAX_RETRIES = 3
    WAIT_TIMES = [1, 2, 3]  # 重试等待时间
    
    # 1. 检查API密钥是否有效
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your_api_key_here":
        logging.warning("DeepSeek API密钥未配置")
        return generate_fallback_summary(title, abstract)
    
    # 2. 构建更有效的提示词
    prompt = (
        "用5-6句话总结以下论文的核心贡献和创新点:\n"
        f"标题: {title}\n"
        f"摘要: {abstract[:2000]}\n"
        "要求:\n"
        "- 用中文回复\n"
        "- 不使用Markdown格式\n"
        "- 不超过400字\n"
        "- 创新点用'◆'符号开头\n" 
        "- 每个创新点单开一行\n"
    )
    
    # 3. 带重试机制的API请求
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                DEEPSEEK_API_URL,
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 500,
                    "stream": False
                },
                timeout=30
            )
            
            # 4. 检查响应状态
            if response.status_code != 200:
                logging.warning(f"DeepSeek API响应错误: {response.status_code}")
                continue
                
            # 5. 解析API响应
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                logging.warning("DeepSeek API返回空内容")
                continue
                
            content = choices[0].get("message", {}).get("content", "").strip()
            if content:
                # 6. 清理响应内容
                content = re.sub(r"\*\*|\#\#\#|\`", "", content)
                return content
                
        except requests.Timeout:
            logging.warning(f"DeepSeek API超时 (尝试 {attempt+1}/{MAX_RETRIES})")
            time.sleep(WAIT_TIMES[attempt])
        except Exception as e:
            logging.error(f"DeepSeek API错误: {type(e).__name__}: {str(e)}")
    
    # 7. 所有重试失败后生成备用摘要
    return generate_fallback_summary(title, abstract)

def generate_fallback_summary(title: str, abstract: str) -> str:
    """智能生成备用摘要"""
    # 1. 尝试从摘要中提取前3句
    sentences = re.split(r'(?<=[.!?])\s+', abstract)
    if len(sentences) >= 3:
        return "◆ " + "\n◆ ".join(sentences[:3])
    elif sentences:
        return "◆ " + sentences[0]
    
    # 2. 如果摘要为空，根据标题生成示例摘要
    topics = ["视觉定位", "三维重建", "特征匹配", "场景理解", "神经网络"]
    techniques = ["深度学习", "卷积神经网络", "自监督学习", "特征金字塔"]
    contributions = ["提高准确率", "降低计算成本", "增强鲁棒性", "解决领域难题"]
    
    return (
        f"◆ 提出了一种新的{random.choice(topics)}方法\n"
        f"◆ 通过{random.choice(techniques)}技术创新\n"
        f"◆ 实现{random.choice(contributions)}\n"
        f"◆ 在多个数据集上验证了有效性"
    )

def fetch_arxiv_results(query, max_results=10):
    """修复参数错误并增强网络稳定性"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            client = arxiv.Client(
                page_size=max_results,
                delay_seconds=3,
                num_retries=2,
            )
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            return list(client.results(search))
        except Exception as e:
            logging.warning(f"arXiv API请求失败 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                logging.info(f"等待 {wait_time:.1f} 秒后重试...")
                time.sleep(wait_time)
    
    # 如果所有重试都失败，尝试直接API调用
    logging.warning("arxiv.py库请求失败，尝试直接调用arXiv API")
    try:
        url = "https://arxiv.org.cn/api/query"  # 国内镜像站点
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        response = requests.get(url, params=params, timeout=15)  # 关键超时设置
        response.raise_for_status()
        feed = feedparser.parse(response.content)
        
        results = []
        for entry in feed.entries:
            # 解析arXiv返回的Atom格式数据
            result = arxiv.Result(
                entry_id=entry.id,
                updated=datetime.datetime.strptime(entry.updated, "%Y-%m-%dT%H:%M:%SZ"),
                published=datetime.datetime.strptime(entry.published, "%Y-%m-%dT%H:%M:%SZ"),
                title=entry.title,
                authors=[arxiv.Result.Author(name=a.name) for a in entry.authors],
                summary=entry.summary,
                comment=entry.get("arxiv_comment", ""),
                journal_ref=entry.get("arxiv_journal_ref", ""),
                doi=entry.get("arxiv_doi", ""),
                primary_category=entry.get("arxiv_primary_category", {}).get("term", ""),
                categories=[t.term for t in entry.tags if t.scheme == "http://arxiv.org/schemas/atom"],
                links=[
                    arxiv.Result.Link(href=l.href, title=l.title, rel=l.rel)
                    for l in entry.links
                ],
            )
            results.append(result)
        return results
    except (requests.Timeout, ConnectionResetError) as e:
        logging.error(f"直接API请求失败: {type(e).__name__}, 建议检查网络或重试")
    except Exception as e:
        logging.error(f"直接arXiv API请求失败: {type(e).__name__}: {str(e)}")
    return []


def fetch_arxiv_papers(query: str, max_results=10) -> list[Paper]:
    papers = []
    for result in fetch_arxiv_results(query, max_results):
        paper_id = result.get_short_id().split("v")[0]
        papers.append(Paper(
            source="arxiv",
            source_id=paper_id,
            title=result.title.strip(),
            authors=[str(author) for author in result.authors],
            abstract=result.summary.replace("\n", " ").strip(),
            published_date=result.published.date(),
            paper_url=f"{arxiv_url}pdf/{paper_id}",
            arxiv_id=paper_id,
            doi=result.doi or "",
        ))
    return papers


def fetch_openreview_papers(
        query: str,
        max_results=10,
        search_limit=50) -> list[Paper]:
    """Search public OpenReview paper notes and normalize their metadata."""
    try:
        response = requests.get(
            openreview_search_url,
            params={"term": query, "limit": min(max(search_limit, max_results), 100)},
            headers=http_headers,
            timeout=30,
        )
        if response.status_code in (403, 429):
            logging.warning(
                f"OpenReview API暂不可用 ({response.status_code})，跳过该来源"
            )
            return []
        response.raise_for_status()
    except requests.RequestException as e:
        logging.warning(f"OpenReview API请求失败: {e}")
        return []

    try:
        notes = response.json().get("notes", [])
    except ValueError:
        logging.warning("OpenReview API返回了无效JSON，跳过该来源")
        return []

    papers = []
    for note in notes:
        content = note.get("content", {})
        if not openreview_value(content, "title"):
            content = note.get("forumContent", {})

        title = str(openreview_value(content, "title", "") or "").strip()
        abstract = str(openreview_value(content, "abstract", "") or "").strip()
        authors = openreview_value(content, "authors", []) or []
        if not isinstance(authors, list):
            authors = [authors]
        venue = str(openreview_value(content, "venue", "") or "")
        if not title or not abstract:
            continue
        if any(word in venue.casefold() for word in ("withdrawn", "desk rejected")):
            continue

        published_date = timestamp_to_date(
            note.get("pdate") or note.get("cdate") or note.get("tcdate")
        )
        if not published_date or not is_date_in_range(published_date):
            continue

        forum_id = note.get("forum") or note.get("id")
        source_id = note.get("id") or forum_id
        if not source_id or not forum_id:
            continue

        pdf = openreview_value(content, "pdf", "") or ""
        html_url = openreview_value(content, "html", "") or ""
        bibtex = openreview_value(content, "_bibtex", "") or ""
        arxiv_id = extract_arxiv_id(pdf, html_url, bibtex)
        doi = str(openreview_value(content, "doi", "") or "")

        papers.append(Paper(
            source="openreview",
            source_id=str(source_id),
            title=title,
            authors=[str(author) for author in authors],
            abstract=abstract.replace("\n", " "),
            published_date=published_date,
            paper_url=f"https://openreview.net/forum?id={forum_id}",
            arxiv_id=arxiv_id,
            doi=doi,
        ))

    papers.sort(key=lambda paper: paper.published_date, reverse=True)
    return papers[:max_results]


def build_semantic_scholar_query(filters: list[str]) -> str:
    terms = []
    for item in filters:
        clean_item = str(item).replace('"', "").strip()
        terms.append(f'"{clean_item}"' if " " in clean_item else clean_item)
    return " | ".join(terms)


def fetch_semantic_scholar_papers(
        filters: list[str],
        max_results=10) -> list[Paper]:
    """Fetch recent Semantic Scholar papers without blocking other sources."""
    global _semantic_scholar_disabled
    if _semantic_scholar_disabled:
        return []

    headers = dict(http_headers)
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

    params = {
        "query": build_semantic_scholar_query(filters),
        "fields": (
            "paperId,title,abstract,authors,publicationDate,year,url,"
            "externalIds,openAccessPdf,venue"
        ),
        "sort": "publicationDate:desc",
        "publicationDateOrYear": (
            f"{MIN_DATE.isoformat()}:{current_date().isoformat()}"
        ),
        "fieldsOfStudy": "Computer Science,Engineering",
    }

    response = None
    for attempt in range(3):
        try:
            response = requests.get(
                semantic_scholar_search_url,
                params=params,
                headers=headers,
                timeout=30,
            )
            if response.status_code in (401, 403):
                logging.warning(
                    "Semantic Scholar API Key无效或无权限，跳过该来源"
                )
                _semantic_scholar_disabled = True
                return []
            if response.status_code != 429:
                response.raise_for_status()
                break
            if not SEMANTIC_SCHOLAR_API_KEY:
                logging.warning(
                    "Semantic Scholar公共接口已限流；配置"
                    "SEMANTIC_SCHOLAR_API_KEY后将自动启用该来源"
                )
                _semantic_scholar_disabled = True
                return []
            if attempt < 2:
                retry_after = response.headers.get("Retry-After")
                wait_time = int(retry_after) if retry_after and retry_after.isdigit() else 2 ** attempt
                time.sleep(min(wait_time, 10))
        except requests.RequestException as e:
            logging.warning(f"Semantic Scholar API请求失败: {e}")
            return []
    else:
        logging.warning("Semantic Scholar API持续限流，跳过该来源")
        return []

    if response is None:
        return []

    try:
        items = response.json().get("data", [])
    except ValueError:
        logging.warning("Semantic Scholar API返回了无效JSON，跳过该来源")
        return []

    papers = []
    for item in items:
        title = str(item.get("title") or "").strip()
        abstract = str(item.get("abstract") or "").strip()
        published_date = parse_iso_date(
            item.get("publicationDate"),
            item.get("year"),
        )
        if not title or not abstract or not published_date:
            continue
        if not is_date_in_range(published_date):
            continue

        external_ids = item.get("externalIds") or {}
        arxiv_id = str(external_ids.get("ArXiv") or "").split("v")[0]
        doi = str(external_ids.get("DOI") or "")
        open_access_pdf = item.get("openAccessPdf") or {}
        paper_url = open_access_pdf.get("url") or item.get("url")
        if not paper_url:
            continue

        source_id = str(item.get("paperId") or "")
        if not source_id:
            continue

        papers.append(Paper(
            source="semantic_scholar",
            source_id=source_id,
            title=title,
            authors=[
                str(author.get("name"))
                for author in item.get("authors") or []
                if author.get("name")
            ],
            abstract=abstract.replace("\n", " "),
            published_date=published_date,
            paper_url=str(paper_url),
            arxiv_id=arxiv_id,
            doi=doi,
        ))

    papers.sort(key=lambda paper: paper.published_date, reverse=True)
    return papers[:max_results]

def get_daily_papers(
        topic,
        filters,
        max_results=10,
        existing_data=None,
        sources=None,
        openreview_search_limit=50):
    """Fetch and merge papers from enabled sources for one topic."""
    papers = {}
    web_content = {}
    existing_papers = existing_data.get(topic, {}) if existing_data else {}

    existing_titles = {}
    for existing_key, entry in existing_papers.items():
        parts = entry.split("|")
        if len(parts) >= 7:
            existing_titles[normalize_title(parts[2].strip())] = existing_key

    source_config = {
        "arxiv": True,
        "openreview": True,
        "semantic_scholar": True,
    }
    source_config.update(sources or {})

    results = []
    if source_config.get("arxiv"):
        arxiv_query = " OR ".join(
            f'"{item}"' if " " in item else item
            for item in filters
        )
        arxiv_papers = fetch_arxiv_papers(arxiv_query, max_results)
        logging.info(f"arXiv返回 {len(arxiv_papers)} 篇: {topic}")
        results.extend(arxiv_papers)

    if source_config.get("openreview"):
        openreview_papers = fetch_openreview_papers(
            topic,
            max_results=max_results,
            search_limit=openreview_search_limit,
        )
        logging.info(f"OpenReview返回 {len(openreview_papers)} 篇: {topic}")
        results.extend(openreview_papers)

    if source_config.get("semantic_scholar"):
        semantic_scholar_papers = fetch_semantic_scholar_papers(
            filters,
            max_results=max_results,
        )
        logging.info(
            f"Semantic Scholar返回 {len(semantic_scholar_papers)} 篇: {topic}"
        )
        results.extend(semantic_scholar_papers)

    results = deduplicate_papers(results)
    if not results:
        logging.warning(f"所有来源均未返回主题 '{topic}' 的近期论文")
        return {}, {}

    for paper in results:
        try:
            if not is_date_in_range(paper.published_date):
                continue

            paper_key = paper.storage_key
            normalized_title = normalize_title(paper.title)
            existing_title_key = existing_titles.get(normalized_title)
            if existing_title_key and existing_title_key != paper_key:
                logging.info(f"现有数据已包含同标题论文: {paper.title}")
                continue

            summary = None
            existing_key = paper_key if paper_key in existing_papers else existing_title_key
            if existing_key:
                existing_entry = existing_papers[existing_key]
                parts = existing_entry.split('|')
                if len(parts) >= 7:
                    existing_summary = parts[6].strip()
                    if existing_summary and existing_summary not in ["无", "null", ""]:
                        summary = existing_summary
                        logging.info(f"使用现有摘要: {paper_key}")

            if not summary:
                summary = get_paper_summary(paper.title, paper.abstract)
                logging.info(f"生成新摘要: {paper_key}")

            code_link = get_official_code_link(
                paper.arxiv_id,
                paper.title,
                paper.authors,
            )
            code_display = "无"
            if code_link:
                code_display = f"[代码]({code_link})"

            title = sanitize_entry_text(paper.title)
            authors = sanitize_entry_text(
                get_authors(paper.authors, first_author=True)
            )
            summary = sanitize_entry_text(summary)
            paper_link = f"[{paper.link_label}]({paper.paper_url})"
            papers[paper_key] = (
                f"|{paper.published_date}|{title}|{authors}|{paper_link}|"
                f"{code_display}|{summary}|\n"
            )

            web_entry = (
                f"- {paper.published_date}, {paper.title}, {authors}, "
                f"论文: {paper_link}"
            )
            if code_link:
                web_entry += f", 代码: [链接]({code_link})"
            web_entry += f", 摘要: {summary}\n"
            web_content[paper_key] = web_entry

        except Exception as e:
            logging.error(f"处理 {paper.source} 论文出错: {str(e)}")

    return {topic: papers}, {topic: web_content}

def update_paper_links(filename):
    """更新JSON文件中的代码链接并应用过滤"""
    def parse_arxiv_string(s):
        parts = s.split("|")
        if len(parts) < 7:
            return None, None, None, None, None, None
        
        date = parts[1].strip()
        title = parts[2].strip()
        authors = parts[3].strip()
        # 提取arxiv_id
        arxiv_match = re.search(r'$$(.*?)$$', parts[4])
        arxiv_id = arxiv_match.group(1) if arxiv_match else ""
        code = parts[5].strip()
        summary = parts[6].strip()
        
        return date, title, authors, arxiv_id, code, summary

    with open(filename, "r") as f:
        content = f.read()
        data = json.loads(content) if content else {}

    # 应用全局过滤
    for topic in list(data.keys()):
        data[topic] = filter_old_papers(data[topic])
        # 删除空主题
        if not data[topic]:
            del data[topic]
            logging.info(f"删除空主题: {topic}")

    updated_data = data.copy()

    for keyword, papers in data.items():
        logging.info(f'更新关键词: {keyword}')
        for paper_id, content_str in papers.items():
            try:
                date, title, author, arxiv_id, code, summary = parse_arxiv_string(content_str)
                if not arxiv_id:
                    continue
                
                # 如果代码列为"无"或为空，尝试更新
                if code in ["无", "null", ""]:
                    try:
                        # 尝试paperswithcode
                        code_url = base_url + arxiv_id
                        response = requests.get(code_url, timeout=10)
                        if response.status_code == 200:
                            result = response.json()
                            if result.get("official") and result["official"].get("url"):
                                new_code = f"[代码]({result['official']['url']})"
                                
                                # 更新内容但保留摘要
                                new_content = f"|{date}|{title}|{author}|[{arxiv_id}](http://arxiv.org/pdf/{arxiv_id})|{new_code}|{summary}|\n"
                                updated_data[keyword][paper_id] = new_content
                                logging.info(f'为 {arxiv_id} 更新代码链接')
                                continue
                        
                        # 尝试GitHub搜索
                        gh_link = get_official_code_link(arxiv_id, title, [])
                        if gh_link:
                            new_code = f"[极代码]({gh_link})"
                            new_content = f"|{date}|{title}|{author}|[{arxiv_id}](http://arxiv.org/pdf/{arxiv_id})|{new_code}|{summary}|\n"
                            updated_data[keyword][paper_id] = new_content
                            logging.info(f'为 {arxiv_id} 添加GitHub代码链接')
                    except Exception as e:
                        logging.error(f"更新代码链接时出错: {str(e)}")
            except Exception as e:
                logging.error(f"解析论文条目时出错: {str(e)}")
    
    # 保存更新后的数据
    with open(filename, "w") as f:
        json.dump(updated_data, f, indent=2)

def update_json_file(filename, data_dict):
    """更新JSON文件并应用过滤"""
    try:
        # 读取现有数据
        with open(filename, "r") as f:
            existing_data = json.load(f)
        
        # 应用全局过滤
        for topic in list(existing_data.keys()):
            existing_data[topic] = filter_old_papers(existing_data[topic])
            # 删除空主题
            if not existing_data[topic]:
                del existing_data[topic]
                logging.info(f"清理空主题: {topic}")
                
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = {}
    
    # 更新数据并再次过滤
    for data in data_dict:
        for topic, papers in data.items():
            # 过滤新数据
            filtered_papers = filter_old_papers(papers)
            if not filtered_papers:
                continue
                
            if topic in existing_data:
                # 合并前过滤现有数据
                existing_data[topic] = filter_old_papers(existing_data[topic])
                existing_data[topic].update(filtered_papers)
            else:
                existing_data[topic] = filtered_papers
    
    # 保存更新
    with open(filename, "w") as f:
        json.dump(existing_data, f, indent=2)

def json_to_md(filename, md_filename,
               task='',
               to_web=False,
               use_title=True,
               use_tc=True,
               show_badge=True,
               use_b2t=True):
    """生成Markdown文件并应用最终过滤"""
    today = datetime.date.today().strftime('%Y.%m.%d')
    
    # 1. 加载并过滤数据
    try:
        with open(filename, "r") as f:
            data = json.load(f)
            
        # 应用最终过滤
        for topic in list(data.keys()):
            data[topic] = filter_old_papers(data[topic])
            if not data[topic]:
                del data[topic]
                
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    
    # 2. 创建Markdown文件
    with open(md_filename, "w+", encoding="utf-8") as f:
        # 添加标题和介绍
        f.write(f"# SLAM领域最新论文 ({today})\n\n")
        f.write("> 每日自动更新SLAM领域的 arXiv、OpenReview 与 Semantic Scholar 论文\n\n")
        f.write("> 使用说明: [点击查看](./docs/README.md#usage)\n\n")
        
        # 3. 优化表格CSS
#         f.write("""<style>
# .table-container {
#   overflow-x: auto;
#   margin-bottom: 20px;
# }
# table {
#   width: 100%;
#   font-size: 0.85em;
#   border-collapse: collapse;
# }
# th, td {
#   border: 1px solid #ddd;
#   padding: 10px;
#   text-align: left;
#   vertical-align: top;
# }
# th {
#   background-color: #f8f9fa;
#   font-weight: bold;
#   position: sticky;
#   top: 0;
# }
# /* 标题列样式 */
# td:nth-child(2) {
#   max-width: none;
#   word-wrap: break-word;
#   overflow-wrap: anywhere;
# }
# td:nth-child(4) {
#   max-width: 400px;
#   word-wrap: break-word;
#   line-height: 1.6;
# }
# /* 响应式设计 */
# @media (max-width: 768px) {
#   .table-container {
#     font-size: 0.75em;
#     display: block;
#     overflow-x: auto;
#   }
#   td:nth-child(2) {
#     min-width: 200px;
#   }
# }
# </style>\n\n""")
        
        # 4. 添加目录（如果需要）
        if use_tc:
            f.write("<details>\n<summary>分类目录</summary>\n<ol>\n")
            for keyword in data.keys():
                if data[keyword]:
                    kw_slug = re.sub(r'\W+', '-', keyword.lower())
                    f.write(f"<li><a href='#{kw_slug}'>{keyword}</a></li>\n")
            f.write("</ol>\n</details>\n\n")
        
        # 5. 添加各个主题部分
        for keyword, papers in data.items():
            if not papers:
                continue
                
            kw_slug = re.sub(r'\W+', '-', keyword.lower())
            f.write(f"<h2 id='{kw_slug}'>{keyword}</h2>\n\n")
            
            f.write('<div class="table-container">\n')
            f.write("<table>\n")
            f.write("<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th></tr></thead>\n")
            f.write("<tbody>\n")
            
            sorted_papers = sorted(
                papers.items(),
                key=lambda item: item[1].split("|")[1],
                reverse=True,
            )
            
            for paper_id, paper_entry in sorted_papers:
                entry_parts = paper_entry.strip().split('|')
                if len(entry_parts) >= 7:
                    date_str = entry_parts[1].strip()
                    title = entry_parts[2].strip()
                    paper_link = entry_parts[4].strip()
                    code_link = entry_parts[5].strip()
                    summary = entry_parts[6].strip()
                    

                    if not is_date_above_min(date_str):
                        continue
                    
                    if not summary or summary in ["无", "null"]:
                        summary = "摘要生成中..."
                    
                    # 合并论文链接和代码链接
                    paper_display = paper_link
                    if code_link not in ["无", "null", ""]:
                        # 修正正则表达式提取URL
                        code_url_match = re.search(r'$(.*?)$', code_link)
                        if code_url_match:
                            code_url = code_url_match.group(1)
                            paper_display = f"{paper_link}<br><a href='{code_url}'>[代码]</a>"
                    
                    f.write("<tr>")
                    f.write(f"<td>{html.escape(date_str)}</td>")
                    f.write(f"<td>{html.escape(title)}</td>")
                    f.write(f"<td>{paper_display}</td>")
                    f.write("</tr>\n")
                    f.write("<tr>")
                    f.write(
                        f'<td colspan="3">{render_summary_html(summary)}</td>'
                    )
                    f.write("</tr>\n")
            
            f.write("</tbody>\n")
            f.write("</table>\n")
            f.write("</div>\n\n")
            
            if use_b2t:
                f.write(f"<div align='right'><a href='#top'>↑ 返回顶部</a></div>\n\n")
        
        # 7. 添加页脚
        f.write("---\n")
        f.write("> 本列表自动生成 | [反馈问题](https://github.com/your-repo/issues)\n")
        f.write(f"> 更新于: {today}\n")
    
    logging.info(f"{task} 已完成，保存到 {md_filename}")

def demo(**config):
    data_collector = []
    data_collector_web = []
    
    # 解析配置
    keywords = config['keywords']
    max_results = config['max_results']
    sources = config.get('sources', {})
    openreview_search_limit = config.get('openreview_search_limit', 50)
    publish_readme = config['publish_readme']
    publish_gitpage = config['publish_gitpage']
    publish_wechat = config['publish_wechat']
    show_badge = config['show_badge']
    update_links = config['update_paper_links']
    
    # 读取现有数据（用于保留已有摘要）
    existing_data = {}
    if publish_readme and not update_links:
        try:
            with open(config['json_readme_path'], 'r') as f:
                existing_data = json.load(f)
            
            # 清理现有数据中的旧论文
            for topic in list(existing_data.keys()):
                existing_data[topic] = filter_old_papers(existing_data[topic])
                if not existing_data[topic]:
                    del existing_data[topic]
                    
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = {}
    
    logging.info(f'更新论文链接: {update_links}')
    if not update_links:
        logging.info("开始获取每日论文")
        for topic, settings in keywords.items():
            logging.info(f"关键词: {topic}")
            data, data_web = get_daily_papers(
                topic,
                filters=settings['filters'],
                max_results=max_results,
                existing_data=existing_data,
                sources=sources,
                openreview_search_limit=openreview_search_limit,
            )
            data_collector.append(data)
            data_collector_web.append(data_web)
        logging.info("获取每日论文完成")
    
    # 更新README.md
    if publish_readme:
        json_file = config['json_readme_path']
        md_file = config['md_readme_path']
        if update_links:
            update_paper_links(json_file)
        else:
            update_json_file(json_file, data_collector)
        json_to_md(json_file, md_file, task='更新README.md')
    
    # 更新GitHub Pages
    if publish_gitpage:
        json_file = config['json_gitpage_path']
        md_file = config['md_gitpage_path']
        if update_links:
            update_paper_links(json_file)
        else:
            update_json_file(json_file, data_collector)
        json_to_md(json_file, md_file, task='更新GitPage', to_web=True, 
                   use_tc=True, use_b2t=False)
    
    # 更新微信文档
    if publish_wechat:
        json_file = config['json_wechat_path']
        md_file = config['md_wechat_path']
        if update_links:
            update_paper_links(json_file)
        else:
            update_json_file(json_file, data_collector_web)
        json_to_md(json_file, md_file, task='更新微信', to_web=False, 
                   use_title=False)

def is_date_above_min(date_str: str) -> bool:
    date_value = datetime.date.fromisoformat(date_str.replace('**', ''))
    return is_date_in_range(date_value)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--update_paper_links', action='store_true',
                       help='是否更新论文链接')
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(logging.INFO)
    logging.info(
        f"启动论文速递更新 (日期范围: {MIN_DATE} 至 {current_date()})"
    )
    
    # 加载配置并运行
    config = load_config(args.config_path)
    config['update_paper_links'] = args.update_paper_links
    demo(**config)
    
    logging.info("更新完成")
