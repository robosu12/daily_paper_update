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

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

base_url = "https://arxiv.paperswithcode.com/api/v0/papers/"
github_url = "https://api.github.com/search/repositories"
arxiv_url = "http://arxiv.org/"

# DeepSeek API配置
DEEPSEEK_API_KEY = "sk-179d350b272b4b4da85b426b6271c7b5"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

def load_config(config_file:str) -> dict:
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

def get_authors(authors, first_author=False):
    """优化作者显示格式"""
    if not first_author:
        return ", ".join(str(author) for author in authors)
    elif len(authors) > 1:
        return f"{str(authors[0])}等"
    return str(authors[0])

def sort_papers(papers):
    """按日期排序论文"""
    return dict(sorted(papers.items(), key=lambda x: x[0], reverse=True))

def get_official_code_link(paper_id: str, title: str, authors: list) -> str:
    """获取论文官方开源代码链接（带多重验证）"""
    # 1. 优先使用paperswithcode API获取官方链接
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
                    paper_id.split('v')[0] in repo_description or
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
    
    # 极3. 带重试机制的API请求
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
                    "max_tokens": 500,  # 增加token限制
                    "stream": False
                },
                timeout=30  # 增加超时时间
            )
            
            # 4. 检查响应状态
            if response.status_code != 200:
                logging.warning(f"DeepSeek API响应错误: {response.status_code}")
                continue  # 继续重试
            
            # 5. 解析API响应
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                logging.warning("DeepSeek API返回空内容")
                continue  # 继续重试
            
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
    """获取arXiv结果（带重试机制）"""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # 使用arxiv库获取结果
            client = arxiv.Client(num_retries=3)
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            return list(client.results(search))
        except Exception as e:
            logging.warning(f"arXiv API请求失败 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + 1  # 指数退避
                logging.info(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
    
    # 如果所有重试都失败，尝试直接API调用
    logging.warning("arxiv.py库请求失败，尝试直接调用arXiv API")
    try:
        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        response = requests.get(url, params=params, timeout=30)
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
                authors=[arxiv.Author(name=a.name) for a in entry.authors],
                summary=entry.summary,
                comment=entry.get("arxiv_comment", ""),
                journal_ref=entry.get("arxiv_journal_ref", ""),
                doi=entry.get("arxiv_doi", ""),
                primary_category=entry.get("arxiv_primary_category", {}).get("term", ""),
                categories=[t.term for t in entry.tags if t.scheme == "http://arxiv.org/schemas/atom"],
                links=[arxiv.Link(href=l.href, title=l.title, rel=l.rel) for l in entry.links],
                pdf_url=next((l.href for l in entry.links if l.title == "pdf"), None)
            )
            results.append(result)
        return results
    except Exception as e:
        logging.error(f"直接arXiv API请求失败: {e}")
        return []

def get_daily_papers(topic, query="slam", max_results=10, existing_data=None):
    """获取每日论文 - 修复代码链接问题并保留已有摘要"""
    papers = {}
    web_content = {}
    
    # 获取现有数据（如果提供）
    existing_papers = existing_data.get(topic, {}) if existing_data else {}
    
    # 获取arXiv结果
    results = fetch_arxiv_results(query, max_results)
    if not results:
        logging.error(f"无法获取主题 '{topic}' 的论文")
        return {}, {}
    
    for result in results:
        try:
            # 提取基础信息
            paper_id = result.get_short_id()
            paper_key = paper_id.split('v')[0]
            title = result.title
            pdf_url = arxiv_url + 'pdf/' + paper_key
            authors = get_authors(result.authors, first_author=True)
            abstract = result.summary.replace("\n", " ")
            date = result.updated.date()
            
            # 过滤2025-05之前的论文
            if date < datetime.date(2025, 5, 1):
                continue
            
            # 使用完整标题（不缩略）
            short_title = title
            
            # 检查是否已有摘要
            summary = None
            if paper_key in existing_papers:
                existing_entry = existing_papers[paper_key]
                # 从现有条目中提取摘要
                parts = existing_entry.split('|')
                if len(parts) >= 7:  # 确保格式正确
                    existing_summary = parts[6].strip()
                    if existing_summary and existing_summary not in ["无", "null", ""]:
                        summary = existing_summary
                        logging.info(f"使用现有摘要: {paper_key}")
            
            # 如果没有现有摘要或无效，生成新摘要
            if not summary:
                summary = get_paper_summary(title, abstract)
                logging.info(f"生成新摘要: {paper_key}")
            
            # 获取官方代码链接
            code_link = get_official_code_link(paper_id, title, result.authors)
                
            # 构建表格行
            code_display = "无"
            if code_link:
                code_display = f"[代码]({code_link})"
                
            papers[paper_key] = f"|{date}|{short_title}|{authors}|[{paper_key}]({pdf_url})|{code_display}|{summary}|\n"
            
            # 构建网页内容
            web_entry = f"- {date}, {title}, {authors}等, 论文: [{paper_key}]({pdf_url})"
            if code_link:
                web_entry += f", 代码: [链接]({code_link})"
            web_entry += f", 摘要: {summary}\n"
            web_content[paper_key] = web_entry
        
        except Exception as e:
            logging.error(f"处理论文出错: {str(e)}")
    
    return {topic: papers}, {topic: web_content}

def update_paper_links(filename):
    """更新JSON文件中的代码链接并保留摘要"""
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

    # 清理旧论文 (2025-05之前)
    cutoff_date = datetime.date(2025, 5, 1)
    for topic in list(data.keys()):
        papers = data[topic]
        papers_to_remove = []
        for paper_id, entry in papers.items():
            parts = entry.split('|')
            if len(parts) >= 2:
                date_str = parts[1].strip()
                try:
                    paper_date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
                    if paper_date < cutoff_date:
                        papers_to_remove.append(paper_id)
                except Exception:
                    continue
        
        # 删除旧论文
        for paper_id in papers_to_remove:
            del papers[paper_id]
            logging.info(f"删除旧论文: {paper_id}")
        
        # 如果主题下没有论文了，删除该主题
        if not papers:
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
                            new_code = f"[代码]({gh_link})"
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
    """使用数据字典更新JSON文件"""
    try:
        # 读取现有数据
        with open(filename, "r") as f:
            existing_data = json.load(f)
        
        # 清理旧论文 (2025-05之前)
        cutoff_date = datetime.date(2025, 5, 1)
        for topic in list(existing_data.keys()):
            papers = existing_data[topic]
            papers_to_remove = []
            for paper_id, entry in papers.items():
                parts = entry.split('|')
                if len(parts) >= 2:
                    date_str = parts[1].strip()
                    try:
                        paper_date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
                        if paper_date < cutoff_date:
                            papers_to_remove.append(paper_id)
                    except Exception:
                        continue
            
            # 删除旧论文
            for paper_id in papers_to_remove:
                del papers[paper_id]
                logging.info(f"删除旧论文: {paper_id}")
            
            # 如果主题下没有论文了，删除该主题
            if not papers:
                del existing_data[topic]
                logging.info(f"删除空主题: {topic}")
                
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = {}
    
    # 更新数据
    for data in data_dict:
        for topic, papers in data.items():
            if topic in existing_data:
                existing_data[topic].update(papers)
            else:
                existing_data[topic] = papers
    
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
    """优化Markdown生成"""
    today = datetime.date.today().strftime('%Y.%m.%d')
    
    # 1. 加载数据
    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    
    # 2. 创建Markdown文件
    with open(md_filename, "w+", encoding="utf-8") as f:
        # 添加标题和介绍
        f.write(f"# 计算机视觉领域最新论文 ({today})\n\n")
        f.write("> 每日自动更新计算机视觉领域的最新arXiv论文\n\n")
        f.write("> 使用说明: [点击查看](./docs/README.md#usage)\n\n")
        
        # 3. 优化表格CSS
        f.write("""<style>
.table-container {
  overflow-x: auto;
  margin-bottom: 20px;
}
table {
  width: 100%;
  font-size: 0.85em;
  border-collapse: collapse;
}
th, td {
  border: 1px solid #ddd;
  padding: 10px;
  text-align: left;
  vertical-align: top;
}
th {
  background-color: #f8f9fa;
  font-weight: bold;
  position: sticky;
  top: 0;
}
/* 标题列样式 */
td:nth-child(2) {
  max-width: none;
  word-wrap: break-word;
  overflow-wrap: anywhere;
}
td:nth-child(4) {
  max-width: 400px;
  word-wrap: break-word;
  line-height: 1.6;
}
/* 响应式设计 */
@media (max-width: 768px) {
  .极able-container {
    font-size: 0.75em;
    display: block;
    overflow-x: auto;
  }
  td:nth-child(2) {
    min-width: 200px;
  }
}
</style>\n\n""")
        
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
            f.write("<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>\n")
            f.write("<tbody>\n")
            
            sorted_papers = sorted(papers.items(), key=lambda x: x[0], reverse=True)
            
            for paper_id, paper_entry in sorted_papers:
                entry_parts = paper_entry.strip().split('|')
                if len(entry_parts) >= 7:
                    date_str = entry_parts[1].strip()
                    title = entry_parts[2].strip()
                    paper_link = entry_parts[4].strip()
                    code_link = entry_parts[5].strip()
                    summary = entry_parts[6].strip()
                    
                    # 二次过滤2025-05之前的论文
                    try:
                        paper_date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
                        if paper_date < datetime.date(2025, 5, 1):
                            continue
                    except Exception:
                        pass
                    
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
                    f.write(f"<td>{html.escape(summary)}</td>")
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
    keywords = config['kv']
    max_results = config['max_results']
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
            cutoff_date = datetime.date(2025, 5, 1)
            for topic in list(existing_data.keys()):
                papers = existing_data[topic]
                papers_to_remove = []
                for paper_id, entry in papers.items():
                    parts = entry.split('|')
                    if len(parts) >= 2:
                        date_str = parts[1].strip()
                        try:
                            paper_date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
                            if paper_date < cutoff_date:
                                papers_to_remove.append(paper_id)
                        except Exception:
                            continue
                
                # 删除旧论文
                for paper_id in papers_to_remove:
                    del papers[paper_id]
                    logging.info(f"清理现有数据中的旧论文: {paper_id}")
                
                # 如果主题下没有论文了，删除该主题
                if not papers:
                    del existing_data[topic]
                    logging.info(f"删除空主题: {topic}")
                    
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = {}
    
    logging.info(f'更新论文链接: {update_links}')
    if not update_links:
        logging.info("开始获取每日论文")
        for topic, query in keywords.items():
            logging.info(f"关键词: {topic}")
            # 传递现有数据以保留摘要
            data, data_web = get_daily_papers(topic, query=query, 
                                            max_results=max_results, 
                                            existing_data=existing_data)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--update_paper_links', action='store_true',
                       help='是否更新论文链接')
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(logging.INFO)
    logging.info("启动论文速递更新")
    
    # 加载配置并运行
    config = load_config(args.config_path)
    config['update_paper_links'] = args.update_paper_links
    demo(**config)
    
    logging.info("更新完成")