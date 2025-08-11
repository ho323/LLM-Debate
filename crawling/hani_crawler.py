import requests
from bs4 import BeautifulSoup
import json
import time
import urllib.parse
from datetime import datetime
import os
import sys
import locale

# Windows 콘솔 인코딩 설정
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

class HaniSearchCrawler:
    def __init__(self):
        self.base_url = "https://search.hani.co.kr/search/newslist"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.8,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.hani.co.kr/',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
    def get_search_page(self, keyword, page=1):
        """한겨레 검색 페이지에서 검색 결과 가져오기"""
        params = {
            'searchword': keyword,
            'sort': 'desc',
            'startdate': '2025.01.01',
            'enddate': '2025.07.25',
            'dt': 'all',
            'page': page
        }
        
        try:
            print(f"검색 요청: {self.base_url}?{urllib.parse.urlencode(params)}")
            response = self.session.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            response.encoding = 'utf-8'
            return response.text
        except requests.RequestException as e:
            print(f"페이지 {page} 요청 중 오류 발생: {e}")
            return None
    
    def parse_search_results(self, html):
        """검색 결과 HTML에서 기사 목록 추출"""
        soup = BeautifulSoup(html, 'html.parser')
        articles = []
        
        print("HTML 구조 분석 중...")
        
        # 한겨레 검색 결과 페이지의 다양한 선택자 시도
        possible_selectors = [
            '.search-list li',
            '.result-list li', 
            '.news-list li',
            'ul.search-result li',
            '.search-result-item',
            'li[class*="result"]',
            'div[class*="search"] li',
            'li'  # 최후의 수단
        ]
        
        article_items = []
        for selector in possible_selectors:
            article_items = soup.select(selector)
            if article_items:
                print(f"'{selector}' 선택자로 {len(article_items)}개 항목 발견")
                break
        
        if not article_items:
            print("기사 항목을 찾을 수 없습니다. HTML 구조 확인 필요")
            # 디버그를 위해 HTML 일부 출력
            print("HTML 샘플:")
            print(soup.prettify()[:2000])
            return articles
        
        for i, item in enumerate(article_items):
            try:
                # 다양한 방법으로 제목과 링크 추출 시도
                title_elem = None
                link = None
                
                # 제목과 링크 추출 시도
                possible_title_selectors = [
                    'a.title',
                    '.title a',
                    'h3 a',
                    'h4 a', 
                    'dt a',
                    'a[href*="/articles/"]',
                    'a[href*="hani.co.kr"]',
                    'a'
                ]
                
                for selector in possible_title_selectors:
                    title_elem = item.select_one(selector)
                    if title_elem:
                        break
                
                if not title_elem:
                    continue
                
                title = title_elem.get_text(strip=True)
                link = title_elem.get('href')
                
                # 링크 정규화
                if link:
                    if link.startswith('/'):
                        link = 'https://www.hani.co.kr' + link
                    elif not link.startswith('http'):
                        link = 'https://www.hani.co.kr/' + link
                
                # 날짜 추출 시도
                date = ""
                date_selectors = [
                    '.date',
                    '.time', 
                    '.publish-date',
                    'span[class*="date"]',
                    'span[class*="time"]',
                    'dd',
                    'time'
                ]
                
                for selector in date_selectors:
                    date_elem = item.select_one(selector)
                    if date_elem:
                        date = date_elem.get_text(strip=True)
                        break
                
                # 유효한 기사인지 확인
                if title and link and len(title) > 5:
                    articles.append({
                        'title': title,
                        'url': link,
                        'date': date
                    })
                    print(f"  추출된 기사 {len(articles)}: {title[:50]}...")
                    
            except Exception as e:
                print(f"기사 {i+1} 파싱 중 오류: {e}")
                continue
        
        return articles
    
    def get_article_content(self, url):
        """개별 기사 페이지에서 본문 내용을 문단 구조대로 추출"""
        try:
            print(f"    본문 추출 중: {url}")
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 한겨레 기사 본문 선택자들 (실제 구조에 맞게 조정 필요)
            content_selectors = [
                '.article-text',
                '.article-content', 
                '.news-content',
                '#article-content',
                '.content',
                'div[class*="content"]',
                'div[class*="article"]',
                '.text'
            ]
            
            paragraphs = []
            content_elem = None
            
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    print(f"    본문 요소 발견: {selector}")
                    break
            
            if content_elem:
                # 사이트 구조에 따른 문단 추출
                paragraphs = self.extract_paragraphs_by_structure(content_elem)
                
                if not paragraphs:
                    # 대안: 직접 텍스트 노드 추출
                    paragraphs = self.extract_paragraphs_by_text_nodes(content_elem)
            
            # 날짜 정보 더 정확하게 추출
            date = ""
            date_selectors = [
                '.date',
                '.article-date',
                '.publish-date',
                'time',
                '.time',
                'span[class*="date"]'
            ]
            
            for selector in date_selectors:
                date_elem = soup.select_one(selector)
                if date_elem:
                    date = date_elem.get_text(strip=True)
                    break
            
            return paragraphs, date
            
        except Exception as e:
            print(f"    기사 내용 추출 중 오류 ({url}): {e}")
            return [], ""
    
    def extract_paragraphs_by_structure(self, content_elem):
        """HTML 구조를 기반으로 문단 추출"""
        paragraphs = []
        
        # 1순위: p 태그들
        p_tags = content_elem.find_all('p', recursive=True)
        if p_tags:
            for p in p_tags:
                text = self.clean_paragraph_text(p)
                if text:
                    paragraphs.append(text)
        
        # 2순위: div 태그들 (p 태그가 없거나 부족한 경우)
        if len(paragraphs) < 3:  # 문단이 너무 적으면
            div_paragraphs = []
            div_tags = content_elem.find_all('div', recursive=True)
            for div in div_tags:
                # 자식 p나 div가 없는 div만 선택 (최종 텍스트 노드)
                if not div.find_all(['p', 'div']):
                    text = self.clean_paragraph_text(div)
                    if text:
                        div_paragraphs.append(text)
            
            if len(div_paragraphs) > len(paragraphs):
                paragraphs = div_paragraphs
        
        # 3순위: br 태그로 구분된 텍스트
        if len(paragraphs) < 2:
            br_paragraphs = self.extract_paragraphs_by_br(content_elem)
            if br_paragraphs:
                paragraphs = br_paragraphs
        
        return paragraphs
    
    def extract_paragraphs_by_text_nodes(self, content_elem):
        """텍스트 노드를 기반으로 문단 추출"""
        paragraphs = []
        
        # 모든 텍스트를 가져와서 줄바꿈으로 분리
        full_text = content_elem.get_text(separator='\n', strip=True)
        potential_paragraphs = full_text.split('\n')
        
        for text in potential_paragraphs:
            text = text.strip()
            # 의미있는 문단만 선택 (최소 20자, 특수문자 제외)
            if (len(text) >= 20 and 
                not text.startswith(('ⓒ', '저작권', '기자', '편집', '무단', '전재')) and
                not text.endswith(('기자', '편집자', '데스크')) and
                '.' in text):  # 문장 형태인지 확인
                paragraphs.append(text)
        
        return paragraphs
    
    def extract_paragraphs_by_br(self, content_elem):
        """br 태그로 구분된 문단 추출"""
        paragraphs = []
        
        # br 태그를 특수 구분자로 변경
        for br in content_elem.find_all('br'):
            br.replace_with('\n__PARAGRAPH_BREAK__\n')
        
        text = content_elem.get_text()
        potential_paragraphs = text.split('__PARAGRAPH_BREAK__')
        
        for text in potential_paragraphs:
            text = text.strip()
            if len(text) >= 20:
                paragraphs.append(text)
        
        return paragraphs
    
    def clean_paragraph_text(self, element):
        """문단 텍스트 정리"""
        if not element:
            return ""
        
        # 불필요한 하위 요소 제거
        for unwanted in element.find_all(['script', 'style', 'iframe', 'ad', 'advertisement']):
            unwanted.decompose()
        
        text = element.get_text(strip=True)
        
        # 필터링 조건
        if (len(text) < 20 or  # 너무 짧은 텍스트
            text.startswith(('ⓒ', '저작권', '©', 'Copyright', '기자', '편집', '무단', '전재', '배포', '금지')) or
            text.endswith(('기자', '편집자', '데스크', '뉴스')) or
            text.count('.') == 0 or  # 문장이 아닌 것
            len(text.split()) < 5):  # 단어가 너무 적은 것
            return ""
        
        # 텍스트 정리
        text = ' '.join(text.split())  # 공백 정리
        return text
    
    def crawl_articles(self, keyword="소비쿠폰", target_count=200, batch_size=5):
        """메인 크롤링 함수"""
        print(f"'{keyword}' 키워드로 한겨레 검색 결과에서 최신 기사 {target_count}개 크롤링 시작...")
        
        all_articles = []
        current_page = 1
        consecutive_empty_pages = 0
        max_empty_pages = 3
        
        while len(all_articles) < target_count and consecutive_empty_pages < max_empty_pages:
            print(f"\n=== 페이지 {current_page} 처리 중 ===")
            
            # 검색 결과 페이지 가져오기
            html = self.get_search_page(keyword, current_page)
            if not html:
                print(f"페이지 {current_page} 로드 실패")
                consecutive_empty_pages += 1
                current_page += 1
                continue
            
            # 기사 목록 파싱
            page_articles = self.parse_search_results(html)
            if not page_articles:
                print(f"페이지 {current_page}에서 기사를 찾을 수 없습니다.")
                consecutive_empty_pages += 1
                current_page += 1
                continue
            
            consecutive_empty_pages = 0  # 기사를 찾았으므로 리셋
            print(f"페이지 {current_page}에서 {len(page_articles)}개 기사 발견")
            
            # 각 기사의 본문 내용 가져오기
            for i, article in enumerate(page_articles):
                if len(all_articles) >= target_count:
                    break
                
                print(f"  기사 {i+1}/{len(page_articles)} 처리 중...")
                
                content_paragraphs, article_date = self.get_article_content(article['url'])
                
                article_data = {
                    'title': article['title'],
                    'url': article['url'],
                    'content': content_paragraphs,  # 문단 배열로 저장
                    'date': article_date or article['date'],
                    'position': 'progressive',  # 모든 기사에 추가
                }
                
                all_articles.append(article_data)
                print(f"    완료: {article['title'][:50]}... ({len(content_paragraphs)}개 문단)")
                
                # 요청 간격 조절 (서버 부하 방지)
                time.sleep(1)
            
            # 배치 단위로 중간 저장
            if current_page % batch_size == 0:
                batch_num = current_page // batch_size
                filename = f"hani_search_batch_{batch_num}.json"
                self.save_to_json(all_articles, filename)
                print(f"배치 {batch_num} 저장 완료 ({len(all_articles)}개 기사)")
            
            current_page += 1
            time.sleep(2)  # 페이지 간 간격
        
        # 최종 저장
        final_articles = all_articles[:target_count]
        self.save_to_json(final_articles, "hani_searching_final.json")
        
        print(f"\n크롤링 완료! 총 {len(final_articles)}개 기사 수집")
        return final_articles
    
    def save_to_json(self, articles, filename):
        """JSON 파일로 저장"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(articles, f, ensure_ascii=False, indent=2)
            print(f"파일 저장 완료: {filename}")
        except Exception as e:
            print(f"파일 저장 중 오류: {e}")

# 사용 예시
if __name__ == "__main__":
    crawler = HaniSearchCrawler()
    
    # 크롤링 실행
    articles = crawler.crawl_articles(
        keyword="소비쿠폰",
        target_count=200,
        batch_size=5  # 5페이지마다 중간 저장
    )
    
    # 결과 요약 출력
    print(f"\n=== 크롤링 결과 요약 ===")
    print(f"총 수집 기사: {len(articles)}개")
    
    if articles:
        print(f"첫 번째 기사: {articles[0]['title']}")
        print(f"마지막 기사: {articles[-1]['title']}")
        if articles[0]['content']:
            total_paragraphs = sum(len(a['content']) for a in articles if a['content'])
            avg_paragraphs = total_paragraphs // len(articles) if articles else 0
            print(f"평균 문단 수: {avg_paragraphs}개")
        
        # 샘플 출력
        print(f"\n=== 첫 번째 기사 샘플 ===")
        print(f"제목: {articles[0]['title']}")
        print(f"URL: {articles[0]['url']}")
        print(f"날짜: {articles[0]['date']}")
        print(f"총 문단 수: {len(articles[0]['content'])}개")
        
        # 문단별 출력 샘플
        if articles[0]['content']:
            print(f"\n=== 문단 구조 샘플 ===")
            for i, paragraph in enumerate(articles[0]['content'][:3], 1):  # 처음 3개 문단만
                print(f"문단 {i}: {paragraph[:100]}...")
                if i >= 3:
                    print(f"... (총 {len(articles[0]['content'])}개 문단 중 3개만 표시)")
                    break