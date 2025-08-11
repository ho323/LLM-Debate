import requests
import json
import time
from bs4 import BeautifulSoup
from datetime import datetime
import re
from urllib.parse import urlparse

class SingleArticleCrawler:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def get_site_type(self, url):
        """URL을 통해 사이트 타입 판별"""
        domain = urlparse(url).netloc.lower()
        
        if 'hani.co.kr' in domain:
            if 'h21' in domain:
                return 'h21'  # 한겨레21
            elif 'seouland' in url.lower():
                return 'seouland'  # seouland
            else:
                return 'hani'  # 한겨레
        else:
            return 'unknown'
    
    def extract_date(self, soup, url):
        """다양한 날짜 추출 방법"""
        try:
            # 1. 일반적인 날짜 선택자들
            date_selectors = [
                'time[datetime]',
                '.date-time',
                '.article-date',
                '.byline-date',
                '.published-date',
                '.post-date',
                '.date',
                '.meta-date',
                '[data-date]',
                '.article-info .date'
            ]
            
            for selector in date_selectors:
                date_elem = soup.select_one(selector)
                if date_elem:
                    date_text = (date_elem.get('datetime') or 
                               date_elem.get('data-date') or 
                               date_elem.get_text(strip=True))
                    
                    # ISO 형식 날짜 찾기
                    date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', date_text)
                    if date_match:
                        return date_match.group(0)
            
            # 2. URL에서 날짜 추출 시도
            url_date_match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', url)
            if url_date_match:
                year, month, day = url_date_match.groups()
                return f"{year}-{month}-{day}"
            
            # 3. 기사 본문에서 날짜 패턴 찾기
            text = soup.get_text()
            date_patterns = [
                r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일',
                r'(\d{4})-(\d{2})-(\d{2})',
                r'(\d{4})\.(\d{2})\.(\d{2})',
                r'(\d{4})/(\d{2})/(\d{2})',
                r'등록\s*:?\s*(\d{4})-(\d{2})-(\d{2})',
                r'작성\s*:?\s*(\d{4})-(\d{2})-(\d{2})'
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, text)
                if match:
                    groups = match.groups()
                    if len(groups) >= 3:
                        year, month, day = groups[:3]
                        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            
            # 4. 현재 날짜로 기본값 설정
            return datetime.now().strftime("%Y-%m-%d")
            
        except Exception as e:
            print(f"날짜 추출 오류: {e}")
            return datetime.now().strftime("%Y-%m-%d")
    
    def extract_title(self, soup):
        """기사 제목 추출"""
        title_selectors = [
            'h1.title',
            'h1.article-title',
            'h1.headline',
            '.article-header h1',
            '.content-header h1',
            'h1',
            '.title',
            '.article-title',
            'title'
        ]
        
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                title = title_elem.get_text(strip=True)
                # 사이트명이나 불필요한 텍스트 제거
                title = re.sub(r'\s*[-|]\s*(한겨레|hani|seouland).*$', '', title, flags=re.I)
                if title and len(title) > 5:  # 의미있는 제목만
                    return title
        
        return "제목을 찾을 수 없습니다"
    
    def extract_content(self, soup, site_type):
        """사이트별 맞춤 본문 추출"""
        content_paragraphs = []
        
        # 사이트별 본문 선택자
        if site_type == 'h21':
            content_selectors = [
                '.article-text',
                '.article-content',
                '#article-text',
                '.content-body'
            ]
        elif site_type == 'seouland':
            content_selectors = [
                '.article-content',
                '.content-body',
                '.post-content'
            ]
        else:  # 일반 한겨레
            content_selectors = [
                '.article-text',
                '.article-content',
                '#article-text',
                '.content',
                '.article-body'
            ]
        
        # 불필요한 요소들 먼저 제거
        for unwanted in soup.find_all(['script', 'style', 'iframe', 'ins', 'noscript']):
            unwanted.decompose()
        
        # 광고 관련 요소 제거
        for ad in soup.find_all(class_=re.compile(r'(ad|advertisement|banner|popup)', re.I)):
            ad.decompose()
        
        # 본문 추출
        for selector in content_selectors:
            content_div = soup.select_one(selector)
            if content_div:
                # 본문 내 불필요한 요소 제거
                for unwanted in content_div.find_all(['script', 'style', 'iframe', 'ins', '.ad', '.advertisement', '.related', '.share']):
                    unwanted.decompose()
                
                # 문단 추출 (p, div 태그)
                paragraphs = content_div.find_all(['p', 'div'])
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    
                    # 필터링 조건
                    if (text and 
                        len(text) > 10 and  # 최소 길이
                        not any(skip_word in text.lower() for skip_word in [
                            '광고', 'ad', '©', 'copyright', '저작권', 
                            '기자', 'reporter', '편집자', 'editor',
                            '구독', 'subscribe', '댓글', 'comment',
                            '공유', 'share', '카카오', 'facebook', 'twitter'
                        ]) and
                        not re.match(r'^[\s\-=]+$', text) and  # 구분선 제외
                        not text.startswith('▲') and  # 사진 설명 제외
                        not text.startswith('△')):
                        
                        content_paragraphs.append(text)
                
                if content_paragraphs:
                    break
        
        # 본문을 찾지 못한 경우 전체 텍스트에서 추출
        if not content_paragraphs:
            print("기본 본문 선택자로 내용을 찾지 못했습니다. 전체 텍스트에서 추출합니다.")
            
            # 헤더, 푸터, 네비게이션 등 제거
            for unwanted in soup.find_all(['header', 'footer', 'nav', 'aside', '.menu', '.navigation', '.sidebar']):
                unwanted.decompose()
            
            all_text = soup.get_text()
            # 문장 단위로 분리
            sentences = re.split(r'[.!?]\s+', all_text)
            content_paragraphs = [s.strip() for s in sentences 
                                if len(s.strip()) > 20 and 
                                not any(skip in s.lower() for skip in ['광고', '저작권', '구독'])]
        
        # 빈 결과인 경우 기본 메시지
        if not content_paragraphs:
            content_paragraphs = ["본문 내용을 추출할 수 없습니다."]
        
        return content_paragraphs
    
    def get_political_position(self, url):
        """URL 기반으로 언론사의 정치적 성향 판별"""
        domain = urlparse(url).netloc.lower()
        
        if 'hani.co.kr' in domain:
            return 'progressive'  # 한겨레 계열은 진보
        else:
            return 'unknown'
    
    def crawl_single_article(self, url):
        """단일 기사 URL 크롤링"""
        try:
            print(f"기사 크롤링 시작: {url}")
            
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 사이트 타입 판별
            site_type = self.get_site_type(url)
            print(f"사이트 타입: {site_type}")
            
            # 제목 추출
            title = self.extract_title(soup)
            print(f"제목: {title}")
            
            # 날짜 추출
            date = self.extract_date(soup, url)
            print(f"날짜: {date}")
            
            # 본문 추출
            content = self.extract_content(soup, site_type)
            print(f"추출된 문단 수: {len(content)}")
            
            # 정치적 성향
            position = self.get_political_position(url)
            
            return {
                'title': title,
                'url': url,
                'content': content,
                'date': date,
                'position': position
            }
            
        except Exception as e:
            print(f"크롤링 오류 ({url}): {e}")
            return None
    
    def crawl_multiple_articles(self, urls):
        """여러 기사 URL을 한번에 크롤링"""
        articles = []
        
        for i, url in enumerate(urls, 1):
            print(f"\n[{i}/{len(urls)}] 크롤링 중...")
            article = self.crawl_single_article(url)
            
            if article:
                articles.append(article)
                print("✓ 성공")
            else:
                print("✗ 실패")
            
            # 서버 부하 방지
            if i < len(urls):
                time.sleep(1)
        
        return articles
    
    def save_to_json(self, articles, output_file="articles.json"):
        """크롤링 결과를 JSON 파일로 저장"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        
        print(f"\n{len(articles)}개 기사가 '{output_file}'에 저장되었습니다.")
        
        # 저장 형식 미리보기
        if articles:
            print("\n=== 저장 형식 미리보기 ===")
            example = articles[0]
            preview = {
                "title": example['title'][:50] + "..." if len(example['title']) > 50 else example['title'],
                "url": example['url'],
                "content": example['content'][:2] if len(example['content']) >= 2 else example['content'],
                "date": example['date'],
                "position": example['position']
            }
            print(json.dumps(preview, ensure_ascii=False, indent=2))

# 사용 예시
if __name__ == "__main__":
    crawler = SingleArticleCrawler()
    
    # 1. 단일 기사 크롤링
    single_url = "https://h21.hani.co.kr/arti/politics/politics_general/57552.html"
    article = crawler.crawl_single_article(single_url)
    
    if article:
        # 단일 기사 저장
        crawler.save_to_json([article], "single_article.json")
        
        # 결과 출력
        print(f"\n=== 크롤링 결과 ===")
        print(f"제목: {article['title']}")
        print(f"URL: {article['url']}")
        print(f"날짜: {article['date']}")
        print(f"문단 수: {len(article['content'])}")
        print(f"정치적 성향: {article['position']}")
        
        if article['content']:
            print(f"\n첫 번째 문단: {article['content'][0][:100]}...")
    
    # 2. 여러 기사 동시 크롤링 예시
    multiple_urls = [
        "https://h21.hani.co.kr/arti/politics/politics_general/57552.html",
        "https://h21.hani.co.kr/arti/politics/politics_general/57378.html",
        "https://www.seouland.com/arti/society/society_general/21622.html?_gl=1*k7e8x6*_ga*MTQ0NTIzMjI5NC4xNzUzNDAzMDk4*_ga_6MQZGS06GJ*czE3NTM0MDMwOTgkbzEkZzEkdDE3NTM0MDM3MTUkajYwJGwwJGgw",
        "https://www.seouland.com/arti/society/society_general/21633.html?_gl=1*1nsewbr*_ga*MTQ0NTIzMjI5NC4xNzUzNDAzMDk4*_ga_6MQZGS06GJ*czE3NTM0MDMwOTgkbzEkZzEkdDE3NTM0MDM3NDEkajM0JGwwJGgw",
        "https://www.seouland.com/arti/society/society_general/21717.html?_gl=1*117gzv6*_ga*MTQ0NTIzMjI5NC4xNzUzNDAzMDk4*_ga_6MQZGS06GJ*czE3NTM0MDMwOTgkbzEkZzEkdDE3NTM0MDM3NTYkajE5JGwwJGgw",
        

        # 추가 URL들을 여기에 넣으세요
    ]
    
    articles = crawler.crawl_multiple_articles(multiple_urls)
    crawler.save_to_json(articles, "multiple_articles.json")