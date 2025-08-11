#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import requests
from datetime import datetime, timedelta
import re
import time
import sys
import os

# 한글 출력을 위한 인코딩 설정
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    os.system('chcp 65001 > nul')

class YouTubeCommentCrawler:
    def __init__(self, api_key):
        """
        YouTube Data API v3를 사용한 댓글 크롤러
        
        Args:
            api_key (str): YouTube Data API v3 키
        """
        self.api_key = api_key  # 수정: 파라미터로 받은 키 사용
        self.base_url = "https://www.googleapis.com/youtube/v3"
        
    def get_channel_id_from_handle(self, channel_handle):
        """
        채널 핸들(@MBCNEWS11)로부터 채널 ID를 가져옵니다.
        """
        
        if channel_handle.startswith('@'):
            channel_handle = channel_handle[1:]
        
        print(f"채널 핸들 처리 중: {channel_handle}")
        
        # search API로 채널 검색
        search_url = f"{self.base_url}/search"
        search_params = {
            'part': 'snippet',
            'q': f'{channel_handle} MBC 뉴스',
            'type': 'channel',
            'maxResults': 10,
            'key': self.api_key
        }
        
        search_response = requests.get(search_url, params=search_params)
        search_data = search_response.json()
        
        print(f"검색 결과: {search_data}")
        
        if 'items' in search_data:
            for item in search_data['items']:
                channel_title = item['snippet']['title']
                channel_id = item['snippet']['channelId']
                print(f"찾은 채널: {channel_title} (ID: {channel_id})")
                
                # MBC 뉴스 관련 채널인지 확인
                if any(keyword in channel_title.lower() for keyword in ['mbc', '뉴스', 'news']):
                    return channel_id
        
        #forHandle 시도
        url = f"{self.base_url}/channels"
        params = {
            'part': 'id,snippet',
            'forHandle': channel_handle,
            'key': self.api_key
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        print(f"forHandle 결과: {data}")
        
        if 'items' in data and len(data['items']) > 0:
            return data['items'][0]['id']
        
        # 알려진 MBC 뉴스 채널 ID들 시도
        known_channel_ids = [
            "UCF4Wxdo3inmxP-Y59wXDsFw",  
            "UCyF4yV5qhk-IiPDLYjsX1gw",  
        ]
        
        for channel_id in known_channel_ids:
            url = f"{self.base_url}/channels"
            params = {
                'part': 'snippet',
                'id': channel_id,
                'key': self.api_key
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'items' in data and len(data['items']) > 0:
                channel_title = data['items'][0]['snippet']['title']
                print(f"확인된 채널: {channel_title} (ID: {channel_id})")
                if 'mbc' in channel_title.lower() and '뉴스' in channel_title.lower():
                    return channel_id
        
        raise Exception(f"채널을 찾을 수 없습니다: @{channel_handle}. API 응답을 확인해주세요.")
    
    def search_videos_by_keyword(self, channel_id, keyword, days_back=30, max_results=50):
        """
        특정 채널에서 키워드가 포함된 비디오를 검색합니다.
        
        Args:
            channel_id (str): 채널 ID
            keyword (str): 검색할 키워드
            days_back (int): 검색할 기간 (일)
            max_results (int): 최대 결과 수
        """
        # 한 달 전 날짜 계산
        published_after = (datetime.now() - timedelta(days=days_back)).isoformat() + 'Z'
        
        url = f"{self.base_url}/search"
        params = {
            'part': 'snippet',
            'channelId': channel_id,
            'q': keyword,
            'type': 'video',
            'publishedAfter': published_after,
            'order': 'date',
            'maxResults': max_results,
            'key': self.api_key
        }
        
        videos = []
        next_page_token = None
        
        while True:
            if next_page_token:
                params['pageToken'] = next_page_token
                
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'error' in data:
                print(f"API 오류: {data['error']['message']}")
                break
            
            if 'items' in data:
                for item in data['items']:
                    video_info = {
                        'video_id': item['id']['videoId'],
                        'video_url': f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                        'video_title': item['snippet']['title'],
                        'published_at': item['snippet']['publishedAt']
                    }
                    videos.append(video_info)
            
            # 다음 페이지가 있는지 확인
            if 'nextPageToken' in data and len(videos) < max_results:
                next_page_token = data['nextPageToken']
            else:
                break
                
            time.sleep(0.1)  
        
        return videos
    
    def get_video_comments(self, video_id, max_comments=50):
        """
        특정 비디오의 댓글을 좋아요 순으로 가져옵니다.
        
        Args:
            video_id (str): 비디오 ID
            max_comments (int): 가져올 최대 댓글 수 
        """
        url = f"{self.base_url}/commentThreads"
        params = {
            'part': 'snippet',
            'videoId': video_id,
            'order': 'relevance',  # 좋아요 순 정렬
            'maxResults': min(100, max_comments),  
            'key': self.api_key
        }
        
        comments = []
        next_page_token = None
        
        while len(comments) < max_comments:
            if next_page_token:
                params['pageToken'] = next_page_token
                
            try:
                response = requests.get(url, params=params)
                data = response.json()
                
                if 'error' in data:
                    print(f"댓글 가져오기 오류 (비디오 ID: {video_id}): {data['error']['message']}")
                    break
                
                if 'items' in data:
                    for item in data['items']:
                        comment_text = item['snippet']['topLevelComment']['snippet']['textDisplay']
                        like_count = item['snippet']['topLevelComment']['snippet']['likeCount']
                        
                        # HTML 태그 제거
                        comment_text = re.sub('<[^<]+?>', '', comment_text)
                        
                        comments.append({
                            'text': comment_text.strip(),
                            'likes': like_count
                        })
                
                # 다음 페이지가 있는지 확인
                if 'nextPageToken' in data and len(comments) < max_comments:
                    next_page_token = data['nextPageToken']
                else:
                    break
                    
            except Exception as e:
                print(f"댓글 가져오기 중 오류 발생: {e}")
                break
                
            time.sleep(0.1)  # API 제한 방지
        
        # 좋아요 수로 정렬하여 상위 댓글만 반환
        comments.sort(key=lambda x: x['likes'], reverse=True)
        return comments[:max_comments]
    
    def crawl_comments(self, channel_handle, keyword, output_file="youtube_comments.json", days_back=30, max_comments_per_video=50):
       
        try:
            print(f"채널 정보를 가져오는 중: {channel_handle}")
        except UnicodeEncodeError:
            print(f"Getting channel info: {channel_handle}")
        
        try:
            # 채널 ID 가져오기
            channel_id = self.get_channel_id_from_handle(channel_handle)
            print(f"채널 ID: {channel_id}")
            
            # 키워드로 비디오 검색
            print(f"'{keyword}' 키워드로 비디오 검색 중...")
            videos = self.search_videos_by_keyword(channel_id, keyword, days_back)
            print(f"찾은 비디오 수: {len(videos)}")
            
            if len(videos) == 0:
                print("검색된 비디오가 없습니다. 검색 조건을 확인해주세요.")
                return []
            
            # 각 비디오의 댓글 수집
            all_data = []
            
            for i, video in enumerate(videos):
                try:
                    print(f"비디오 {i+1}/{len(videos)} 처리 중: {video['video_title'][:50]}...")
                except UnicodeEncodeError:
                    print(f"Processing video {i+1}/{len(videos)}...")
                
                comments = self.get_video_comments(video['video_id'], max_comments_per_video)
                
                if len(comments) == 0:
                    print(f"  댓글이 비활성화되어 있거나 댓글이 없습니다.")
                    continue
                
                for comment in comments:
                    comment_text = comment['text']
                    like_count = comment['likes']
                    
                    # 문장별로 나누기 (마침표, 느낌표, 물음표 기준)
                    sentences = re.split(r'[.!?]\s+', comment_text)
                    
                    # 빈 문장 제거 및 정리
                    clean_sentences = []
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if sentence:  # 빈 문장 제외
                            clean_sentences.append(sentence)
                    
                    # 문장이 있는 경우에만 데이터 추가
                    if clean_sentences:
                        data_entry = {
                            "video_id": video['video_id'],
                            "video_url": video['video_url'],
                            "video_title": video['video_title'],
                            "comment": clean_sentences if len(clean_sentences) > 1 else clean_sentences[0],
                            "like_count": like_count
                        }
                        all_data.append(data_entry)
                
                print(f"  수집된 댓글 수: {len(comments)}")
                time.sleep(1)  # API 제한 방지
            
            # JSON 파일로 저장 (UTF-8 BOM 없이)
            with open(output_file, 'w', encoding='utf-8-sig') as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2)
            
            try:
                print(f"\n크롤링 완료!")
                print(f"총 수집된 댓글 수: {len(all_data)}")
                print(f"저장된 파일: {output_file}")
            except UnicodeEncodeError:
                print(f"\nCrawling completed!")
                print(f"Total comments collected: {len(all_data)}")
                print(f"Saved file: {output_file}")
            
            return all_data
            
        except Exception as e:
            print(f"오류 발생: {e}")
            return []

def main():
    # 여기에 실제 API 키를 넣으세요
    API_KEY = "AIzaSyBeHApgqnaT1IO9n4y1tH37h9SUl8vJAJE"
    
    # API 키 유효성 검사 (빈 값이나 기본값인지 확인)
    if not API_KEY or API_KEY == "AIzaSyBeHApgqnaT1IO9n4y1tH37h9SUl8vJAJE":
        try:
            print("YouTube Data API v3 키를 설정해주세요!")
            print("1. Google Cloud Console에서 YouTube Data API v3를 활성화하세요")
            print("2. API 키를 생성하세요") 
            print("3. 코드의 API_KEY 변수에 키를 입력하세요")
        except UnicodeEncodeError:
            print("YouTube Data API v3 key is required!")
            print("1. Enable YouTube Data API v3 in Google Cloud Console")
            print("2. Create API key")
            print("3. Set the API_KEY variable in the code")
        return
    
    # API 키 유효성 테스트
    test_url = "https://www.googleapis.com/youtube/v3/search"
    test_params = {
        'part': 'snippet',
        'q': 'test',
        'type': 'video',
        'maxResults': 1,
        'key': API_KEY
    }
    
    try:
        test_response = requests.get(test_url, params=test_params)
        test_data = test_response.json()
        
        if 'error' in test_data:
            print(f"API 키 오류: {test_data['error']['message']}")
            print("API 키를 다시 확인해주세요.")
            return
        else:
            print("API 키가 유효합니다. 크롤링을 시작합니다...")
            
    except Exception as e:
        print(f"API 연결 테스트 중 오류: {e}")
        return
    
    # 크롤러 인스턴스 생성
    crawler = YouTubeCommentCrawler(API_KEY)
    
    # 크롤링 실행
    crawler.crawl_comments(
        channel_handle="@MBCNEWS11",
        keyword="소비쿠폰",
        output_file="mbc_consumption_coupon_comments.json",
        days_back=30,
        max_comments_per_video=50  # 좋아요 상위 50개 댓글
    )

if __name__ == "__main__":
    main()