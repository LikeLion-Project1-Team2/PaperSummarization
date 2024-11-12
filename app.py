from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file
import requests
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pymupdf
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt
import PyPDF2
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast, BartTokenizer
from dotenv import load_dotenv
import textwrap
import json
from PIL import Image
import io
from glob import glob
from markdownify import markdownify as markdown
from bs4 import BeautifulSoup
from selenium.webdriver.common.action_chains import ActionChains
import time



# .env 파일 로드
load_dotenv()

# 환경 변수에서 API 키 불러오기
api_key = os.getenv("API_KEY")

app = Flask(__name__)


# 모델 경로 설정
MODEL_PATH_KO =  "C:/Users/karen/summarization/flask_app/model_ko/kobart_finetuned"
# MODEL_PATH_EN =  "C:/Users/karen/summarization/flask_app/model_ko/kobart_finetuned"

# 모델과 토크나이저 로드
tokenizer_ko = PreTrainedTokenizerFast.from_pretrained(MODEL_PATH_KO)
model_ko = BartForConditionalGeneration.from_pretrained(MODEL_PATH_KO).to('cpu')  # 디바이스 설정 필요 시 변경
tokenizer_en = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model_en = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# 크롬 옵션 설정
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--start-maximized")

def start_driver():
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)

def extract_text_from_pdf(pdf_input):
    # pdf_input이 FileStorage 객체인지 파일 경로인지 확인
    if isinstance(pdf_input, str):  # 파일 경로인 경우
        pdf_path = pdf_input
    else:  # FileStorage 객체인 경우
        pdf_path = "temp.pdf"  # 임시 파일 경로
        pdf_input.save(pdf_path)  # 업로드된 파일을 임시 경로에 저장
    
    # PDF 파일에서 텍스트 추출
    text = ""
    with open(pdf_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text() + "\n"

    # 임시 파일 삭제 (FileStorage 객체일 때만)
    if not isinstance(pdf_input, str):
        os.remove(pdf_path)

    return text

def extract_keywords_from_pdf(pdf_file):
    # pdf 파일에서 텍스트 추출
    text = extract_text_from_pdf(pdf_file)

    # 텍스트 전처리
    cleaned_text = re.sub(r'[^\w\s]', '', text) # 특수문자 제거
    cleaned_text = re.sub(r'\d+', '', cleaned_text)  # 숫자 제거

    # 형태소 분석 및 명사 추출
    okt = Okt()
    nouns = okt.nouns(cleaned_text)
    english_words = re.findall(r'[a-zA-Z]+', cleaned_text)
    combined_nouns = ' '.join(nouns + english_words)

    # TF-IDF 벡터라이저 초기화
    stop_words = ['of', 'and', 'the', 'in', 'to', 'for', 'on', 'with', 'at', 'by', 'an', 'be', 'this', 'is', 'are', 'al', 'et', 'that', 'as', 'from', 'arxiv', 'which', 'we',
                  '이', '그', '저', '것', '수', '등', '들', '그리고', '그러나', '때문에']  # 필요시 추가
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=stop_words, max_features=5000)
    tfidf_matrix = vectorizer.fit_transform([combined_nouns])

    # TF-IDF 결과를 가져와서 높은 순으로 키워드 10개 추출
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_scores = tfidf_matrix.toarray().flatten()

    # TF-IDF 가중치를 기준으로 정렬
    tfidf_sorting = np.argsort(tfidf_scores)[::-1]

    # 중복된 키워드 제거 및 상위 20개 키워드 추출
    seen_keywords = set()
    keyword_scores = {}

    for idx in tfidf_sorting:
        keyword = feature_array[idx]
        if keyword not in seen_keywords:
            keyword_scores[keyword] = tfidf_scores[idx]  # 키워드와 점수 저장
            seen_keywords.add(keyword)
        if len(keyword_scores) >= 20:  # 상위 20개 키워드 추출
            break
    
    return keyword_scores


# def crawl_and_download(keyword, language):
#     driver = start_driver()
#     driver.get("https://scholar.google.co.kr/schhp?hl=ko" if language == "korean" else "https://scholar.google.com")

#     WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, "lr")))

#     # 라디오 버튼 클릭 (한국어 웹 선택 시)
#     if language == "korean":
#         radio_button = driver.find_element(By.ID, "gs_hp_lr1")
#         radio_button.click()

#     downloaded_pdfs = []

#     # 키워드로 검색
#     search_box = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "gs_hdr_tsi")))
#     search_box.clear()
#     search_box.send_keys(keyword)

#     search_btn = driver.find_element(By.ID, "gs_hdr_tsb")
#     search_btn.click()

#     WebDriverWait(driver, 10).until(EC.title_contains(keyword))

#     try:
#         pdf_links = driver.find_elements(By.XPATH, "//a[contains(@href, 'arxiv.org')]")
        
#         # 첫 번째 링크만 사용
#         if pdf_links:
#             pdf_link_url = pdf_links[0].get_attribute('href')

#         else:
#             print("해당 도메인에서 PDF 링크를 찾을 수 없습니다.")
#             return downloaded_pdfs

#         print(f"Clicked PDF link for '{keyword}'. URL: {pdf_link_url}")
        

#         # 헤더 추가
#         headers = {
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'
#         }

#         response = requests.get(pdf_link_url, headers=headers)

#         if response.status_code == 200:
#             pdf_filename = os.path.join("pdf_downloads", f"{keyword}.pdf")
#             os.makedirs("pdf_downloads", exist_ok=True)
#             with open(pdf_filename, 'wb') as pdf_file:
#                 pdf_file.write(response.content)
#             downloaded_pdfs.append({'keyword': keyword, 'pdf_link': pdf_filename})
#             print(f"Downloaded PDF for '{keyword}' to '{pdf_filename}'.")
#         else:
#             print(f"Failed to download PDF for '{keyword}'. Status code: {response.status_code}")

#     except Exception as e:
#         print(f"Error while clicking PDF link for '{keyword}': {e}")

#     driver.quit()
#     return downloaded_pdfs

def crawl_and_download(keyword, language):
    driver = webdriver.Chrome()
    
    # Google Scholar 접근
    base_url = "https://scholar.google.co.kr" if language == "korean" else "https://scholar.google.com"
    search_url = f"{base_url}/scholar?q={keyword}&hl=en&as_sdt=0,5"
    driver.get(search_url)
    
    downloaded_pdfs = []

    def find_pdf_link():
        try:
            # "fing.edu.uy" 도메인 내 PDF 링크를 찾음
            pdf_link = driver.find_element(By.XPATH, "//a[contains(@href, 'fing.edu.uy')]/span[contains(text(), '[PDF]')]/..")
            return pdf_link.get_attribute('href')
        except Exception as e:
            print("PDF 링크를 찾을 수 없습니다.", e)
            return None
    
    pdf_link_url = find_pdf_link()
    
    # 첫 페이지에서 PDF 링크를 찾지 못하면 두 번째 페이지로 이동
    if not pdf_link_url:
        try:
            # 두 번째 페이지로 이동
            second_page_url = f"{base_url}/scholar?start=10&q={keyword}&hl=en&as_sdt=0,5"
            driver.get(second_page_url)
            time.sleep(3)  # 페이지 로딩 대기
            
            # 두 번째 페이지에서 다시 PDF 링크 찾기
            pdf_link_url = find_pdf_link()
        except Exception as e:
            print("두 번째 페이지로 이동할 수 없습니다.", e)
    
    if pdf_link_url:
        print(f"PDF 링크를 찾았습니다: {pdf_link_url}")
        
        # 헤더 설정 및 PDF 파일 다운로드
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'
        }
        response = requests.get(pdf_link_url, headers=headers)
        
        if response.status_code == 200:
            os.makedirs("pdf_downloads", exist_ok=True)
            pdf_filename = os.path.join("pdf_downloads", f"{keyword}.pdf")
            with open(pdf_filename, 'wb') as pdf_file:
                pdf_file.write(response.content)
            downloaded_pdfs.append({'keyword': keyword, 'pdf_link': pdf_filename})
            print(f"PDF 다운로드 완료: {pdf_filename}")
        else:
            print(f"PDF 다운로드 실패. 상태 코드: {response.status_code}")
    else:
        print("PDF 링크를 찾을 수 없습니다.")
    
    driver.quit()
    return downloaded_pdfs

def summarize_text(input_text, model, tokenizer, language):

    if language == 'korean':
        # 텍스트를 일정한 길이로 분할
        text_parts = [input_text[i:i+400] for i in range(0, len(input_text), 400)]
        
        # 각 부분 요약 생성
        summaries = []
        for part in text_parts:
            inputs = tokenizer(part, max_length=512, truncation=True, return_tensors='pt')
            summary_ids = model.generate(
                inputs['input_ids'], 
                num_beams=10, 
                max_length=200, 
                min_length=100, 
                length_penalty=2.0, 
                early_stopping=True
            )
            part_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(part_summary)
        
        # 부분 요약을 결합하여 최종 요약 생성
        combined_summary = " ".join(summaries)
        inputs = tokenizer(combined_summary, max_length=512, truncation=True, return_tensors='pt')
        final_summary_ids = model.generate(
            inputs['input_ids'], 
            num_beams=8, 
            max_length=150, 
            min_length=80, 
            length_penalty=2.0, 
            early_stopping=True
        )
        final_summary = tokenizer.decode(final_summary_ids[0], skip_special_tokens=True)
        
        return final_summary

    else:
        # 영어 텍스트는 전체를 한 번에 요약
        inputs = tokenizer(input_text, max_length=1024, truncation=True, return_tensors='pt')
        summary_ids = model.generate(
            inputs['input_ids'], 
            num_beams=4, 
            max_length=250, 
            min_length=100, 
            length_penalty=1.0, 
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary


def process_pdf_and_get_summary(pdf_path, language):
    # 1. PDF 경로에서 텍스트 추출
    processor = PDFProcessor(pdf_path, api_key)
    text = processor.process_pdf() 
    
    # 2. 텍스트 전처리
    processed_text = preprocess_text(text, language)
    print("Processed Text: ", processed_text)

    # 3. 요약 모델 선택 (언어에 따라 한국어 또는 영어 모델 사용)
    if language == "korean":
        model = model_ko
        tokenizer = tokenizer_ko
    else:
        model = model_en
        tokenizer = tokenizer_en

    # 4. 요약 생성
    summary = summarize_text(processed_text, model, tokenizer, language)

    return summary


class PDFProcessor:
    def __init__(self, pdf_file, api_key):
        self.pdf_file = pdf_file
        self.api_key = api_key
        self.output_folder = os.path.join("pdf_downloads", os.path.splitext(os.path.basename(pdf_file))[0])
        
        print("Output folder path:", self.output_folder)

        self.split_files = []
        self.json_files = sorted(glob(os.path.join(self.output_folder, "*.json")))
        self.filename = os.path.splitext(os.path.basename(pdf_file))[0]


    def split_pdf(self, batch_size=1):
        input_pdf = pymupdf.open(self.pdf_file)
        num_pages = len(input_pdf)
        os.makedirs(self.output_folder, exist_ok=True)

        for start_page in range(0, num_pages, batch_size):
            end_page = min(start_page + batch_size, num_pages) - 1
            output_file = f"{self.output_folder}/{os.path.basename(self.pdf_file)}_{start_page:04d}_{end_page:04d}.pdf"
            
            with pymupdf.open() as output_pdf:
                output_pdf.insert_pdf(input_pdf, from_page=start_page, to_page=end_page)
                output_pdf.save(output_file)
                self.split_files.append(output_file)
                print(f"Split PDF created: {output_file}")
        input_pdf.close()
        return self.split_files

    def analyze_layout(self):
        for file in self.split_files:
            response = requests.post(
                "https://api.upstage.ai/v1/document-ai/layout-analysis",
                headers={"Authorization": f"Bearer {self.api_key}"},
                data={"ocr": False},
                files={"document": open(file, "rb")},
            )
            if response.status_code == 200:
                output_file = os.path.splitext(file)[0] + ".json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(response.json(), f, ensure_ascii=False)
                print(f"JSON file created: {output_file}")
            else:
                print(f"Failed to analyze layout for {file}, status_code: {response.status_code}")

    def extract_images(self):
        pdf_document = pymupdf.open(self.pdf_file)
        image_count = 0
        for page_number in range(len(pdf_document)):
            page = pdf_document[page_number]
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                image_filename = f"{self.output_folder}/image_page{page_number + 1}_img{img_index + 1}.png"
                image.save(image_filename)
                image_count += 1
        pdf_document.close()

    def generate_html_and_markdown(self):
        figure_count = {}
        html_content = []

        # JSON 파일 목록을 처리
        for json_file in self.json_files:
            print(f"Attempting to load JSON file: {json_file}")

            print(f"Processing JSON file: {json_file}")
            json_data = self._load_json(json_file)
            page_sizes = self._get_page_sizes(json_data)
                    # JSON 데이터가 제대로 로드되었는지 확인

            if json_data:
                print(f"Loaded JSON data from {json_file}: {json_data}")
            else:
                print(f"Failed to load JSON data from {json_file}")

            page_range = os.path.basename(json_file).split("_")[1:]
            start_page = int(page_range[0])

            # 각 요소를 처리하여 HTML 콘텐츠에 추가
            for element in json_data["elements"]:
                print(f"Processing element with category: {element['category']}")
                
                if element["category"] == "figure":
                    # 이미지에 대한 설명으로 대체
                    page_num = start_page + element["page"]
                    if page_num not in figure_count:
                        figure_count[page_num] = 1
                    else:
                        figure_count[page_num] += 1
                    placeholder_text = f"<p>[Image: Figure {figure_count[page_num]} on page {page_num}]</p>"
                    html_content.append(placeholder_text)

                elif element["category"] in ["paragraph", "heading1", "heading2", "heading3"]:
                    # 텍스트 요소는 그대로 추가
                    soup = BeautifulSoup(element["html"], "html.parser")
                    text_content = soup.get_text()
                    html_content.append(f"<p>{text_content}</p>")

        # HTML 파일 생성
        html_output_file = os.path.join(self.output_folder, f"{self.filename}.html")
        combined_html_content = "\n".join(html_content)
        with open(html_output_file, "w", encoding="utf-8") as f:
            f.write(combined_html_content)
        print(f"HTML file created: {html_output_file}")

        # Markdown 파일 생성
        md_output_file = os.path.join(self.output_folder, f"{self.filename}.md")
        md_output = markdown(combined_html_content)
        with open(md_output_file, "w", encoding="utf-8") as f:
            f.write(md_output)
        print(f"Markdown file created: {md_output_file}")

        return md_output_file

    def markdown_to_text(self, md_file_path):
        if md_file_path is None:
            print("Markdown file path is None, cannot convert to text.")
            return None
            
        txt_output_file = os.path.splitext(md_file_path)[0] + ".txt"
        with open(md_file_path, "r", encoding="utf-8") as md_file:
            content = md_file.read()
        
        with open(txt_output_file, "w", encoding="utf-8") as txt_file:
            txt_file.write(content)
        
        print(f"Text file created: {txt_output_file}")
        return txt_output_file

    def process_pdf(self):
        print("Starting PDF split...")
        self.split_pdf()
        print("PDF split completed.")

        print("Starting layout analysis...")
        self.analyze_layout()
        print("Layout analysis completed.")

        print("Starting image extraction...")
        self.extract_images()
        print("Image extraction completed.")
        
        # HTML 및 Markdown 생성 후 Markdown을 텍스트 파일로 변환
        print("Attempting to call generate_html_and_markdown")
        md_file_path = self.generate_html_and_markdown()
        if md_file_path is None:
            print("Failed to generate Markdown file. Exiting process.")
            return None

        txt_file_path = self.markdown_to_text(md_file_path)
        if txt_file_path is None:
            print("Failed to generate text file from Markdown. Exiting process.")
            return None
        print("Successfully called generate_html_and_markdown")
        
        # 텍스트 파일의 내용을 읽어 text 변수로 반환
        with open(txt_file_path, "r", encoding="utf-8") as txt_file:
            text = txt_file.read()
        
        return text

    def pdf_to_image(self, page_num, dpi=400):
        with pymupdf.open(self.pdf_file) as doc:
            page = doc[page_num - 1].get_pixmap(dpi=dpi)
            target_page_size = [page.width, page.height]
            page_img = Image.frombytes("RGB", target_page_size, page.samples)
        return page_img

    @staticmethod
    def normalize_coordinates(coordinates, output_page_size):
        x_values = [coord["x"] for coord in coordinates]
        y_values = [coord["y"] for coord in coordinates]
        x1, y1, x2, y2 = min(x_values), min(y_values), max(x_values), max(y_values)
        return (
            x1 / output_page_size[0],
            y1 / output_page_size[1],
            x2 / output_page_size[0],
            y2 / output_page_size[1],
        )

    @staticmethod
    def crop_image(img, coordinates, output_file):
        img_width, img_height = img.size
        x1, y1, x2, y2 = [
            int(coord * dim) for coord, dim in zip(coordinates, [img_width, img_height] * 2)
        ]
        cropped_img = img.crop((x1, y1, x2, y2))
        cropped_img.save(output_file)

    @staticmethod
    def _load_json(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _get_page_sizes(json_data):
        page_sizes = {}
        for page_element in json_data["metadata"]["pages"]:
            width = page_element["width"]
            height = page_element["height"]
            page_num = page_element["page"]
            page_sizes[page_num] = [width, height]
        return page_sizes
    
    def process_pdf(self):
        # PDF 분할, 레이아웃 분석, 이미지 추출, HTML 및 Markdown 파일 생성
        self.split_pdf()
        self.analyze_layout()
        self.extract_images()
        
        # HTML 및 Markdown 생성 후 Markdown을 텍스트 파일로 변환
        md_file_path = self.generate_html_and_markdown()
        txt_file_path = self.markdown_to_text(md_file_path)
        
        # 텍스트 파일의 내용을 읽어 text 변수로 반환
        with open(txt_file_path, "r", encoding="utf-8") as txt_file:
            text = txt_file.read()
        
        return text


def preprocess_text(raw_text, language):
    cleaned_text = re.sub(r'\[.*?\]\(.*?\)', '', raw_text)  # 링크 제거
    cleaned_text = re.sub(r'#+\s?', '', cleaned_text)  # 헤더 제거
    cleaned_text = re.sub(r'\*+', '', cleaned_text)  # 리스트 마커 제거
    cleaned_text = re.sub(r'[!#$]+', '', cleaned_text)  # 불필요한 특수문자 제거
    cleaned_text = re.sub(r'\n{2,}', '\n', cleaned_text)  # 연속 개행 제거
    cleaned_text = re.sub(r'\\-', '', cleaned_text)  # `\-` 제거

    processed_text = cleaned_text

    if language == "korean":
        processed_text = re.sub(r'(?<![가-힣])\b[A-Za-z]+\b(?![가-힣])', '', cleaned_text)
        processed_text = re.sub(r'\\[A-Za-z0-9]+', '', processed_text)  # 숫자와 특수문자 뒤 이스케이프 문자 제거
        processed_text = re.sub(r'\(\d+\)', '', processed_text)  # 논문 레퍼런스 번호 제거
        processed_text = re.sub(r'[\\=:;@+*]', '', processed_text)  # 기타 특수 문자 제거
        processed_text = re.sub(r'\s+', ' ', processed_text)  # 다중 공백 제거
        processed_text = re.sub(r'=+', '', processed_text)  # 중복되는 "=" 제거
        processed_text = re.sub(r"(Fig\.|Table)\s?\d.*", '', processed_text)  # "Fig.1" 및 "Table1" 제거

        sentences = re.split(r'(?<=[.?!])\s+', processed_text)

        # 한국어 문장 사이의 불필요한 영어 단어 제거
        def remove_english_between_korean(sentences):
            cleaned_sentences = []
            for sentence in sentences:
                cleaned = re.sub(r'(?<=[가-힣])\s*[A-Za-z]+\s*(?=[가-힣])', '', sentence)
                cleaned = re.sub(r'^[A-Za-z\s.,()]+(?=[가-힣])', '', cleaned)  # 문장 시작의 영어 제거
                cleaned = re.sub(r'(?<=[가-힣])\s*[A-Za-z\s.,()]+$', '', cleaned)  # 문장 끝의 영어 제거
                cleaned = cleaned.strip()  # 빈 줄이나 공백만 있는 경우 제거
                if cleaned:
                    cleaned_sentences.append(cleaned)
            return cleaned_sentences

        # 한국어 문장이 많은 경우를 확인하는 함수
        def is_korean_sentence(sentence):
            korean_count = len(re.findall('[가-힣]', sentence))
            english_count = len(re.findall('[a-zA-Z]', sentence))
            return korean_count > english_count

        # 불필요한 영어 문장 제거 함수
        def remove_english_between_korean2(sentences):
            cleaned_sentences = []
            for i in range(len(sentences)):
                current = sentences[i].strip()
                if not current:
                    continue
                prev_is_korean = is_korean_sentence(sentences[i-1]) if i > 0 else False
                next_is_korean = is_korean_sentence(sentences[i+1]) if i < len(sentences) - 1 else False
                if is_korean_sentence(current) or (prev_is_korean and next_is_korean):
                    cleaned_sentences.append(current)
            return cleaned_sentences

        # 1차와 2차로 영어 단어 및 문장 제거
        processed_sentences = remove_english_between_korean(sentences)
        processed_sentences = remove_english_between_korean2(processed_sentences)

        # 최종 전처리 텍스트 생성
        final_text = ' '.join(processed_sentences)
        wrapped_text = textwrap.fill(final_text, width=80)

        return wrapped_text

    elif language == "english":
        # processed_text = re.sub(r'\(\d+\)', '', cleaned_text)  # 논문 인용 번호 제거
        # processed_text = re.sub(r'\[.*?\]', '', processed_text)  # 괄호 안 텍스트 제거 (주로 인용 처리)
        # processed_text = re.sub(r'(Fig\.|Table|Equation)\s?\d+', '', processed_text)  # "Fig.1" 등 제거
        # processed_text = re.sub(r'https?://\S+|www\.\S+', '', processed_text)  # URL 제거
        # processed_text = re.sub(r'\s+', ' ', processed_text)  # 다중 공백 제거

        return cleaned_text

    return processed_text

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/extractkey', methods=['POST'])
def extractkey():
    if 'file' not in request.files:
        return jsonify({'error': '파일이 없습니다.'}), 400

    pdf_file = request.files['file']
    language = request.form.get('language')
    
    # PDF 파일에서 키워드 추출
    keyword_scores = extract_keywords_from_pdf(pdf_file)
    print(f"Extracted keywords: {keyword_scores}")

    keywords = list(keyword_scores.keys())
    scores = [str(score) for score in keyword_scores.values()]

    # 키워드 선택 페이지로 리다이렉트
    return redirect(url_for('keywords_page', keywords=','.join(keywords), scores=','.join(scores), language=language))


@app.route('/keywords', methods=['GET'])
def keywords_page():
    keywords = request.args.get('keywords').split(',') # 단순하게 URL 쿼리 파라미터로 키워드를 받아오는 방식
    scores = request.args.get('scores').split(',')
    language = request.args.get('language')

    keyword_scores = {keyword: float(score) for keyword, score in zip(keywords, scores)}
    
    return render_template('keywords.html', keyword_scores=keyword_scores, language=language)


@app.route('/crawl', methods=['GET'])
def crawl():
    keyword = request.args.get('keyword')
    language = request.args.get('language')

    # 크롤링 및 다운로드 로직 호출
    downloaded_pdf = crawl_and_download(keyword, language)

    pdf_path = downloaded_pdf[0]['pdf_link'] if downloaded_pdf else None
    
    if pdf_path:
        summary = process_pdf_and_get_summary(pdf_path, language)
    else:
        summary = "다운로드된 PDF가 없습니다."

    filename = os.path.basename(pdf_path) if pdf_path else None
    return render_template('summary.html', pdf_link=url_for('download_file', filename=filename), summary=summary)

@app.route('/download/<filename>')
def download_file(filename):
    pdf_path = os.path.join('pdf_downloads', filename)
    if os.path.exists(pdf_path):
        return send_file(pdf_path, as_attachment=True)
    else:
        return "파일을 찾을 수 없습니다.", 404

if __name__ == '__main__':
    if not os.path.exists('pdf_downloads'):
        os.makedirs('pdf_downloads')
    
    app.run(debug=True, host='0.0.0.0', port=5000)






