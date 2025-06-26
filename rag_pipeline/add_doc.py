# This script is used for document parsing.

# For legal documents (found on adilet.kz) MISTRAL OCT is used. It is required to set up Mistral API key, which allows a free parsing within limits 
# To get API-key: https://console.mistral.ai/api-keys

from mistral import Mistral 
import argparse
from pathlib import Path
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import json
import re
import pymupdf
from mistralai import DocumentURLChunk, ImageURLChunk, TextChunk
from mistralai.models import OCRResponse
from langchain.schema import Document
from typing import List
import os
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

"""
ОБЯЗАТЕЛЬНЫЕ ПАРАМЕТРЫ СКРИПТА:
- pdf_path_path — путь к PDF документу
- model_type — "mistral" для приказов с zan.kz (смотреть обрзец в data/raw), "other" для остальных документов
- openai_api_key — api OpenAI для эмбеддингов
ДОПОЛНИТЕЛЬНЫЕ ПАРАМЕТРЫ:
- output — название JSON файла для сохранения сырого результата распознанного текста. В противном случае он не будет сохранен 
- mistral_api_key — API ключ от MISTRAL (взять из https://console.mistral.ai/api-keys), с лимитом, но бесплатно
- vectorstore — путь до папки с векторной базой для RAG. Если задан, то новые документы добавляются в существующую базу
"""
# парсинг аргументов
parser = argparse.ArgumentParser(description="Document Parsing Script")
# обязательные
parser.add_argument("pdf_path", help="Путь к PDF файлу")
parser.add_argument('model_type', choices=['mistral', 'other'],
                    help='Тип модели: "mistral" для приказов с zan.kz (см. образец в data/raw), "other" для остальных документов'
)
parser.add_argument('openai_api_key', help="OpenAI api key для эмбеддингов")
# дополнительные
parser.add_argument("--output", help="Имя JSON-файла для сохранения распознанного текста")
parser.add_argument("--mistral_api_key", help="Mistral API key, если mode=mistral")
parser.add_argument("--vectorstore", help="Папка с расположением базы данных FAISS (для векторного поиска)")

args = parser.parse_args()

os.environ["OPENAI_API_KEY"] = args.openai_api_key
# проверка есть ли PDF
pdf_file = Path(args.pdf_path)
assert pdf_file.is_file()

# дополнительные функции для обработки текста
def split_kazakh_russian_text(text: str) -> List[Document]:
    """
    Функция для разбивки текста закона на сепараторы (параграфы и главы) на казахском и на русскомю 
    
    Args:
        text (str): распознанный текст
    
    Returns:
        List[Document]: лист объектов LangChain Document для дальнейшего заполнения базы
    """
    
    documents = []

    # regex для разбивки 
    split_pattern = r'(?=(?:\d+[-\s]*(?:тарау|параграф|приложение|қосымша)|(?:тарау|параграф|қосымша|Глава|Параграф|Приложение)[-\s]*\d+|#\s*Приложение\s+\d+))'

    sections = re.split(split_pattern, text)
    
    for idx, section in enumerate(sections):
        if not section.strip():
            continue
            
        content = section.strip()
        metadata = {
        }
        
        patterns_metadata = [
            (r'(\d+)[-\s]*тарау', 'chapter', 'kk'),
            (r'(\d+)[-\s]*параграф', 'paragraph', 'kk'), 
            (r'(\d+)[-\s]*қосымша', 'appendix', 'kk'),
            (r'тарау[-\s]*(\d+)', 'chapter', 'kk'),
            (r'параграф[-\s]*(\d+)', 'paragraph', 'ru'),
            (r'қосымша[-\s]*(\d+)', 'appendix', 'ru'),
            (r'Глава[-\s]*(\d+)', 'chapter', 'ru'),
            (r'Параграф[-\s]*(\d+)', 'paragraph', 'ru'),
            (r'(\d+)[-\s]*приложение', 'appendix', 'ru'),
            (r'#\s*Приложение\s+(\d+)', 'appendix', 'ru'),
        ]
        
        matched = False
        for pattern, content_type, language in patterns_metadata:
            match = re.match(pattern, content, re.IGNORECASE)
            if match:
                metadata.update({
                'source': args.pdf_path,
                })
                matched = True
                break
        
        if not matched:
            metadata['content_type'] = 'content'
        
        doc = Document(
            page_content=content,
            metadata=metadata
        )
        documents.append(doc)
    
    return documents

def is_separator_only(chunk: str) -> bool:
        """Проверка на наличие одного из разделителей текста"""
        separator_patterns = [
            r'^Глава\s+\d+$',
            r'^Параграф\s+\d+$',
            r'^\d+-тарау$',
            r'^\d+-параграф$',
            r'^\d+-қосымша$',
            r'^тарау\s+\d+$',
            r'^параграф\s+\d+$',
            r'^қосымша\s+\d+$',
            r'^#\s*Приложение\s+\d+$',
            r'^\d+-приложение$',
        ]
        
        for pattern in separator_patterns:
            if re.match(pattern, chunk.strip(), re.IGNORECASE):
                return True
        return False

# если был указан mistral_api_key — для парсинга файла используется Mistral OCR
if args.mistral_api_key is not None and args.model_type=="mistral":
    client = Mistral(api_key=args.mistral_api_key)
    uploaded_file = client.files.upload(
        file = {"file_name": pdf_file.stem, "content": pdf_file.read_bytes()},
        purpose="ocr"
    )
    signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
    pdf_response = client.ocr.process(
        document=DocumentURLChunk(document_url=signed_url.url),
        model="mistral-ocr-latest",
        include_image_base64=True
        )
    response_dict = json.loads(pdf_response.model_dump_json())
    if args.output is not None:
        with open(args.output, 'w') as f:
            json.dump(response_dict, f)
    
    # сборка и предобработка текста 
    full_text = " ".join(i['markdown'] for i in response_dict['pages'])
    pattern = r'!\[img-\d+\.jpeg\]\(img-\d+\.jpeg\)'
    langchain_docs = split_kazakh_russian_text(re.sub(pattern, '', full_text))

    # очистка листа от документов, где только текст сепараторов
    new_doc = []
    for doc in langchain_docs:
        if is_separator_only(doc.page_content) or doc.page_content=="#":
            continue
        else:
            new_doc.append(doc)
elif args.mistral_api_key is None and args.model_type=="mistral":
    raise ValueError("Provide Mistral API key!")
else:
    # pymupdf, если не MISTRAL OCR
    langchain_docs = []
    pdf_doc = pymupdf.open(pdf_file)
    for page_num in range(pdf_doc.page_count):
        page = pdf_doc[page_num]
        text = page.get_text()
        if text.strip():
                # создание LangChain Document
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": args.pdf_path,
                    }
                )
                langchain_docs.append(doc)
    pdf_doc.close()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100,
    )
    new_doc = []
    for doc in langchain_docs:
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            chunked_doc = Document(
                page_content=chunk,
                metadata={
                    **doc.metadata,
                    "chunk": i + 1,
                    "total_chunks_on_page": len(chunks)
                }
            )
            new_doc.append(chunked_doc)

if args.vectorstore is not None and Path(args.vectorstore).exists():
    # Создание новой базы данных и добавление документов батчами
    db = FAISS.load_local(args.vectorstore, embeddings=OpenAIEmbeddings(model="text-embedding-3-large"))
    for i in range(0, len(new_doc), 10):
        batch_docs = new_doc[i:min(i + 10, len(new_doc))]
        db.add_documents(batch_docs)
    db.save_local(args.vectorstore)
else:
    # создание новой базы
    db = FAISS.from_documents(new_doc, embeddings=OpenAIEmbeddings(model="text-embedding-3-large"))
    save_path = args.vectorstore if args.vectorstore else "faiss_vectorstore"
    db.save_local(save_path)