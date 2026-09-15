FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies for pdfplumber
RUN yum install -y gcc gcc-c++ && yum clean all

# Copy requirements and install
COPY requirements-lambda.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements-lambda.txt

# Download NLTK data needed by pdf_extraction
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/opt/nltk_data'); nltk.download('punkt', download_dir='/opt/nltk_data'); nltk.download('punkt_tab', download_dir='/opt/nltk_data'); nltk.download('wordnet', download_dir='/opt/nltk_data')"
ENV NLTK_DATA=/opt/nltk_data

# Copy application code
COPY pipeline/config.py .
COPY pipeline/storage.py .
COPY pipeline/lambda_handler.py .
COPY pipeline/pdf_extraction.py .
COPY pipeline/flowsheet_extraction.py .
COPY pipeline/table_parser.py .
COPY pipeline/normalize_ids.py .
COPY pipeline/aggregate_results.py .
COPY pipeline/preprocess_handler.py .
COPY TERMS.md .

# Disable CLIP on Lambda
ENV ENABLE_CLIP=false

# Lambda handler
CMD ["lambda_handler.handler"]
