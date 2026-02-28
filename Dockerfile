FROM python:3.12-slim

# LightGBM needs OpenMP; git for SHA capture
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch CPU-only first (keeps image small ~200MB vs ~2GB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (skip torch — already installed CPU-only above)
COPY requirements.txt .
RUN grep -v '^torch' requirements.txt > requirements_no_torch.txt && \
    pip install --no-cache-dir -r requirements_no_torch.txt && \
    rm requirements_no_torch.txt

# Copy processed data (~18 MB)
COPY data/processed/ data/processed/

# Copy source code, configs, and package definition
COPY configs/ configs/
COPY src/ src/
COPY pyproject.toml .

# Install package in editable mode
RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "src.models.train", "--save-artifacts", "--artifact-dir", "/output/artifacts"]
