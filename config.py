"""
Configuration file for the semantic PDF model pipeline
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model Configuration
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Directory Paths
DATA_DIR = "data"
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
PDF_EXTRACT_DIR = os.path.join(DATA_DIR, "pdf_extracts")  # Module 1 output
TEXT_STRUCTURE_DIR = os.path.join(DATA_DIR, "text_structures")  # Module 2 output
JSON_DIR = os.path.join(DATA_DIR, "json")  # Legacy support
KG_DIR = os.path.join(DATA_DIR, "kg")
SEMANTIC_MODEL_DIR = os.path.join(DATA_DIR, "semantic_models")

# Processing Configuration
CHUNK_SIZE = 2000  # Characters per chunk for long documents
OVERLAP = 200      # Overlap between chunks

# KG Generation Parameters
MAX_ENTITIES_PER_CHUNK = 50
MAX_RELATIONS_PER_CHUNK = 100

# Semantic Model Parameters
MIN_CLUSTER_SIZE = 3
MIN_SAMPLES = 2

# Enhanced Pipeline Parameters
# Module 2: Entity Extraction
SPACY_MODEL = "en_core_web_trf"  # Transformer-based spaCy model

# Module 3: Global Graph
FUZZY_MATCHING_THRESHOLD = 85  # 0-100, for entity resolution
EMBEDDING_SIMILARITY_THRESHOLD = 0.85  # 0-1, for semantic entity matching

# Module 4: Graph Enhancement
PAGERANK_ALPHA = 0.85  # Damping factor for PageRank
CENTRALITY_COMPUTE = True  # Compute centrality metrics

# Module 5: Semantic Clustering
CLUSTERING_METHOD = "community"  # Options: 'community', 'spectral', 'node2vec', 'ensemble'
NODE2VEC_DIMENSIONS = 128  # Embedding dimensions for Node2Vec
NODE2VEC_WALK_LENGTH = 30
NODE2VEC_NUM_WALKS = 200

# Quality Thresholds
MIN_MODULARITY = 0.5  # Minimum acceptable modularity score
MIN_COVERAGE = 0.8  # Minimum % of entities that should be clustered
