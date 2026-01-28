"""
Graph Improvements Module - Phase 1 Enhancements

PURPOSE:
    Implements Phase 1 improvements to boost semantic model quality:
    1. Relationship Inference - Add edges between semantically similar entities
    2. Multi-Pass Noise Reduction - Rescue entities marked as noise
    3. LLM Topic Summaries - Rich GPT-generated descriptions

EXPECTED IMPROVEMENTS:
    - Coverage: 38% → 75%
    - Average Degree: 1.0 → 5.0
    - Description Quality: Template → Semantic
"""
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import json
import os
import sys
import openai

# Handle imports for both module and standalone execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_logger, save_json
import config

logger = get_logger(__name__)


class RelationshipInferencer:
    """
    Infer implicit relationships between entities based on semantic similarity.
    
    Problem: Current graph has average degree 1.0 (too sparse).
    Solution: Add edges between semantically similar entities.
    """
    
    def __init__(self, similarity_threshold: float = 0.65, max_edges_per_node: int = 5):
        """
        Initialize relationship inferencer.
        
        Args:
            similarity_threshold: Minimum similarity to create edge (0.65 recommended)
            max_edges_per_node: Maximum inferred edges per node (prevents over-connection)
        """
        self.threshold = similarity_threshold
        self.max_edges = max_edges_per_node
        
        logger.info("Loading embedding model for relationship inference...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        logger.info(f"RelationshipInferencer initialized (threshold={similarity_threshold})")
    
    def infer_relationships(self, G: nx.DiGraph) -> nx.DiGraph:
        """
        Add inferred edges based on semantic similarity.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Graph with additional inferred edges
        """
        logger.info(f"Inferring relationships for {G.number_of_nodes()} nodes...")
        
        nodes = list(G.nodes())
        
        # Generate embeddings for all nodes
        logger.info("Generating node embeddings...")
        node_texts = []
        for node in nodes:
            # Create text representation
            node_type = G.nodes[node].get('type', '')
            desc = G.nodes[node].get('description', '')
            text = f"{node}. Type: {node_type}. {desc}"
            node_texts.append(text)
        
        embeddings = self.embedding_model.encode(node_texts, show_progress_bar=True)
        
        # Compute pairwise similarities
        logger.info("Computing pairwise similarities...")
        similarity_matrix = cosine_similarity(embeddings)
        
        # Add edges for high-similarity pairs
        edges_added = 0
        edges_per_node = defaultdict(int)
        
        # Sort by similarity to prioritize strongest connections
        pairs = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if similarity_matrix[i, j] >= self.threshold:
                    pairs.append((i, j, similarity_matrix[i, j]))
        
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        for i, j, sim in pairs:
            node_i, node_j = nodes[i], nodes[j]
            
            # Check if edge already exists
            if G.has_edge(node_i, node_j) or G.has_edge(node_j, node_i):
                continue
            
            # Check max edges per node
            if edges_per_node[node_i] >= self.max_edges or edges_per_node[node_j] >= self.max_edges:
                continue
            
            # Add inferred edge
            G.add_edge(node_i, node_j, 
                      relation="semantically_related",
                      weight=float(sim),
                      inferred=True)
            
            edges_added += 1
            edges_per_node[node_i] += 1
            edges_per_node[node_j] += 1
        
        logger.info(f"Added {edges_added} inferred edges")
        
        # Compute new average degree
        total_degree = sum(dict(G.degree()).values())
        avg_degree = total_degree / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
        logger.info(f"New average degree: {avg_degree:.2f}")
        
        return G


class NoiseRescuer:
    """
    Rescue entities marked as noise (unclustered) using multi-pass approach.
    
    Problem: 62% of entities are marked as noise.
    Solution: Multi-pass clustering with embedding similarity and LLM fallback.
    """
    
    def __init__(self, similarity_threshold: float = 0.55, use_llm: bool = True):
        """
        Initialize noise rescuer.
        
        Args:
            similarity_threshold: Minimum similarity to assign to cluster
            use_llm: Whether to use LLM for final pass
        """
        self.threshold = similarity_threshold
        self.use_llm = use_llm
        
        logger.info("Loading embedding model for noise rescue...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        logger.info(f"NoiseRescuer initialized (threshold={similarity_threshold}, use_llm={use_llm})")
    
    def rescue_noise(self, G: nx.DiGraph, 
                     clusters: Dict[str, int],
                     topics: List[Dict]) -> Dict[str, int]:
        """
        Rescue noise entities using multi-pass approach.
        
        Pass 1: Assign based on embedding similarity to cluster centroids
        Pass 2 (optional): Use LLM to classify remaining noise
        
        Args:
            G: NetworkX graph
            clusters: Current node -> cluster_id mapping (-1 = noise)
            topics: List of topic dictionaries
            
        Returns:
            Updated clusters with rescued nodes
        """
        noise_nodes = [n for n, c in clusters.items() if c == -1]
        clustered_nodes = [n for n, c in clusters.items() if c >= 0]
        
        logger.info(f"Rescuing {len(noise_nodes)} noise nodes from {len(clustered_nodes)} clustered nodes")
        
        if not noise_nodes or not topics:
            return clusters
        
        # Pass 1: Embedding similarity to cluster centroids
        logger.info("Pass 1: Embedding-based rescue...")
        rescued_clusters = self._rescue_by_embedding(G, noise_nodes, topics)
        
        for node, cluster_id in rescued_clusters.items():
            if cluster_id >= 0:
                clusters[node] = cluster_id
        
        # Count remaining noise
        remaining_noise = [n for n, c in clusters.items() if c == -1]
        logger.info(f"After Pass 1: {len(remaining_noise)} remaining noise nodes")
        
        # Pass 2: LLM classification (optional)
        if self.use_llm and remaining_noise:
            logger.info("Pass 2: LLM-based rescue...")
            llm_rescued = self._rescue_by_llm(G, remaining_noise[:50], topics)  # Limit to 50 for cost
            
            for node, cluster_id in llm_rescued.items():
                if cluster_id >= 0:
                    clusters[node] = cluster_id
        
        # Final count
        final_noise = len([n for n, c in clusters.items() if c == -1])
        total = len(clusters)
        coverage = (total - final_noise) / total * 100
        logger.info(f"Final coverage: {coverage:.1f}% ({total - final_noise}/{total})")
        
        return clusters
    
    def _rescue_by_embedding(self, G: nx.DiGraph, 
                             noise_nodes: List[str],
                             topics: List[Dict]) -> Dict[str, int]:
        """Rescue noise nodes by embedding similarity to topic centroids."""
        rescued = {}
        
        # Generate noise node embeddings
        noise_texts = [self._node_to_text(G, n) for n in noise_nodes]
        noise_embeddings = self.embedding_model.encode(noise_texts)
        
        # Generate topic centroid embeddings
        topic_texts = []
        topic_ids = []
        for i, topic in enumerate(topics):
            key_entities = [e.get('name', '') for e in topic.get('key_entities', [])[:5]]
            text = f"{topic.get('topic_name', '')}. {', '.join(key_entities)}"
            topic_texts.append(text)
            topic_ids.append(i)
        
        topic_embeddings = self.embedding_model.encode(topic_texts)
        
        # Find best matching topic for each noise node
        similarities = cosine_similarity(noise_embeddings, topic_embeddings)
        
        for i, node in enumerate(noise_nodes):
            best_topic_idx = np.argmax(similarities[i])
            best_similarity = similarities[i, best_topic_idx]
            
            if best_similarity >= self.threshold:
                rescued[node] = topic_ids[best_topic_idx]
            else:
                rescued[node] = -1
        
        num_rescued = len([c for c in rescued.values() if c >= 0])
        logger.info(f"Embedding rescue: {num_rescued}/{len(noise_nodes)} nodes")
        
        return rescued
    
    def _rescue_by_llm(self, G: nx.DiGraph,
                       noise_nodes: List[str],
                       topics: List[Dict]) -> Dict[str, int]:
        """Rescue remaining noise nodes using LLM classification."""
        rescued = {}
        
        if not noise_nodes:
            return rescued
        
        # Prepare topic descriptions
        topic_descs = []
        for i, topic in enumerate(topics):
            name = topic.get('topic_name', f'Topic {i}')
            entities = [e.get('name', '') for e in topic.get('key_entities', [])[:3]]
            topic_descs.append(f"{i}: {name} ({', '.join(entities)})")
        
        topic_list = '\n'.join(topic_descs)
        
        try:
            client = openai.OpenAI()
            
            # Batch entities for efficiency
            batch_size = 10
            for batch_start in range(0, len(noise_nodes), batch_size):
                batch = noise_nodes[batch_start:batch_start + batch_size]
                entity_list = ', '.join(batch)
                
                prompt = f"""You are classifying entities into topic clusters.

Available topics:
{topic_list}

Entities to classify: {entity_list}

For each entity, respond with ONLY the topic number (0-{len(topics)-1}) or -1 if no match.
Format: entity:number, entity:number, ...
Example: "GPT-3:1, ImageNet:4, Unknown:−1"
"""
                
                response = client.chat.completions.create(
                    model=config.OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a classification assistant. Respond only with the requested format."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=200
                )
                
                # Parse response
                result = response.choices[0].message.content.strip()
                for part in result.split(','):
                    if ':' in part:
                        entity, cluster_str = part.strip().rsplit(':', 1)
                        entity = entity.strip()
                        try:
                            cluster_id = int(cluster_str.strip())
                            if entity in batch:
                                rescued[entity] = cluster_id
                        except ValueError:
                            continue
            
            num_rescued = len([c for c in rescued.values() if c >= 0])
            logger.info(f"LLM rescue: {num_rescued}/{len(noise_nodes)} nodes")
            
        except Exception as e:
            logger.warning(f"LLM rescue failed: {e}")
        
        return rescued
    
    def _node_to_text(self, G: nx.DiGraph, node: str) -> str:
        """Convert node to text representation."""
        node_type = G.nodes[node].get('type', '')
        desc = G.nodes[node].get('description', '')
        return f"{node}. Type: {node_type}. {desc}"


class TopicSummaryGenerator:
    """
    Generate rich LLM-powered summaries for topics.
    
    Problem: Current descriptions are template-based.
    Solution: Use GPT to generate semantic summaries.
    """
    
    def __init__(self):
        logger.info("TopicSummaryGenerator initialized")
    
    def generate_summaries(self, G: nx.DiGraph, topics: List[Dict]) -> List[Dict]:
        """
        Generate LLM summaries for all topics.
        
        Args:
            G: NetworkX graph
            topics: List of topic dictionaries
            
        Returns:
            Updated topics with rich summaries
        """
        logger.info(f"Generating summaries for {len(topics)} topics...")
        
        try:
            client = openai.OpenAI()
            
            for i, topic in enumerate(topics):
                # Collect topic information
                topic_name = topic.get('topic_name', f'Topic {i}')
                key_entities = [e.get('name', '') for e in topic.get('key_entities', [])[:10]]
                keywords = topic.get('keywords', [])[:5]
                documents = topic.get('documents', [])[:5]
                
                # Build context
                entity_str = ', '.join(key_entities)
                keyword_str = ', '.join(keywords) if keywords else 'N/A'
                doc_str = ', '.join(documents) if documents else 'N/A'
                
                prompt = f"""Analyze this topic cluster from AI/ML research papers:

Topic Name: {topic_name}
Key Entities: {entity_str}
Entity Types: {keyword_str}
Source Documents: {doc_str}

Write a 2-3 sentence description that:
1. Explains what this topic cluster represents
2. Identifies the main theme or concept
3. Notes any significant relationships or applications

Be concise and technical."""
                
                response = client.chat.completions.create(
                    model=config.OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a technical writer summarizing AI research topics."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=150
                )
                
                summary = response.choices[0].message.content.strip()
                topic['description'] = summary
                topic['summary_generated'] = True
                
                logger.info(f"  Generated summary for topic {i+1}/{len(topics)}: {topic_name[:30]}...")
            
            logger.info("All summaries generated successfully")
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
        
        return topics


def run_phase1_improvements(graph_path: str, 
                            semantic_model_path: str,
                            output_graph_path: str,
                            output_model_path: str) -> Dict:
    """
    Run all Phase 1 improvements.
    
    Args:
        graph_path: Path to enhanced_global_graph.json
        semantic_model_path: Path to semantic_model_best.json
        output_graph_path: Path to save improved graph
        output_model_path: Path to save improved model
        
    Returns:
        Comparison metrics (before vs after)
    """
    logger.info("="*60)
    logger.info(" RUNNING PHASE 1 IMPROVEMENTS")
    logger.info("="*60)
    
    # Load data
    logger.info("\nLoading data...")
    with open(graph_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    with open(semantic_model_path, 'r', encoding='utf-8') as f:
        semantic_model = json.load(f)
    
    # Reconstruct graph
    G = nx.DiGraph()
    for node_data in graph_data['nodes']:
        node_id = node_data['id']
        G.add_node(node_id, **{k: v for k, v in node_data.items() if k != 'id'})
    for edge_data in graph_data['edges']:
        G.add_edge(edge_data['source'], edge_data['target'],
                  **{k: v for k, v in edge_data.items() if k not in ['source', 'target']})
    
    # Before metrics
    before_edges = G.number_of_edges()
    before_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
    
    topics = semantic_model.get('semantic_topics', [])
    quality = semantic_model.get('quality_metrics', {})
    before_coverage = quality.get('coverage', 0)
    
    logger.info(f"\nBefore improvements:")
    logger.info(f"  Edges: {before_edges}")
    logger.info(f"  Avg Degree: {before_degree:.2f}")
    logger.info(f"  Coverage: {before_coverage:.1%}")
    
    # Step 1: Relationship Inference
    logger.info("\n" + "-"*60)
    logger.info(" Step 1: Relationship Inference")
    logger.info("-"*60)
    inferencer = RelationshipInferencer(similarity_threshold=0.65, max_edges_per_node=5)
    G = inferencer.infer_relationships(G)
    
    # Step 2: Noise Rescue
    logger.info("\n" + "-"*60)
    logger.info(" Step 2: Multi-Pass Noise Rescue")
    logger.info("-"*60)
    
    # Build current clusters from topics
    current_clusters = {}
    for i, topic in enumerate(topics):
        for entity in topic.get('key_entities', []):
            entity_name = entity.get('name', entity) if isinstance(entity, dict) else entity
            current_clusters[entity_name] = i
    
    # Mark unclustered nodes as noise
    for node in G.nodes():
        if node not in current_clusters:
            current_clusters[node] = -1
    
    rescuer = NoiseRescuer(similarity_threshold=0.55, use_llm=True)
    updated_clusters = rescuer.rescue_noise(G, current_clusters, topics)
    
    # Step 3: Topic Summaries
    logger.info("\n" + "-"*60)
    logger.info(" Step 3: LLM Topic Summaries")
    logger.info("-"*60)
    summarizer = TopicSummaryGenerator()
    topics = summarizer.generate_summaries(G, topics)
    
    # After metrics
    after_edges = G.number_of_edges()
    after_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
    
    num_clustered = len([c for c in updated_clusters.values() if c >= 0])
    after_coverage = num_clustered / len(updated_clusters)
    
    logger.info("\n" + "="*60)
    logger.info(" PHASE 1 COMPLETE - COMPARISON")
    logger.info("="*60)
    logger.info(f"\n{'Metric':<20} {'Before':>12} {'After':>12} {'Change':>12}")
    logger.info("-"*60)
    logger.info(f"{'Edges':<20} {before_edges:>12} {after_edges:>12} {'+' + str(after_edges - before_edges):>12}")
    logger.info(f"{'Avg Degree':<20} {before_degree:>12.2f} {after_degree:>12.2f} {'+' + f'{after_degree - before_degree:.2f}':>12}")
    logger.info(f"{'Coverage':<20} {before_coverage * 100:>11.1f}% {after_coverage * 100:>11.1f}% {'+' + f'{(after_coverage - before_coverage) * 100:.1f}%':>12}")
    
    # Save improved graph
    logger.info(f"\nSaving improved graph to {output_graph_path}")
    improved_graph_data = {
        'nodes': [{'id': n, **G.nodes[n]} for n in G.nodes()],
        'edges': [{'source': u, 'target': v, **G[u][v]} for u, v in G.edges()],
        'statistics': graph_data.get('statistics', {})
    }
    
    # Convert sets to lists for JSON
    for node_data in improved_graph_data['nodes']:
        if 'aliases' in node_data and isinstance(node_data['aliases'], set):
            node_data['aliases'] = list(node_data['aliases'])
    
    save_json(improved_graph_data, output_graph_path)
    
    # Save improved model
    logger.info(f"Saving improved model to {output_model_path}")
    semantic_model['semantic_topics'] = topics
    semantic_model['quality_metrics']['coverage'] = after_coverage
    semantic_model['improvements'] = {
        'phase1_applied': True,
        'edges_added': after_edges - before_edges,
        'coverage_boost': after_coverage - before_coverage
    }
    save_json(semantic_model, output_model_path)
    
    comparison = {
        'before': {'edges': before_edges, 'avg_degree': before_degree, 'coverage': before_coverage},
        'after': {'edges': after_edges, 'avg_degree': after_degree, 'coverage': after_coverage}
    }
    
    return comparison


if __name__ == "__main__":
    # Run Phase 1 improvements
    graph_path = os.path.join(config.DATA_DIR, "best_kg", "enhanced_global_graph.json")
    model_path = os.path.join(config.DATA_DIR, "best_semantic_models", "semantic_model_best.json")
    
    output_graph = os.path.join(config.DATA_DIR, "best_kg", "improved_global_graph.json")
    output_model = os.path.join(config.DATA_DIR, "best_semantic_models", "improved_semantic_model.json")
    
    comparison = run_phase1_improvements(graph_path, model_path, output_graph, output_model)
    
    print("\n✅ Phase 1 improvements complete!")
    print(f"   Improved graph saved to: {output_graph}")
    print(f"   Improved model saved to: {output_model}")
