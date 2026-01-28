"""
Hierarchical Retrieval Module (Module 7)

PURPOSE:
    Navigate the hierarchical semantic tree to retrieve relevant entities
    for a given query. Uses tree traversal instead of flat vector search.

RETRIEVAL ALGORITHM:
    1. Query enters at root level
    2. Compute similarity between query and all cluster summaries at current level
    3. Select top-k most similar clusters (beam search)
    4. Navigate into selected clusters
    5. Repeat similarity check at next level
    6. Continue until:
       - Reaching leaf level (return entities)
       - Similarity drops below threshold (stop early)
    7. Return entities from selected leaf clusters

ADVANTAGES OVER FLAT RETRIEVAL:
    - Logarithmic complexity: O(log N) vs O(N) for flat search
    - Hierarchical filtering reduces noise
    - Multi-granularity results (can return at different levels)
    - More interpretable (shows traversal path)
    - Better for broad queries (returns topic summaries)

RETRIEVAL MODES:
    1. 'tree_traversal': Navigate down tree (default)
    2. 'collapsed_tree': Search all levels simultaneously
    3. 'adaptive': Start broad, zoom in based on query specificity
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Optional, Any
import json
import os
import sys

# Handle imports for both module and standalone execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_logger
import config

logger = get_logger(__name__)



class HierarchicalRetriever:
    """
    Retrieve relevant entities by navigating the hierarchical semantic tree.
    """
    
    def __init__(self, 
                 hierarchical_model: Dict,
                 similarity_threshold: float = 0.3,
                 beam_width: int = 3,
                 max_results: int = 20):
        """
        Initialize hierarchical retriever.
        
        Args:
            hierarchical_model: Output from HierarchicalSemanticTree
            similarity_threshold: Minimum similarity to continue traversal
            beam_width: Number of branches to explore at each level (beam search)
            max_results: Maximum entities to return
        """
        self.model = hierarchical_model
        self.similarity_threshold = similarity_threshold
        self.beam_width = beam_width
        self.max_results = max_results
        
        # Load embedding model
        logger.info("Loading embedding model for query encoding...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Build retrieval index from level_index
        self._build_index()
        
        logger.info(f"HierarchicalRetriever initialized (beam_width={beam_width})")
    
    def _build_index(self):
        """Build retrieval index from hierarchical model."""
        self.level_index = {}
        
        level_index_data = self.model.get('hierarchical_tree', {}).get('level_index', {})
        
        for level_str, nodes in level_index_data.items():
            level = int(level_str)
            self.level_index[level] = {
                'ids': [n['id'] for n in nodes],
                'names': [n['name'] for n in nodes],
                'embeddings': np.array([n['embedding'] for n in nodes])
            }
        
        # Get max level (root)
        self.max_level = max(self.level_index.keys()) if self.level_index else 0
        
        # Build node lookup from hierarchy
        self.node_lookup = {}
        self._build_node_lookup(self.model.get('hierarchical_tree', {}).get('hierarchy', {}))
        
        logger.info(f"Index built: {len(self.level_index)} levels, {len(self.node_lookup)} nodes")
    
    def _build_node_lookup(self, node: Dict, parent_id: Optional[str] = None):
        """Recursively build node lookup from tree hierarchy."""
        if not node:
            return
        
        node_id = node.get('id', '')
        if node_id:
            self.node_lookup[node_id] = {
                'name': node.get('name', ''),
                'summary': node.get('summary', ''),
                'level': node.get('level', 0),
                'type': node.get('type', ''),
                'entity_count': node.get('entity_count', 0),
                'key_entities': node.get('key_entities', []),
                'children': [c.get('id', '') for c in node.get('children', [])],
                'parent': parent_id,
                'topic_details': node.get('topic_details', {})
            }
        
        # Recurse into children
        for child in node.get('children', []):
            self._build_node_lookup(child, node_id)
    
    def retrieve(self, 
                 query: str, 
                 mode: str = 'tree_traversal',
                 return_path: bool = True) -> Dict:
        """
        Retrieve relevant entities for a query.
        
        Args:
            query: User query string
            mode: Retrieval mode ('tree_traversal', 'collapsed_tree', 'adaptive')
            return_path: Whether to return the traversal path
            
        Returns:
            Dict with:
            - 'entities': List of relevant entities
            - 'topics': List of relevant topics
            - 'path': Traversal path (if return_path=True)
            - 'similarity_scores': Similarity at each level
        """
        logger.info(f"Retrieving for query: '{query[:50]}...' (mode={mode})")
        
        # Encode query
        query_embedding = self.embedding_model.encode(query)
        
        if mode == 'tree_traversal':
            return self._tree_traversal_retrieve(query, query_embedding, return_path)
        elif mode == 'collapsed_tree':
            return self._collapsed_tree_retrieve(query, query_embedding)
        elif mode == 'adaptive':
            return self._adaptive_retrieve(query, query_embedding, return_path)
        else:
            logger.warning(f"Unknown mode: {mode}, using tree_traversal")
            return self._tree_traversal_retrieve(query, query_embedding, return_path)
    
    def _tree_traversal_retrieve(self, 
                                  query: str,
                                  query_embedding: np.ndarray,
                                  return_path: bool) -> Dict:
        """
        Navigate tree from root to leaves following most similar branches.
        
        Algorithm:
        1. Start at root level
        2. Compute similarity with all nodes at current level
        3. Select top-k most similar nodes (beam search)
        4. Move to children of selected nodes
        5. Repeat until reaching leaves or threshold
        """
        path = []
        current_level = self.max_level
        current_nodes = self.level_index.get(current_level, {}).get('ids', [])
        selected_entities = []
        selected_topics = []
        similarity_scores = []
        
        while current_level >= 0 and current_nodes:
            level_data = self.level_index.get(current_level, {})
            
            if 'embeddings' not in level_data or len(level_data['embeddings']) == 0:
                current_level -= 1
                continue
            
            # Filter to only current_nodes
            node_indices = [
                level_data['ids'].index(nid) 
                for nid in current_nodes 
                if nid in level_data['ids']
            ]
            
            if not node_indices:
                break
            
            embeddings = level_data['embeddings'][node_indices]
            node_ids = [level_data['ids'][i] for i in node_indices]
            
            # Compute similarities
            similarities = cosine_similarity(
                query_embedding.reshape(1, -1),
                embeddings
            )[0]
            
            # Get top-k most similar
            top_indices = np.argsort(similarities)[::-1][:self.beam_width]
            
            for idx in top_indices:
                score = similarities[idx]
                node_id = node_ids[idx]
                
                if score >= self.similarity_threshold:
                    node_info = self.node_lookup.get(node_id, {})
                    
                    path.append({
                        'level': current_level,
                        'node_id': node_id,
                        'name': node_info.get('name', ''),
                        'similarity': float(score),
                        'summary': node_info.get('summary', '')[:100]
                    })
                    
                    similarity_scores.append(float(score))
                    
                    # Collect topic info
                    if node_info.get('type') == 'leaf':
                        selected_topics.append({
                            'topic_id': node_id,
                            'name': node_info.get('name', ''),
                            'similarity': float(score),
                            'entity_count': node_info.get('entity_count', 0)
                        })
                        selected_entities.extend(node_info.get('key_entities', []))
            
            # Move to next level (children of selected nodes)
            if current_level > 0:
                next_nodes = []
                for idx in top_indices:
                    if similarities[idx] >= self.similarity_threshold:
                        node_id = node_ids[idx]
                        node_info = self.node_lookup.get(node_id, {})
                        next_nodes.extend(node_info.get('children', []))
                
                current_nodes = next_nodes
            
            current_level -= 1
        
        # Deduplicate entities
        unique_entities = list(dict.fromkeys(selected_entities))[:self.max_results]
        
        result = {
            'query': query,
            'entities': unique_entities,
            'topics': selected_topics,
            'entity_count': len(unique_entities),
            'topic_count': len(selected_topics),
            'avg_similarity': np.mean(similarity_scores) if similarity_scores else 0
        }
        
        if return_path:
            result['traversal_path'] = path
        
        logger.info(f"Retrieved {len(unique_entities)} entities, {len(selected_topics)} topics")
        
        return result
    
    def _collapsed_tree_retrieve(self,
                                  query: str,
                                  query_embedding: np.ndarray) -> Dict:
        """
        Search all levels simultaneously and combine results.
        Better for queries that match concepts at different granularities.
        """
        all_matches = []
        
        # Search each level
        for level, level_data in self.level_index.items():
            if 'embeddings' not in level_data or len(level_data['embeddings']) == 0:
                continue
            
            embeddings = level_data['embeddings']
            similarities = cosine_similarity(
                query_embedding.reshape(1, -1),
                embeddings
            )[0]
            
            for i, (node_id, score) in enumerate(zip(level_data['ids'], similarities)):
                if score >= self.similarity_threshold:
                    node_info = self.node_lookup.get(node_id, {})
                    all_matches.append({
                        'level': level,
                        'node_id': node_id,
                        'name': node_info.get('name', ''),
                        'similarity': float(score),
                        'type': node_info.get('type', ''),
                        'entities': node_info.get('key_entities', [])
                    })
        
        # Sort by similarity
        all_matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Collect entities from top matches
        entities = []
        topics = []
        
        for match in all_matches[:self.beam_width * 3]:
            entities.extend(match.get('entities', []))
            if match.get('type') == 'leaf':
                topics.append({
                    'topic_id': match['node_id'],
                    'name': match['name'],
                    'similarity': match['similarity']
                })
        
        unique_entities = list(dict.fromkeys(entities))[:self.max_results]
        
        return {
            'query': query,
            'entities': unique_entities,
            'topics': topics,
            'entity_count': len(unique_entities),
            'topic_count': len(topics),
            'all_level_matches': all_matches[:10]
        }
    
    def _adaptive_retrieve(self,
                           query: str,
                           query_embedding: np.ndarray,
                           return_path: bool) -> Dict:
        """
        Adaptive retrieval that adjusts depth based on query specificity.
        
        - Specific queries (high similarity at leaf level) → go deep
        - Broad queries (high similarity at higher levels) → stay shallow
        """
        # First, try collapsed tree to find best matching level
        level_best_scores = {}
        
        for level, level_data in self.level_index.items():
            if 'embeddings' not in level_data or len(level_data['embeddings']) == 0:
                continue
            
            similarities = cosine_similarity(
                query_embedding.reshape(1, -1),
                level_data['embeddings']
            )[0]
            
            level_best_scores[level] = max(similarities)
        
        # Find level with best match
        if not level_best_scores:
            return {'query': query, 'entities': [], 'topics': []}
        
        best_level = max(level_best_scores, key=level_best_scores.get)
        
        # If best match is at high level, query is broad → return high-level summary
        # If best match is at leaf level, query is specific → use full tree traversal
        if best_level == 0 or level_best_scores.get(0, 0) > 0.7:
            # Specific query - use tree traversal
            return self._tree_traversal_retrieve(query, query_embedding, return_path)
        else:
            # Broad query - return higher-level results
            level_data = self.level_index[best_level]
            similarities = cosine_similarity(
                query_embedding.reshape(1, -1),
                level_data['embeddings']
            )[0]
            
            top_indices = np.argsort(similarities)[::-1][:self.beam_width]
            
            topics = []
            entities = []
            
            for idx in top_indices:
                node_id = level_data['ids'][idx]
                node_info = self.node_lookup.get(node_id, {})
                
                topics.append({
                    'topic_id': node_id,
                    'name': node_info.get('name', ''),
                    'summary': node_info.get('summary', ''),
                    'similarity': float(similarities[idx]),
                    'level': best_level
                })
                entities.extend(node_info.get('key_entities', []))
            
            return {
                'query': query,
                'entities': list(dict.fromkeys(entities))[:self.max_results],
                'topics': topics,
                'matching_level': best_level,
                'query_type': 'broad'
            }
    
    def get_topic_by_id(self, topic_id: str) -> Optional[Dict]:
        """Get full topic information by ID."""
        return self.node_lookup.get(topic_id)
    
    def get_children(self, node_id: str) -> List[Dict]:
        """Get children of a node for drill-down navigation."""
        node = self.node_lookup.get(node_id, {})
        children_ids = node.get('children', [])
        
        return [
            self.node_lookup.get(cid, {})
            for cid in children_ids
        ]
    
    def explain_retrieval(self, result: Dict) -> str:
        """Generate human-readable explanation of retrieval path."""
        if 'traversal_path' not in result:
            return "No traversal path available."
        
        path = result['traversal_path']
        
        explanation = [f"Query: \"{result['query']}\"\n"]
        explanation.append("Retrieval Path:")
        
        for step in path:
            explanation.append(
                f"  Level {step['level']}: {step['name']} "
                f"(similarity: {step['similarity']:.2f})"
            )
        
        explanation.append(f"\nRetrieved {result['entity_count']} entities from {result['topic_count']} topics")
        explanation.append(f"Average similarity: {result.get('avg_similarity', 0):.2f}")
        
        return '\n'.join(explanation)


def load_retriever(hierarchical_model_path: str) -> HierarchicalRetriever:
    """
    Load hierarchical retriever from saved model.
    
    Args:
        hierarchical_model_path: Path to hierarchical_semantic_model.json
        
    Returns:
        Initialized HierarchicalRetriever
    """
    logger.info(f"Loading hierarchical model from {hierarchical_model_path}")
    
    with open(hierarchical_model_path, 'r', encoding='utf-8') as f:
        model = json.load(f)
    
    return HierarchicalRetriever(model)


if __name__ == "__main__":
    # Test retrieval
    model_path = os.path.join(
        config.DATA_DIR, "best_semantic_models", "hierarchical_semantic_model.json"
    )
    
    if os.path.exists(model_path):
        retriever = load_retriever(model_path)
        
        # Test queries
        test_queries = [
            "How does the Vision Transformer work?",
            "What is attention mechanism?",
            "Compare GPT-3 and BERT",
            "Image classification with transformers"
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            result = retriever.retrieve(query, mode='tree_traversal')
            print(retriever.explain_retrieval(result))
            print(f"Top entities: {result['entities'][:5]}")
    else:
        print(f"Hierarchical model not found at {model_path}")
        print("Run hierarchical_clustering.py first to generate the model.")
