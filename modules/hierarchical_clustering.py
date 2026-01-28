"""
Hierarchical Semantic Clustering Module (Module 6)

PURPOSE:
    This module builds a HIERARCHICAL TREE structure of semantic topics.
    Instead of flat clusters, we create multi-level abstraction:
    - Level 0 (leaves): Fine-grained entity clusters
    - Level 1: Grouped topics with summaries
    - Level 2: Higher-level themes
    - Level N (root): Single unified view of all knowledge

ALGORITHM (RAPTOR-inspired):
    1. Start with base clusters from Module 5 (Louvain community detection)
    2. Generate summary/description for each cluster using LLM
    3. Create "super-nodes" representing each cluster
    4. Cluster the super-nodes to form higher-level groups
    5. Generate summaries for higher-level clusters
    6. Repeat until reaching root (or max depth)

RETRIEVAL STRATEGY:
    1. Query enters at root
    2. Compute similarity with cluster summaries at current level
    3. Navigate to most similar cluster(s)
    4. Repeat until reaching leaf level or threshold
    5. Return entities from selected leaf clusters

ADVANTAGES OVER FLAT CLUSTERING:
    - Multi-scale understanding (broad overview → fine details)
    - Efficient retrieval via tree traversal (log complexity)
    - Reduces noise by hierarchical filtering
    - Supports cross-granularity reasoning
    - Better context for LLM-based Q&A

OUTPUT:
    hierarchical_semantic_model.json containing:
    - tree: Nested tree structure with summaries at each level
    - leaf_topics: Original flat topics (from Module 5)
    - retrieval_index: Embeddings for each level for fast retrieval
"""
import networkx as nx
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
import json
import os
import sys

# Handle imports for both module and standalone execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openai
from utils import get_logger, save_json
import config

logger = get_logger(__name__)



class HierarchicalSemanticTree:
    """
    Build a hierarchical tree of semantic topics using recursive clustering.
    
    This implements a RAPTOR-like approach:
    1. Cluster entities → generate summaries → cluster summaries → repeat
    2. Creates multi-level abstraction for efficient retrieval
    """
    
    def __init__(self, 
                 min_cluster_size: int = 2,
                 max_depth: int = 4,
                 similarity_threshold: float = 0.7,
                 use_llm_summaries: bool = True):
        """
        Initialize hierarchical tree builder.
        
        Args:
            min_cluster_size: Minimum clusters at each level before stopping
            max_depth: Maximum tree depth (prevents infinite recursion)
            similarity_threshold: Threshold for hierarchical clustering
            use_llm_summaries: Whether to use LLM for generating summaries
        """
        self.min_cluster_size = min_cluster_size
        self.max_depth = max_depth
        self.similarity_threshold = similarity_threshold
        self.use_llm_summaries = use_llm_summaries
        
        # Load embedding model for cluster similarity
        logger.info("Loading sentence transformer for cluster embeddings...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Tree structure
        self.tree = {}  # level -> list of clusters
        self.tree_nodes = {}  # node_id -> node_data
        
        logger.info(f"HierarchicalSemanticTree initialized (max_depth={max_depth})")
    
    def build_tree(self, 
                   semantic_model: Dict,
                   graph: nx.DiGraph) -> Dict:
        """
        Build hierarchical tree from flat semantic model.
        
        Algorithm:
        1. Level 0: Use topics from Module 5 as leaf clusters
        2. Generate summary embedding for each topic
        3. Cluster topics based on summary similarity
        4. Generate higher-level summaries
        5. Repeat until reaching root or max_depth
        
        Args:
            semantic_model: Output from Module 5 (flat topics)
            graph: Enhanced knowledge graph
            
        Returns:
            Hierarchical tree structure
        """
        logger.info("Building hierarchical semantic tree...")
        
        # Level 0: Leaf nodes (original topics from Module 5)
        leaf_topics = semantic_model.get('semantic_topics', [])
        logger.info(f"Starting with {len(leaf_topics)} leaf topics")
        
        if len(leaf_topics) == 0:
            logger.error("No topics found in semantic model!")
            return {'error': 'No topics found'}
        
        # Initialize tree structure
        self.tree = {0: []}
        self.tree_nodes = {}
        
        # Create leaf nodes (Level 0)
        for topic in leaf_topics:
            node_id = topic['topic_id']
            
            # Generate summary text for embedding
            summary_text = self._generate_topic_summary_text(topic, graph)
            
            # Generate embedding for this topic
            embedding = self.embedding_model.encode(summary_text)
            
            node_data = {
                'id': node_id,
                'level': 0,
                'type': 'leaf',
                'topic_name': topic['topic_name'],
                'summary': topic.get('description', ''),
                'entity_count': topic.get('entity_count', 0),
                'key_entities': [e['name'] for e in topic.get('key_entities', [])[:5]],
                'embedding': embedding.tolist(),
                'children': [],  # Leaf nodes have no children
                'parent': None,  # Will be set when building hierarchy
                'original_topic': topic  # Keep reference to original
            }
            
            self.tree_nodes[node_id] = node_data
            self.tree[0].append(node_id)
        
        # Build hierarchy recursively
        current_level = 0
        current_nodes = self.tree[0]
        
        while len(current_nodes) > self.min_cluster_size and current_level < self.max_depth:
            logger.info(f"Building level {current_level + 1} from {len(current_nodes)} nodes...")
            
            # Cluster current level nodes
            next_level_nodes = self._cluster_level(current_nodes, current_level)
            
            if len(next_level_nodes) == 0 or len(next_level_nodes) >= len(current_nodes):
                # No reduction, stop building
                logger.info(f"Stopping at level {current_level} (no further reduction)")
                break
            
            current_level += 1
            self.tree[current_level] = next_level_nodes
            current_nodes = next_level_nodes
            
            logger.info(f"Level {current_level}: {len(next_level_nodes)} clusters")
        
        # Create root node if multiple nodes at top level
        if len(current_nodes) > 1:
            root = self._create_root_node(current_nodes, current_level)
            current_level += 1
            self.tree[current_level] = [root['id']]
            self.tree_nodes[root['id']] = root
        
        # Build final tree structure
        hierarchy = self._build_tree_structure()
        
        logger.info(f"Hierarchical tree built: {current_level + 1} levels, {len(self.tree_nodes)} total nodes")
        
        return hierarchy
    
    def _generate_topic_summary_text(self, topic: Dict, graph: nx.DiGraph) -> str:
        """Generate text summary for a topic to create embedding."""
        parts = []
        
        # Topic name
        parts.append(topic.get('topic_name', 'Unknown Topic'))
        
        # Key entities
        key_entities = topic.get('key_entities', [])
        entity_names = [e.get('name', '') for e in key_entities[:10]]
        if entity_names:
            parts.append(f"Key concepts: {', '.join(entity_names)}")
        
        # Keywords/types
        keywords = topic.get('keywords', [])
        if keywords:
            parts.append(f"Types: {', '.join(keywords[:5])}")
        
        # Description
        description = topic.get('description', '')
        if description:
            parts.append(description)
        
        return '. '.join(parts)
    
    def _cluster_level(self, node_ids: List[str], level: int) -> List[str]:
        """
        Cluster nodes at current level to create next level.
        
        Args:
            node_ids: List of node IDs at current level
            level: Current level number
            
        Returns:
            List of new cluster node IDs for next level
        """
        if len(node_ids) <= self.min_cluster_size:
            return []
        
        # Get embeddings for all nodes
        embeddings = np.array([
            self.tree_nodes[nid]['embedding']
            for nid in node_ids
        ])
        
        # Determine number of clusters (reduce by ~half each level)
        n_clusters = max(self.min_cluster_size, len(node_ids) // 2)
        n_clusters = min(n_clusters, len(node_ids) - 1)  # Can't have more clusters than nodes
        
        # Hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',
            linkage='average'
        )
        
        labels = clustering.fit_predict(embeddings)
        
        # Group nodes by cluster
        clusters = defaultdict(list)
        for i, node_id in enumerate(node_ids):
            clusters[labels[i]].append(node_id)
        
        # Create new parent nodes for each cluster
        new_level = level + 1
        new_node_ids = []
        
        for cluster_id, child_ids in clusters.items():
            parent_node = self._create_parent_node(child_ids, cluster_id, new_level)
            self.tree_nodes[parent_node['id']] = parent_node
            new_node_ids.append(parent_node['id'])
            
            # Update children to point to parent
            for child_id in child_ids:
                self.tree_nodes[child_id]['parent'] = parent_node['id']
        
        return new_node_ids
    
    def _create_parent_node(self, child_ids: List[str], cluster_id: int, level: int) -> Dict:
        """Create a parent node from a cluster of child nodes."""
        
        # Collect info from children
        all_entities = []
        all_topic_names = []
        child_embeddings = []
        
        for child_id in child_ids:
            child = self.tree_nodes[child_id]
            all_entities.extend(child.get('key_entities', []))
            all_topic_names.append(child.get('topic_name', ''))
            child_embeddings.append(child['embedding'])
        
        # Parent embedding = average of children
        parent_embedding = np.mean(child_embeddings, axis=0)
        
        # Generate parent summary
        if self.use_llm_summaries and len(all_topic_names) > 0:
            summary = self._generate_llm_summary(all_topic_names, all_entities[:20])
        else:
            # Simple concatenation summary
            summary = f"Group containing: {', '.join(all_topic_names[:5])}"
            if len(all_topic_names) > 5:
                summary += f" and {len(all_topic_names) - 5} more"
        
        # Parent name from most common terms
        parent_name = self._generate_cluster_name(all_topic_names, all_entities)
        
        node_id = f"cluster_L{level}_{cluster_id}"
        
        return {
            'id': node_id,
            'level': level,
            'type': 'branch',
            'topic_name': parent_name,
            'summary': summary,
            'entity_count': len(set(all_entities)),
            'key_entities': list(set(all_entities))[:10],
            'embedding': parent_embedding.tolist(),
            'children': child_ids,
            'parent': None  # Will be set at next level
        }
    
    def _create_root_node(self, child_ids: List[str], level: int) -> Dict:
        """Create the root node encompassing all knowledge."""
        
        # Collect all entities and topics
        all_entities = []
        all_topic_names = []
        child_embeddings = []
        
        for child_id in child_ids:
            child = self.tree_nodes[child_id]
            all_entities.extend(child.get('key_entities', []))
            all_topic_names.append(child.get('topic_name', ''))
            child_embeddings.append(child['embedding'])
        
        # Root embedding
        root_embedding = np.mean(child_embeddings, axis=0)
        
        # Generate root summary
        if self.use_llm_summaries:
            summary = self._generate_llm_summary(
                all_topic_names, 
                all_entities[:30],
                is_root=True
            )
        else:
            summary = f"Complete knowledge base covering: {', '.join(all_topic_names[:10])}"
        
        root_id = "root"
        
        # Update children to point to root
        for child_id in child_ids:
            self.tree_nodes[child_id]['parent'] = root_id
        
        return {
            'id': root_id,
            'level': level + 1,
            'type': 'root',
            'topic_name': 'Knowledge Base Root',
            'summary': summary,
            'entity_count': len(set(all_entities)),
            'key_entities': list(set(all_entities))[:20],
            'embedding': root_embedding.tolist(),
            'children': child_ids,
            'parent': None
        }
    
    def _generate_llm_summary(self, topic_names: List[str], 
                              entities: List[str],
                              is_root: bool = False) -> str:
        """Generate summary using LLM (OpenAI)."""
        try:
            client = openai.OpenAI()
            
            if is_root:
                prompt = f"""You are summarizing an entire knowledge base.

Topics covered: {', '.join(topic_names[:15])}
Key concepts: {', '.join(entities[:25])}

Write a 2-3 sentence summary describing what this knowledge base covers. Be concise and informative."""
            else:
                prompt = f"""You are summarizing a group of related topics.

Topics in this group: {', '.join(topic_names[:10])}
Key concepts: {', '.join(entities[:15])}

Write a 1-2 sentence summary describing the common theme of these topics. Be concise."""
            
            response = client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a technical writer creating concise summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.warning(f"LLM summary failed: {e}, using fallback")
            return f"Group containing: {', '.join(topic_names[:5])}"
    
    def _generate_cluster_name(self, topic_names: List[str], entities: List[str]) -> str:
        """Generate a name for a cluster based on its contents."""
        
        # Find most common important words
        from collections import Counter
        
        words = []
        for name in topic_names:
            # Split by common separators
            parts = name.replace('-', ' ').replace('_', ' ').split()
            words.extend([p for p in parts if len(p) > 2])
        
        # Get most common
        word_counts = Counter(words)
        common_words = [w for w, c in word_counts.most_common(3)]
        
        if common_words:
            return ' & '.join(common_words[:2])
        elif topic_names:
            return topic_names[0]
        else:
            return "Cluster"
    
    def _build_tree_structure(self) -> Dict:
        """Build the final hierarchical tree structure for output."""
        
        # Find root node
        root_id = None
        max_level = max(self.tree.keys()) if self.tree else 0
        
        if max_level in self.tree and len(self.tree[max_level]) > 0:
            root_id = self.tree[max_level][0]
        
        # Build nested structure
        def build_subtree(node_id: str) -> Dict:
            node = self.tree_nodes[node_id]
            
            subtree = {
                'id': node['id'],
                'name': node['topic_name'],
                'summary': node['summary'],
                'level': node['level'],
                'type': node['type'],
                'entity_count': node['entity_count'],
                'key_entities': node['key_entities'][:5],
            }
            
            # Add children recursively
            if node['children']:
                subtree['children'] = [
                    build_subtree(child_id) 
                    for child_id in node['children']
                ]
            
            # For leaf nodes, include original topic data
            if node['type'] == 'leaf' and 'original_topic' in node:
                subtree['topic_details'] = {
                    'score': node['original_topic'].get('score', 0),
                    'keywords': node['original_topic'].get('keywords', []),
                    'documents': node['original_topic'].get('documents', []),
                    'quality': node['original_topic'].get('quality', {})
                }
            
            return subtree
        
        # Build full tree
        if root_id:
            tree_structure = build_subtree(root_id)
        else:
            # No root, return flat structure
            tree_structure = {
                'id': 'virtual_root',
                'name': 'Knowledge Base',
                'children': [
                    build_subtree(nid) for nid in self.tree.get(0, [])
                ]
            }
        
        # Create output
        output = {
            'hierarchy': tree_structure,
            'metadata': {
                'total_levels': max_level + 1,
                'total_nodes': len(self.tree_nodes),
                'leaf_count': len(self.tree.get(0, [])),
            },
            'level_index': {
                str(level): [
                    {
                        'id': nid,
                        'name': self.tree_nodes[nid]['topic_name'],
                        'embedding': self.tree_nodes[nid]['embedding']
                    }
                    for nid in node_ids
                ]
                for level, node_ids in self.tree.items()
            }
        }
        
        return output


def build_hierarchical_model(semantic_model_path: str,
                             graph_path: str,
                             output_path: str,
                             max_depth: int = 4,
                             use_llm: bool = True) -> Dict:
    """
    Build hierarchical semantic model from flat model.
    
    Args:
        semantic_model_path: Path to semantic_model_best.json
        graph_path: Path to enhanced_global_graph.json
        output_path: Path to save hierarchical model
        max_depth: Maximum tree depth
        use_llm: Whether to use LLM for summaries
        
    Returns:
        Hierarchical semantic model
    """
    logger.info(f"Building hierarchical model from {semantic_model_path}")
    
    # Load semantic model
    with open(semantic_model_path, 'r', encoding='utf-8') as f:
        semantic_model = json.load(f)
    
    # Load graph
    with open(graph_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    # Reconstruct graph
    G = nx.DiGraph()
    for node_data in graph_data['nodes']:
        node_id = node_data['id']
        G.add_node(node_id, **{k: v for k, v in node_data.items() if k != 'id'})
    for edge_data in graph_data['edges']:
        G.add_edge(edge_data['source'], edge_data['target'])
    
    # Build hierarchical tree
    tree_builder = HierarchicalSemanticTree(
        max_depth=max_depth,
        use_llm_summaries=use_llm
    )
    
    hierarchy = tree_builder.build_tree(semantic_model, G)
    
    # Combine with original model
    hierarchical_model = {
        'hierarchical_tree': hierarchy,
        'flat_topics': semantic_model.get('semantic_topics', []),
        'quality_metrics': semantic_model.get('quality_metrics', {}),
        'global_metrics': semantic_model.get('global_metrics', {})
    }
    
    # Save
    save_json(hierarchical_model, output_path)
    logger.info(f"Hierarchical model saved to {output_path}")
    
    return hierarchical_model


if __name__ == "__main__":
    # Test with existing semantic model
    semantic_model_path = os.path.join(
        config.DATA_DIR, "best_semantic_models", "semantic_model_best.json"
    )
    graph_path = os.path.join(
        config.DATA_DIR, "best_kg", "enhanced_global_graph.json"
    )
    output_path = os.path.join(
        config.DATA_DIR, "best_semantic_models", "hierarchical_semantic_model.json"
    )
    
    build_hierarchical_model(
        semantic_model_path,
        graph_path,
        output_path,
        max_depth=4,
        use_llm=True
    )
