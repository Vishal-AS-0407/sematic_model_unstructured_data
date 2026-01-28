"""
Enhanced Module 4: Graph Enhancement with Centrality Metrics

PURPOSE:
    This module computes graph-theoretic metrics to identify important entities
    and their roles in the knowledge graph. These metrics are crucial for:
    - Ranking entities by importance (PageRank)
    - Identifying bridge entities (betweenness centrality)
    - Finding well-connected entities (degree, eigenvector centrality)
    - Preliminary community detection (Louvain algorithm)

WHY THIS MODULE:
    - Not all entities are equally important
    - Graph structure reveals implicit importance beyond mention counts
    - Centrality metrics help identify key concepts in the domain
    - These metrics are used in Module 5 for high-quality clustering

CENTRALITY METRICS COMPUTED:

    1. PageRank (Larry Page & Sergey Brin, 1996):
       - Measures importance based on incoming links
       - Used by Google Search to rank web pages
       - Formula: PR(A) = (1-d)/N + d * Σ(PR(Ti)/C(Ti))
       - Parameters: alpha=0.85 (damping factor), max_iter=100
       - Interpretation: Entities referenced by important entities are important

    2. Betweenness Centrality (Brandes' algorithm, 2001):
       - Measures how often an entity lies on shortest paths
       - Identifies "bridge" entities connecting different topics
       - High betweenness = important for information flow
       - Algorithm: O(VE) using breadth-first search
       - Interpretation: Entities that connect disparate concepts

    3. Eigenvector Centrality (Bonacich, 1987):
       - Measures importance based on neighbor importance
       - Similar to PageRank but undirected
       - Uses principal eigenvector of adjacency matrix
       - Interpretation: Entities connected to important neighbors

    4. Degree Centrality:
       - Simple count of connections (in-degree + out-degree)
       - Fast to compute, easy to interpret
       - Good baseline for importance

    5. Clustering Coefficient:
       - Measures how clustered an entity's neighborhood is
       - High clustering = entity part of dense subgraph
       - Range: 0 to 1

COMBINED IMPORTANCE SCORE:
    Weighted combination of multiple metrics:
    - 30% PageRank (global importance)
    - 20% Betweenness (bridge importance)
    - 20% Eigenvector (neighborhood quality)
    - 15% Degree (connection count)
    - 15% Mentions (raw frequency)

    WHY WEIGHTED:
        - No single metric captures all aspects of importance
        - Ensemble approach more robust than any single metric
        - Weights empirically chosen based on semantic relevance

PRELIMINARY COMMUNITY DETECTION:
    Louvain algorithm for modularity optimization:
    - Detects preliminary topic clusters
    - Helps understand graph structure before final clustering
    - Modularity score indicates cluster quality

OUTPUT:
    Enhanced graph with additional node attributes:
    - pagerank: PageRank score (0-1, sum=1)
    - betweenness: Betweenness centrality (0-1 normalized)
    - eigenvector: Eigenvector centrality (0-1 normalized)
    - in_degree, out_degree, total_degree: Connection counts
    - clustering_coefficient: Local clustering (0-1)
    - community: Preliminary community ID
    - importance: Combined weighted score (0-1)

COMPUTATIONAL COMPLEXITY:
    - PageRank: O(k * E) where k=iterations, E=edges (~2-3 seconds)
    - Betweenness: O(V * E) (~5-10 seconds for 600 nodes)
    - Eigenvector: O(k * V²) (~1-2 seconds)
    - Total: ~10-15 seconds for typical graph
"""
import networkx as nx
import community as community_louvain
import numpy as np
from typing import Dict, List, Tuple
import json
from pathlib import Path
from utils import get_logger, save_json
import config

logger = get_logger(__name__)


class GraphEnhancer:
    """
    Enhance knowledge graph with metrics and analysis
    """

    def __init__(self):
        """Initialize graph enhancer"""
        logger.info("Graph enhancer initialized")

    def compute_centrality_metrics(self, G: nx.DiGraph) -> Dict[str, Dict[str, float]]:
        """
        Compute various centrality metrics for all nodes

        Args:
            G: NetworkX graph

        Returns:
            Dictionary mapping node -> metrics
        """
        logger.info("Computing centrality metrics...")

        metrics = {}

        # PageRank - importance based on incoming links
        logger.info("Computing PageRank...")
        try:
            pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
        except:
            pagerank = {node: 1.0/G.number_of_nodes() for node in G.nodes()}

        # Betweenness Centrality - node importance as bridge
        logger.info("Computing betweenness centrality...")
        try:
            betweenness = nx.betweenness_centrality(G)
        except:
            betweenness = {node: 0 for node in G.nodes()}

        # Eigenvector Centrality - importance based on neighbor importance
        logger.info("Computing eigenvector centrality...")
        try:
            # Convert to undirected for eigenvector centrality
            G_undirected = G.to_undirected()
            eigenvector = nx.eigenvector_centrality(G_undirected, max_iter=1000)
        except:
            eigenvector = {node: 1.0/G.number_of_nodes() for node in G.nodes()}

        # Degree Centrality
        logger.info("Computing degree centrality...")
        in_degree = dict(G.in_degree())
        out_degree = dict(G.out_degree())
        total_degree = {node: in_degree.get(node, 0) + out_degree.get(node, 0)
                       for node in G.nodes()}

        # Clustering Coefficient (for undirected version)
        logger.info("Computing clustering coefficient...")
        G_undirected = G.to_undirected()
        clustering = nx.clustering(G_undirected)

        # Combine all metrics
        for node in G.nodes():
            metrics[node] = {
                'pagerank': pagerank.get(node, 0),
                'betweenness': betweenness.get(node, 0),
                'eigenvector': eigenvector.get(node, 0),
                'in_degree': in_degree.get(node, 0),
                'out_degree': out_degree.get(node, 0),
                'total_degree': total_degree.get(node, 0),
                'clustering_coefficient': clustering.get(node, 0),
            }

        logger.info("Centrality metrics computed")
        return metrics

    def detect_communities(self, G: nx.DiGraph) -> Dict[str, int]:
        """
        Detect communities using Louvain algorithm

        Args:
            G: NetworkX graph

        Returns:
            Dictionary mapping node -> community_id
        """
        logger.info("Detecting communities...")

        # Convert to undirected for community detection
        G_undirected = G.to_undirected()

        # Apply Louvain algorithm
        communities = community_louvain.best_partition(G_undirected)

        num_communities = len(set(communities.values()))
        logger.info(f"Detected {num_communities} communities")

        return communities

    def compute_entity_importance(self, G: nx.DiGraph, metrics: Dict[str, Dict]) -> Dict[str, float]:
        """
        Compute overall entity importance score

        Combines multiple metrics into single importance score

        Args:
            G: NetworkX graph
            metrics: Centrality metrics

        Returns:
            Dictionary mapping node -> importance_score (0-1)
        """
        logger.info("Computing entity importance scores...")

        importance = {}

        # Normalize metrics to 0-1 range
        def normalize(values):
            if not values or max(values) == 0:
                return {k: 0 for k in values}
            max_val = max(values.values()) if isinstance(values, dict) else max(values)
            min_val = min(values.values()) if isinstance(values, dict) else min(values)
            if max_val == min_val:
                return {k: 0.5 for k in values}
            return {k: (v - min_val) / (max_val - min_val)
                   for k, v in values.items()}

        # Extract individual metrics
        pagerank_norm = normalize({n: metrics[n]['pagerank'] for n in G.nodes()})
        betweenness_norm = normalize({n: metrics[n]['betweenness'] for n in G.nodes()})
        eigenvector_norm = normalize({n: metrics[n]['eigenvector'] for n in G.nodes()})
        degree_norm = normalize({n: metrics[n]['total_degree'] for n in G.nodes()})

        # Mentions (from node attributes)
        mentions = {n: G.nodes[n].get('mentions', 1) for n in G.nodes()}
        mentions_norm = normalize(mentions)

        # Weighted combination
        weights = {
            'pagerank': 0.3,
            'betweenness': 0.2,
            'eigenvector': 0.2,
            'degree': 0.15,
            'mentions': 0.15,
        }

        for node in G.nodes():
            importance[node] = (
                weights['pagerank'] * pagerank_norm[node] +
                weights['betweenness'] * betweenness_norm[node] +
                weights['eigenvector'] * eigenvector_norm[node] +
                weights['degree'] * degree_norm[node] +
                weights['mentions'] * mentions_norm[node]
            )

        logger.info("Entity importance scores computed")
        return importance

    def infer_entity_types(self, G: nx.DiGraph) -> Dict[str, str]:
        """
        Infer entity types from graph context

        Args:
            G: NetworkX graph

        Returns:
            Dictionary mapping node -> inferred_type
        """
        logger.info("Inferring entity types from context...")

        inferred_types = {}

        for node in G.nodes():
            # Start with existing type
            current_type = G.nodes[node].get('type', 'unknown')

            # If already has specific type, keep it
            if current_type not in ['unknown', 'concept']:
                inferred_types[node] = current_type
                continue

            # Infer from relationships
            outgoing_relations = []
            incoming_relations = []

            for _, target, data in G.out_edges(node, data=True):
                relations = data.get('relations', [])
                outgoing_relations.extend(relations)

            for source, _, data in G.in_edges(node, data=True):
                relations = data.get('relations', [])
                incoming_relations.extend(relations)

            # Type inference rules
            if 'uses' in outgoing_relations or 'implements' in outgoing_relations:
                inferred_types[node] = 'method'
            elif 'is_a' in incoming_relations:
                inferred_types[node] = 'category'
            elif 'contains' in outgoing_relations:
                inferred_types[node] = 'architecture'
            elif len(incoming_relations) > len(outgoing_relations) and incoming_relations:
                inferred_types[node] = 'component'
            else:
                inferred_types[node] = current_type

        logger.info("Entity types inferred")
        return inferred_types

    def enhance_graph(self, G: nx.DiGraph) -> nx.DiGraph:
        """
        Enhance graph with all computed metrics

        Args:
            G: NetworkX graph

        Returns:
            Enhanced graph with metrics as node attributes
        """
        logger.info("Enhancing graph with computed metrics...")

        # Compute all metrics
        centrality_metrics = self.compute_centrality_metrics(G)
        communities = self.detect_communities(G)
        importance = self.compute_entity_importance(G, centrality_metrics)
        inferred_types = self.infer_entity_types(G)

        # Add metrics to graph
        for node in G.nodes():
            # Centrality metrics
            G.nodes[node]['pagerank'] = centrality_metrics[node]['pagerank']
            G.nodes[node]['betweenness'] = centrality_metrics[node]['betweenness']
            G.nodes[node]['eigenvector'] = centrality_metrics[node]['eigenvector']
            G.nodes[node]['in_degree'] = centrality_metrics[node]['in_degree']
            G.nodes[node]['out_degree'] = centrality_metrics[node]['out_degree']
            G.nodes[node]['total_degree'] = centrality_metrics[node]['total_degree']
            G.nodes[node]['clustering_coefficient'] = centrality_metrics[node]['clustering_coefficient']

            # Community
            G.nodes[node]['community'] = communities.get(node, -1)

            # Importance
            G.nodes[node]['importance'] = importance[node]

            # Inferred type
            G.nodes[node]['inferred_type'] = inferred_types[node]

        # Compute graph-level statistics
        graph_stats = self.compute_graph_statistics(G, centrality_metrics, communities)

        logger.info("Graph enhancement complete!")
        return G, graph_stats

    def compute_graph_statistics(self, G: nx.DiGraph,
                                 metrics: Dict,
                                 communities: Dict) -> Dict:
        """Compute overall graph statistics"""
        stats = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'num_communities': len(set(communities.values())),

            # Average metrics
            'avg_pagerank': np.mean([metrics[n]['pagerank'] for n in G.nodes()]),
            'avg_betweenness': np.mean([metrics[n]['betweenness'] for n in G.nodes()]),
            'avg_clustering_coeff': np.mean([metrics[n]['clustering_coefficient'] for n in G.nodes()]),

            # Top entities
            'top_entities_by_pagerank': sorted(
                [(n, metrics[n]['pagerank']) for n in G.nodes()],
                key=lambda x: x[1], reverse=True
            )[:10],

            'top_entities_by_betweenness': sorted(
                [(n, metrics[n]['betweenness']) for n in G.nodes()],
                key=lambda x: x[1], reverse=True
            )[:10],
        }

        # Convert tuples to dicts for JSON serialization
        stats['top_entities_by_pagerank'] = [
            {'entity': e, 'score': s} for e, s in stats['top_entities_by_pagerank']
        ]
        stats['top_entities_by_betweenness'] = [
            {'entity': e, 'score': s} for e, s in stats['top_entities_by_betweenness']
        ]

        return stats


def load_graph_from_json(graph_file: str) -> nx.DiGraph:
    """Load graph from JSON format"""
    logger.info(f"Loading graph from {graph_file}")

    with open(graph_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    G = nx.DiGraph()

    # Add nodes
    for node_data in data['nodes']:
        node_id = node_data['id']
        # Convert aliases back to set if it's a list
        if 'aliases' in node_data and isinstance(node_data['aliases'], list):
            node_data['aliases'] = set(node_data['aliases'])
        G.add_node(node_id, **{k: v for k, v in node_data.items() if k != 'id'})

    # Add edges
    for edge_data in data['edges']:
        source = edge_data['source']
        target = edge_data['target']
        G.add_edge(source, target, **{k: v for k, v in edge_data.items()
                                     if k not in ['source', 'target']})

    logger.info(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G


def save_enhanced_graph(G: nx.DiGraph, output_path: str, stats: Dict):
    """Save enhanced graph with all metrics"""
    logger.info(f"Saving enhanced graph to {output_path}")

    graph_data = {
        'nodes': [],
        'edges': [],
        'statistics': stats
    }

    # Save nodes
    for node in G.nodes():
        node_data = {'id': node}
        node_data.update(G.nodes[node])

        # Convert sets to lists for JSON
        if 'aliases' in node_data and isinstance(node_data['aliases'], set):
            node_data['aliases'] = list(node_data['aliases'])

        graph_data['nodes'].append(node_data)

    # Save edges
    for source, target in G.edges():
        edge_data = {
            'source': source,
            'target': target
        }
        edge_data.update(G[source][target])
        graph_data['edges'].append(edge_data)

    save_json(graph_data, output_path)
    logger.info(f"Enhanced graph saved to {output_path}")


def enhance_global_graph(input_file: str, output_file: str):
    """
    Enhance global graph with metrics

    Args:
        input_file: Path to global graph JSON
        output_file: Path to save enhanced graph
    """
    # Load graph
    G = load_graph_from_json(input_file)

    # Enhance
    enhancer = GraphEnhancer()
    G_enhanced, stats = enhancer.enhance_graph(G)

    # Save
    save_enhanced_graph(G_enhanced, output_file, stats)

    logger.info("Graph enhancement complete!")
    return G_enhanced, stats


if __name__ == "__main__":
    import os

    input_file = os.path.join(config.KG_DIR, "global_knowledge_graph.json")
    output_file = os.path.join(config.KG_DIR, "enhanced_global_graph.json")

    enhance_global_graph(input_file, output_file)
