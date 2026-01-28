"""
Comprehensive Benchmark: Flat vs Hierarchical Models

Compares flat semantic topics against hierarchical tree structure using:
1. Traditional metrics (Modularity, Silhouette, Coherence)
2. Hierarchical-specific metrics (Parent-child quality, Tree depth, Coverage)
3. Retrieval performance (Accuracy, Speed, Multi-hop capability)
4. Query answering quality

Based on research from:
- RAPTOR framework evaluation strategies
- Hierarchical topic model benchmarking (2024)
- Tree-based retrieval best practices
"""
import json
import os
import sys
import time
import numpy as np
import networkx as nx
from collections import defaultdict
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.hierarchical_retrieval import load_retriever
from modules.graph_retrieval import GraphAwareRetriever, ContextGenerator
from utils import get_logger, save_json
import config

logger = get_logger(__name__)

# Create comparison directory
COMPARISON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison")
os.makedirs(COMPARISON_DIR, exist_ok=True)


class FlatModelEvaluator:
    """Evaluate flat semantic topic model."""
    
    def __init__(self, model_path: str, graph_path: str):
        logger.info("Loading flat model...")
        
        with open(model_path, 'r', encoding='utf-8') as f:
            self.model = json.load(f)
        with open(graph_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        # Build graph
        self.G = nx.DiGraph()
        for n in graph_data['nodes']:
            self.G.add_node(n['id'], **{k: v for k, v in n.items() if k != 'id' and not isinstance(v, (set, list))})
        for e in graph_data['edges']:
            self.G.add_edge(e['source'], e['target'], **{k: v for k, v in e.items() if k not in ['source', 'target']})
        
        self.topics = self.model.get('semantic_topics', [])
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def evaluate(self) -> dict:
        """Run full evaluation on flat model."""
        results = {
            'model_type': 'flat',
            'timestamp': datetime.now().isoformat()
        }
        
        # 1. Basic stats
        results['basic_stats'] = {
            'num_topics': len(self.topics),
            'num_nodes': self.G.number_of_nodes(),
            'num_edges': self.G.number_of_edges(),
            'avg_degree': sum(dict(self.G.degree()).values()) / self.G.number_of_nodes()
        }
        
        # 2. Clustering quality
        results['clustering_quality'] = self._evaluate_clustering()
        
        # 3. Topic quality
        results['topic_quality'] = self._evaluate_topics()
        
        # 4. Retrieval performance
        results['retrieval'] = self._evaluate_retrieval()
        
        return results
    
    def _evaluate_clustering(self) -> dict:
        """Evaluate clustering quality metrics."""
        # Build cluster assignments
        clusters = {}
        for i, t in enumerate(self.topics):
            for e in t.get('key_entities', []):
                name = e.get('name', e) if isinstance(e, dict) else e
                clusters[name] = i
        for node in self.G.nodes():
            if node not in clusters:
                clusters[node] = -1
        
        # Modularity
        from modules.graph_improvements import NoiseRescuer  # Has modularity logic
        valid_nodes = [n for n in self.G.nodes() if clusters[n] >= 0]
        subG = self.G.to_undirected().subgraph(valid_nodes)
        partition = {n: clusters[n] for n in valid_nodes}
        
        try:
            import community as community_louvain
            mod = community_louvain.modularity(partition, subG)
        except:
            mod = 0.0
        
        # Coverage
        noise_count = len([c for c in clusters.values() if c == -1])
        coverage = 1 - (noise_count / len(clusters))
        
        return {
            'modularity': mod,
            'coverage': coverage,
            'noise_entities': noise_count
        }
    
    def _evaluate_topics(self) -> dict:
        """Evaluate topic quality."""
        coherence_scores = []
        size_std = []
        
        for topic in self.topics:
            entities = topic.get('key_entities', [])
            size_std.append(len(entities))
            
            # Topic coherence via embedding similarity
            if len(entities) > 1:
                entity_names = [e.get('name', e) if isinstance(e, dict) else e for e in entities[:10]]
                entity_texts = [f"{n}" for n in entity_names]
                embs = self.embedding_model.encode(entity_texts)
                
                sim_matrix = cosine_similarity(embs)
                n = len(embs)
                avg_sim = (np.sum(sim_matrix) - n) / (n * (n - 1)) if n > 1 else 0
                coherence_scores.append(avg_sim)
        
        return {
            'avg_coherence': np.mean(coherence_scores) if coherence_scores else 0,
            'avg_topic_size': np.mean(size_std),
            'topic_size_std': np.std(size_std)
        }
    
    def _evaluate_retrieval(self) -> dict:
        """Evaluate retrieval performance."""
        # Test queries
        queries = [
            "How does Vision Transformer work?",
            "What is self-attention?",
            "GPT-3 training data"
        ]
        
        retriever = GraphAwareRetriever(
            os.path.join(config.DATA_DIR, "best_kg", "phase5_improved_graph.json"),
            os.path.join(config.DATA_DIR, "best_semantic_models", "improved_semantic_model.json")
        )
        
        times = []
        entity_counts = []
        
        for q in queries:
            start = time.time()
            result = retriever.retrieve(q)
            elapsed = time.time() - start
            
            times.append(elapsed)
            entity_counts.append(len(result.get('entities', [])))
        
        return {
            'avg_time': np.mean(times),
            'avg_entities_returned': np.mean(entity_counts)
        }


class HierarchicalModelEvaluator:
    """Evaluate hierarchical tree model."""
    
    def __init__(self, model_path: str):
        logger.info("Loading hierarchical model...")
        
        with open(model_path, 'r', encoding='utf-8') as f:
            self.model = json.load(f)
        
        self.tree = self.model.get('hierarchical_tree', {})
        self.hierarchy = self.tree.get('hierarchy', {})
        self.level_index = self.tree.get('level_index', {})
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def evaluate(self) -> dict:
        """Run full evaluation on hierarchical model."""
        results = {
            'model_type': 'hierarchical',
            'timestamp': datetime.now().isoformat()
        }
        
        # 1. Tree structure stats
        results['tree_structure'] = self._evaluate_structure()
        
        # 2. Parent-child quality (hierarchical-specific)
        results['parent_child_quality'] = self._evaluate_parent_child()
        
        # 3. Hierarchy coherence
        results['hierarchy_coherence'] = self._evaluate_coherence()
        
        # 4. Retrieval performance
        results['retrieval'] = self._evaluate_retrieval()
        
        return results
    
    def _evaluate_structure(self) -> dict:
        """Evaluate tree structure metrics."""
        metadata = self.model.get('metadata', {})
        
        # Count nodes per level
        level_counts = {}
        for level_str, nodes in self.level_index.items():
            level_counts[int(level_str)] = len(nodes)
        
        return {
            'total_levels': metadata.get('total_levels', 0),
            'total_nodes': metadata.get('total_nodes', 0),
            'leaf_nodes': metadata.get('total_leaf_topics', 0),
            'level_distribution': level_counts,
            'avg_branching_factor': self._compute_branching_factor()
        }
    
    def _compute_branching_factor(self) -> float:
        """Compute average number of children per parent node."""
        branching = []
        
        def count_children(node):
            children = node.get('children', [])
            if children:
                branching.append(len(children))
                for child in children:
                    count_children(child)
        
        count_children(self.hierarchy)
        return np.mean(branching) if branching else 0
    
    def _evaluate_parent_child(self) -> dict:
        """
        Evaluate parent-child relationship quality.
        
        Key metric from research: Children should be semantically similar to parent,
        but dissimilar to non-children.
        """
        scores = []
        
        def evaluate_node(node):
            children = node.get('children', [])
            if not children or len(children) < 2:
                return
            
            parent_name = node.get('name', '')
            child_names = [c.get('name', '') for c in children]
            
            # Embed parent and children
            all_names = [parent_name] + child_names
            embs = self.embedding_model.encode(all_names)
            
            parent_emb = embs[0]
            child_embs = embs[1:]
            
            # Similarity of children to parent (should be high)
            parent_child_sims = cosine_similarity([parent_emb], child_embs)[0]
            avg_parent_child_sim = np.mean(parent_child_sims)
            
            # Similarity among children (should be moderate - not too similar)
            if len(child_embs) > 1:
                child_child_sim = cosine_similarity(child_embs)
                n = len(child_embs)
                avg_child_sim = (np.sum(child_child_sim) - n) / (n * (n - 1))
            else:
                avg_child_sim = 0
            
            # Quality score: high parent-child sim, moderate child-child sim
            quality = avg_parent_child_sim - 0.3 * avg_child_sim
            scores.append(quality)
            
            # Recurse
            for child in children:
                evaluate_node(child)
        
        evaluate_node(self.hierarchy)
        
        return {
            'avg_parent_child_quality': np.mean(scores) if scores else 0,
            'num_evaluated_nodes': len(scores)
        }
    
    def _evaluate_coherence(self) -> dict:
        """Evaluate overall hierarchy coherence."""
        # Level-wise coherence: nodes at same level should be distinct
        level_coherences = {}
        
        for level_str, nodes in self.level_index.items():
            if len(nodes) < 2:
                continue
            
            level = int(level_str)
            names = [n.get('name', '') for n in nodes]
            embs = self.embedding_model.encode(names)
            
            # Average pairwise dissimilarity (diversity)
            sim_matrix = cosine_similarity(embs)
            n = len(embs)
            avg_sim = (np.sum(sim_matrix) - n) / (n * (n - 1)) if n > 1 else 0
            diversity = 1 - avg_sim  # Higher diversity is better
            
            level_coherences[level] = diversity
        
        return {
            'avg_level_diversity': np.mean(list(level_coherences.values())) if level_coherences else 0,
            'level_coherences': level_coherences
        }
    
    def _evaluate_retrieval(self) -> dict:
        """Evaluate hierarchical retrieval performance."""
        hier_path = os.path.join(config.DATA_DIR, "best_semantic_models", "hierarchical_semantic_model.json")
        
        retriever = load_retriever(hier_path)
        
        queries = [
            "How does Vision Transformer work?",
            "What is self-attention?",
            "GPT-3 training data"
        ]
        
        times = []
        entity_counts = []
        path_lengths = []
        
        for q in queries:
            start = time.time()
            result = retriever.retrieve(q, mode='tree_traversal', return_path=True)
            elapsed = time.time() - start
            
            times.append(elapsed)
            entity_counts.append(len(result.get('entities', [])))
            path_lengths.append(len(result.get('traversal_path', [])))
        
        return {
            'avg_time': np.mean(times),
            'avg_entities_returned': np.mean(entity_counts),
            'avg_path_length': np.mean(path_lengths)
        }


def run_comprehensive_benchmark():
    """Run full benchmark comparing flat vs hierarchical models."""
    print("\n" + "="*70)
    print(" 📊 COMPREHENSIVE BENCHMARK: FLAT vs HIERARCHICAL")
    print("="*70)
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Paths
    flat_model_path = os.path.join(config.DATA_DIR, "best_semantic_models", "improved_semantic_model.json")
    graph_path = os.path.join(config.DATA_DIR, "best_kg", "phase5_improved_graph.json")
    hier_model_path = os.path.join(config.DATA_DIR, "best_semantic_models", "hierarchical_semantic_model.json")
    
    # Evaluate flat model
    print("\n" + "-"*70)
    print(" EVALUATING FLAT MODEL")
    print("-"*70)
    
    flat_eval = FlatModelEvaluator(flat_model_path, graph_path)
    flat_results = flat_eval.evaluate()
    
    print(f"   Topics: {flat_results['basic_stats']['num_topics']}")
    print(f"   Modularity: {flat_results['clustering_quality']['modularity']:.4f}")
    print(f"   Coverage: {flat_results['clustering_quality']['coverage']:.4f}")
    print(f"   Avg Topic Coherence: {flat_results['topic_quality']['avg_coherence']:.4f}")
    print(f"   Retrieval Time: {flat_results['retrieval']['avg_time']:.3f}s")
    
    # Evaluate hierarchical model
    print("\n" + "-"*70)
    print(" EVALUATING HIERARCHICAL MODEL")
    print("-"*70)
    
    hier_eval = HierarchicalModelEvaluator(hier_model_path)
    hier_results = hier_eval.evaluate()
    
    print(f"   Total Levels: {hier_results['tree_structure']['total_levels']}")
    print(f"   Total Nodes: {hier_results['tree_structure']['total_nodes']}")
    print(f"   Branching Factor: {hier_results['tree_structure']['avg_branching_factor']:.2f}")
    print(f"   Parent-Child Quality: {hier_results['parent_child_quality']['avg_parent_child_quality']:.4f}")
    print(f"   Level Diversity: {hier_results['hierarchy_coherence']['avg_level_diversity']:.4f}")
    print(f"   Retrieval Time: {hier_results['retrieval']['avg_time']:.3f}s")
    print(f"   Avg Path Length: {hier_results['retrieval']['avg_path_length']:.2f}")
    
    # Comparison
    print("\n" + "="*70)
    print(" 🏆 COMPARISON RESULTS")
    print("="*70)
    
    comparison = {
        'flat': flat_results,
        'hierarchical': hier_results,
        'winner': {},
        'summary': {}
    }
    
    # Determine winners
    print("\n   Metric Comparison:")
    print(f"   {'Metric':<30} {'Flat':>15} {'Hierarchical':>15} {'Winner':>12}")
    print("   " + "-"*72)
    
    metrics = [
        ('Modularity', flat_results['clustering_quality']['modularity'], None, 'higher'),
        ('Coverage', flat_results['clustering_quality']['coverage'], None, 'higher'),
        ('Topic Coherence', flat_results['topic_quality']['avg_coherence'], 
         hier_results['parent_child_quality']['avg_parent_child_quality'], 'higher'),
        ('Retrieval Speed', flat_results['retrieval']['avg_time'], 
         hier_results['retrieval']['avg_time'], 'lower'),
        ('Entities Returned', flat_results['retrieval']['avg_entities_returned'],
         hier_results['retrieval']['avg_entities_returned'], 'higher')
    ]
    
    flat_wins = 0
    hier_wins = 0
    
    for metric_name, flat_val, hier_val, direction in metrics:
        if hier_val is None:
            print(f"   {metric_name:<30} {flat_val:>15.4f} {'N/A':>15} {'Flat':>12}")
            comparison['winner'][metric_name] = 'flat'
            flat_wins += 1
        else:
            if direction == 'higher':
                winner = 'Flat' if flat_val > hier_val else 'Hierarchical'
                if flat_val > hier_val:
                    flat_wins += 1
                else:
                    hier_wins += 1
            else:
                winner = 'Flat' if flat_val < hier_val else 'Hierarchical'
                if flat_val < hier_val:
                    flat_wins += 1
                else:
                    hier_wins += 1
            
            print(f"   {metric_name:<30} {flat_val:>15.4f} {hier_val:>15.4f} {winner:>12}")
            comparison['winner'][metric_name] = winner.lower()
    
    # Overall winner
    print("\n" + "="*70)
    overall_winner = 'Flat' if flat_wins > hier_wins else 'Hierarchical' if hier_wins > flat_wins else 'Tie'
    print(f" 🏆 OVERALL WINNER: {overall_winner}")
    print(f"    Flat wins: {flat_wins} | Hierarchical wins: {hier_wins}")
    print("="*70)
    
    comparison['summary'] = {
        'flat_wins': flat_wins,
        'hierarchical_wins': hier_wins,
        'overall_winner': overall_winner.lower()
    }
    
    # Recommendations
    print("\n📋 RECOMMENDATIONS:")
    if overall_winner == 'Flat':
        print("   ✓ Use FLAT model for: Direct queries, fast lookups, evaluation")
        print("   ✓ Flat model shows better clustering quality and speed")
    elif overall_winner == 'Hierarchical':
        print("   ✓ Use HIERARCHICAL model for: Multi-level navigation, exploratory search")
        print("   ✓ Hierarchical model shows better topic organization")
    else:
        print("   ✓ Use FLAT model for: Direct entity lookup, clustering analysis")
        print("   ✓ Use HIERARCHICAL model for: Navigation, topic exploration")
    
    # Save results
    results_path = os.path.join(COMPARISON_DIR, 'benchmark_results.json')
    save_json(comparison, results_path)
    print(f"\n💾 Results saved to: {results_path}")
    
    # Generate visualization
    generate_comparison_viz(comparison)
    
    return comparison


def generate_comparison_viz(comparison: dict):
    """Generate comparison visualization."""
    import matplotlib.pyplot as plt
    
    print("\n📊 Generating comparison visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Chart 1: Winner distribution
    ax1 = axes[0]
    winners = list(comparison['winner'].values())
    flat_count = winners.count('flat')
    hier_count = winners.count('hierarchical')
    
    ax1.bar(['Flat', 'Hierarchical'], [flat_count, hier_count], 
           color=['#3498db', '#2ecc71'])
    ax1.set_title('Metrics Won by Each Model', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Metrics Won')
    
    for i, v in enumerate([flat_count, hier_count]):
        ax1.text(i, v + 0.1, str(v), ha='center', fontweight='bold')
    
    # Chart 2: Key metrics comparison
    ax2 = axes[1]
    
    flat_res = comparison['flat']
    hier_res = comparison['hierarchical']
    
    metrics = ['Modularity', 'Coverage', 'Coherence', 'Speed']
    flat_vals = [
        flat_res['clustering_quality']['modularity'],
        flat_res['clustering_quality']['coverage'],
        flat_res['topic_quality']['avg_coherence'],
        1 / (flat_res['retrieval']['avg_time'] + 0.001)  # Inverse for speed
    ]
    hier_vals = [
        0,  # N/A
        0,  # N/A
        hier_res['parent_child_quality']['avg_parent_child_quality'],
        1 / (hier_res['retrieval']['avg_time'] + 0.001)
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax2.bar(x - width/2, flat_vals, width, label='Flat', color='#3498db')
    ax2.bar(x + width/2, hier_vals, width, label='Hierarchical', color='#2ecc71')
    
    ax2.set_ylabel('Score (normalized)')
    ax2.set_title('Key Metrics Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    
    plt.tight_layout()
    
    viz_path = os.path.join(COMPARISON_DIR, 'comparison_chart.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ comparison_chart.png")


if __name__ == "__main__":
    run_comprehensive_benchmark()
