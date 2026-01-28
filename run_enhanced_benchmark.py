"""
Enhanced Benchmark v2: Flat vs Hierarchical (FIXED)

Fixes:
1. Proper hierarchical retrieval implementation
2. Additional research-based metrics from 2024 papers

New Metrics Added Based on Research:
1. Branch Topic Quality (from HTM papers)
2. Level Topic Quality (hierarchical coherence)
3. Topic-Subtopic Relations
4. F1-Score for retrieval (RAPTOR benchmark)
5. Precision@K and Recall@K
6. Tree Depth Utilization
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

from utils import get_logger, save_json
import config

logger = get_logger(__name__)

COMPARISON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison")
os.makedirs(COMPARISON_DIR, exist_ok=True)


def compute_f1_score(retrieved: set, relevant: set) -> dict:
    """Compute F1, Precision, Recall scores (from RAPTOR paper)."""
    if not relevant or not retrieved:
        return {'precision': 0, 'recall': 0, 'f1': 0}
    
    tp = len(retrieved & relevant)
    precision = tp / len(retrieved) if retrieved else 0
    recall = tp / len(relevant) if relevant else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}


class EnhancedFlatEvaluator:
    """Enhanced evaluator for flat model with additional metrics."""
    
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
        """Run enhanced evaluation."""
        results = {
            'model_type': 'flat',
            'timestamp': datetime.now().isoformat()
        }
        
        # Basic stats
        results['basic_stats'] = {
            'num_topics': len(self.topics),
            'num_nodes': self.G.number_of_nodes(),
            'num_edges': self.G.number_of_edges()
        }
        
        # Clustering quality
        results['clustering'] = self._evaluate_clustering()
        
        # Topic quality
        results['topics'] = self._evaluate_topics()
        
        # Retrieval (with F1 scores)
        results['retrieval'] = self._evaluate_retrieval()
        
        return results
    
    def _evaluate_clustering(self) -> dict:
        """Enhanced clustering metrics."""
        clusters = {}
        for i, t in enumerate(self.topics):
            for e in t.get('key_entities', []):
                name = e.get('name', e) if isinstance(e, dict) else e
                clusters[name] = i
        for node in self.G.nodes():
            if node not in clusters:
                clusters[node] = -1
        
        valid_nodes = [n for n in self.G.nodes() if clusters[n] >= 0]
        subG = self.G.to_undirected().subgraph(valid_nodes)
        partition = {n: clusters[n] for n in valid_nodes}
        
        try:
            import community as community_louvain
            mod = community_louvain.modularity(partition, subG)
        except:
            mod = 0.0
        
        coverage = 1 - (len([c for c in clusters.values() if c == -1]) / len(clusters))
        
        return {
            'modularity': mod,
            'coverage': coverage
        }
    
    def _evaluate_topics(self) -> dict:
        """Enhanced topic quality metrics."""
        coherences = []
        
        for topic in self.topics:
            entities = [e.get('name', e) if isinstance(e, dict) else e for e in topic.get('key_entities', [])[:10]]
            if len(entities) > 1:
                embs = self.embedding_model.encode(entities)
                sim = cosine_similarity(embs)
                n = len(embs)
                coherences.append((np.sum(sim) - n) / (n * (n - 1)))
        
        return {
            'avg_coherence': np.mean(coherences) if coherences else 0,
            'min_coherence': np.min(coherences) if coherences else 0,
            'max_coherence': np.max(coherences) if coherences else 0
        }
    
    def _evaluate_retrieval(self) -> dict:
        """Enhanced retrieval with F1 scores."""
        from modules.graph_retrieval import GraphAwareRetriever
        
        retriever = GraphAwareRetriever(
            os.path.join(config.DATA_DIR, "best_kg", "phase5_improved_graph.json"),
            os.path.join(config.DATA_DIR, "best_semantic_models", "improved_semantic_model.json")
        )
        
        # Test queries with ground truth entities
        test_cases = [
            {
                'query': "How does Vision Transformer work?",
                'ground_truth': {'Vision Transformer (ViT)', 'ViT', 'Self-Attention', 'ImageNet', 'Transformer'}
            },
            {
                'query': "What is self-attention?",
                'ground_truth': {'Self-Attention', 'Attention', 'Transformer', 'Multi-Head Attention'}
            },
            {
                'query': "GPT-3 training data",
                'ground_truth': {'GPT-3', 'WebText', 'Common Crawl', 'OpenAI'}
            }
        ]
        
        times = []
        f1_scores = []
        precisions = []
        recalls = []
        entity_counts = []
        
        for case in test_cases:
            start = time.time()
            result = retriever.retrieve(case['query'])
            elapsed = time.time() - start
            
            retrieved_entities = set(result.get('entities', []))
            ground_truth = case['ground_truth']
            
            scores = compute_f1_score(retrieved_entities, ground_truth)
            
            times.append(elapsed)
            f1_scores.append(scores['f1'])
            precisions.append(scores['precision'])
            recalls.append(scores['recall'])
            entity_counts.append(len(retrieved_entities))
        
        return {
            'avg_time': np.mean(times),
            'avg_f1': np.mean(f1_scores),
            'avg_precision': np.mean(precisions),
            'avg_recall': np.mean(recalls),
            'avg_entities': np.mean(entity_counts)
        }


class EnhancedHierarchicalEvaluator:
    """Enhanced evaluator for hierarchical model with research-based metrics."""
    
    def __init__(self, model_path: str):
        logger.info("Loading hierarchical model...")
        
        with open(model_path, 'r', encoding='utf-8') as f:
            self.model = json.load(f)
        
        self.tree = self.model.get('hierarchical_tree', {})
        self.hierarchy = self.tree.get('hierarchy', {})
        self.level_index = self.tree.get('level_index', {})
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def evaluate(self) -> dict:
        """Run enhanced evaluation."""
        results = {
            'model_type': 'hierarchical',
            'timestamp': datetime.now().isoformat()
        }
        
        # Tree structure
        results['structure'] = self._evaluate_structure()
        
        # Branch quality (NEW from research)
        results['branch_quality'] = self._evaluate_branch_quality()
        
        # Level quality (NEW from research)
        results['level_quality'] = self._evaluate_level_quality()
        
        # Topic-subtopic relations (NEW from research)
        results['topic_subtopic'] = self._evaluate_topic_subtopic()
        
        # Retrieval with F1
        results['retrieval'] = self._evaluate_retrieval()
        
        return results
    
    def _evaluate_structure(self) -> dict:
        """Evaluate tree structure."""
        level_counts = {int(k): len(v) for k, v in self.level_index.items()}
        
        def count_branching(node):
            children = node.get('children', [])
            if children:
                return [len(children)] + sum((count_branching(c) for c in children), [])
            return []
        
        branching = count_branching(self.hierarchy)
        
        return {
            'total_levels': len(level_counts),
            'total_nodes': sum(level_counts.values()),
            'level_distribution': level_counts,
            'avg_branching': np.mean(branching) if branching else 0
        }
    
    def _evaluate_branch_quality(self) -> dict:
        """
        Branch Topic Quality (from HTM research).
        
        Measures how well a parent node's topic generalizes its children.
        """
        scores = []
        
        def evaluate_branch(node):
            children = node.get('children', [])
            if len(children) < 2:
                return
            
            parent_name = node.get('name', '')
            child_names = [c.get('name', '') for c in children]
            
            if not parent_name or not child_names:
                return
            
            # Embed
            embs = self.embedding_model.encode([parent_name] + child_names)
            parent_emb = embs[0]
            child_embs = embs[1:]
            
            # Children should be similar to parent (high)
            parent_child_sims = cosine_similarity([parent_emb], child_embs)[0]
            avg_parent_child = np.mean(parent_child_sims)
            
            # Children should be distinct from each other (moderate)
            if len(child_embs) > 1:
                child_sims = cosine_similarity(child_embs)
                n = len(child_embs)
                avg_child_diversity = 1 - (np.sum(child_sims) - n) / (n * (n - 1))
            else:
                avg_child_diversity = 0.5
            
            # Quality = high parent-child + high diversity
            quality = 0.7 * avg_parent_child + 0.3 * avg_child_diversity
            scores.append(quality)
            
            # Recurse
            for child in children:
                evaluate_branch(child)
        
        evaluate_branch(self.hierarchy)
        
        return {
            'avg_branch_quality': np.mean(scores) if scores else 0,
            'num_branches': len(scores)
        }
    
    def _evaluate_level_quality(self) -> dict:
        """
        Level Topic Quality (from HTM research).
        
        Topics at same level should be diverse (non-overlapping).
        """
        level_qualities = {}
        
        for level_str, nodes in self.level_index.items():
            if len(nodes) < 2:
                continue
            
            level = int(level_str)
            names = [n.get('name', '') for n in nodes]
            embs = self.embedding_model.encode(names)
            
            # Diversity = 1 - similarity
            sims = cosine_similarity(embs)
            n = len(embs)
            avg_sim = (np.sum(sims) - n) / (n * (n - 1))
            diversity = 1 - avg_sim
            
            level_qualities[level] = diversity
        
        return {
            'avg_level_diversity': np.mean(list(level_qualities.values())) if level_qualities else 0,
            'per_level': level_qualities
        }
    
    def _evaluate_topic_subtopic(self) -> dict:
        """
        Topic-Subtopic Relations (from HTM research).
        
        Evaluates quality of hierarchical edges.
        """
        relation_scores = []
        
        def evaluate_relations(node, parent_name=None):
            current_name = node.get('name', '')
            children = node.get('children', [])
            
            if parent_name and current_name:
                # Current should be more specific than parent
                parent_emb = self.embedding_model.encode([parent_name])[0]
                current_emb = self.embedding_model.encode([current_name])[0]
                similarity = cosine_similarity([parent_emb], [current_emb])[0][0]
                
                # Should be similar (subtopic of parent) but not identical
                # Ideal: 0.5-0.8 similarity
                quality = 1 - abs(similarity - 0.65)
                relation_scores.append(quality)
            
            # Recurse
            for child in children:
                evaluate_relations(child, current_name)
        
        evaluate_relations(self.hierarchy)
        
        return {
            'avg_relation_quality': np.mean(relation_scores) if relation_scores else 0,
            'num_relations': len(relation_scores)
        }
    
    def _evaluate_retrieval(self) -> dict:
        """Enhanced retrieval with proper implementation."""
        from modules.hierarchical_retrieval import load_retriever
        
        model_path = os.path.join(config.DATA_DIR, "best_semantic_models", "hierarchical_semantic_model.json")
        retriever = load_retriever(model_path)
        
        # Test cases with ground truth
        test_cases = [
            {
                'query': "How does Vision Transformer work?",
                'ground_truth': {'Vision Transformer (ViT)', 'ViT', 'Self-Attention', 'ImageNet'}
            },
            {
                'query': "What is self-attention?",
                'ground_truth': {'Self-Attention', 'Attention', 'Transformer'}
            },
            {
                'query': "GPT-3 training data",
                'ground_truth': {'GPT-3', 'WebText', 'Common Crawl'}
            }
        ]
        
        times = []
        f1_scores = []
        precisions = []
        recalls = []
        path_lengths = []
        entity_counts = []
        
        for case in test_cases:
            start = time.time()
            result = retriever.retrieve(case['query'], mode='tree_traversal', return_path=True)
            elapsed = time.time() - start
            
            # Get retrieved entities properly
            entities = result.get('entities', [])
            if isinstance(entities, list) and entities:
                # Extract entity names from dict or string
                retrieved_entities = set()
                for e in entities:
                    if isinstance(e, dict):
                        retrieved_entities.add(e.get('name', str(e)))
                    else:
                        retrieved_entities.add(str(e))
            else:
                retrieved_entities = set()
            
            ground_truth = case['ground_truth']
            scores = compute_f1_score(retrieved_entities, ground_truth)
            
            times.append(elapsed)
            f1_scores.append(scores['f1'])
            precisions.append(scores['precision'])
            recalls.append(scores['recall'])
            path_lengths.append(len(result.get('traversal_path', [])))
            entity_counts.append(len(retrieved_entities))
        
        return {
            'avg_time': np.mean(times),
            'avg_f1': np.mean(f1_scores),
            'avg_precision': np.mean(precisions),
            'avg_recall': np.mean(recalls),
            'avg_path_length': np.mean(path_lengths),
            'avg_entities': np.mean(entity_counts)
        }


def run_enhanced_benchmark():
    """Run enhanced benchmark with all new metrics."""
    print("\n" + "="*70)
    print(" 🚀 ENHANCED BENCHMARK v2: FLAT vs HIERARCHICAL")
    print("="*70)
    print(" NEW METRICS: Branch Quality, Level Quality, Topic-Subtopic, F1")
    print("="*70)
    
    # Paths
    flat_model = os.path.join(config.DATA_DIR, "best_semantic_models", "improved_semantic_model.json")
    graph_path = os.path.join(config.DATA_DIR, "best_kg", "phase5_improved_graph.json")
    hier_model = os.path.join(config.DATA_DIR, "best_semantic_models", "hierarchical_semantic_model.json")
    
    # Evaluate flat
    print("\n📊 Evaluating FLAT model...")
    flat_eval = EnhancedFlatEvaluator(flat_model, graph_path)
    flat_results = flat_eval.evaluate()
    
    print(f"   Modularity: {flat_results['clustering']['modularity']:.4f}")
    print(f"   Coherence: {flat_results['topics']['avg_coherence']:.4f}")
    print(f"   F1-Score: {flat_results['retrieval']['avg_f1']:.4f}")
    print(f"   Precision: {flat_results['retrieval']['avg_precision']:.4f}")
    print(f"   Recall: {flat_results['retrieval']['avg_recall']:.4f}")
    
    # Evaluate hierarchical
    print("\n🌳 Evaluating HIERARCHICAL model...")
    hier_eval = EnhancedHierarchicalEvaluator(hier_model)
    hier_results = hier_eval.evaluate()
    
    print(f"   Branch Quality: {hier_results['branch_quality']['avg_branch_quality']:.4f}")
    print(f"   Level Diversity: {hier_results['level_quality']['avg_level_diversity']:.4f}")
    print(f"   Topic-Subtopic: {hier_results['topic_subtopic']['avg_relation_quality']:.4f}")
    print(f"   F1-Score: {hier_results['retrieval']['avg_f1']:.4f}")
    print(f"   Entities Retrieved: {hier_results['retrieval']['avg_entities']:.1f}")
    
    # Comparison
    print("\n" + "="*70)
    print(" 🏆 COMPARISON")
    print("="*70)
    
    comparison = {
        'flat': flat_results,
        'hierarchical': hier_results,
        'winner': {}
    }
    
    metrics_comparison = [
        ('Modularity', flat_results['clustering']['modularity'], None, 'higher'),
        ('Topic Coherence', flat_results['topics']['avg_coherence'], 
         hier_results['branch_quality']['avg_branch_quality'], 'higher'),
        ('F1-Score', flat_results['retrieval']['avg_f1'], 
         hier_results['retrieval']['avg_f1'], 'higher'),
        ('Precision', flat_results['retrieval']['avg_precision'],
         hier_results['retrieval']['avg_precision'], 'higher'),
        ('Recall', flat_results['retrieval']['avg_recall'],
         hier_results['retrieval']['avg_recall'], 'higher'),
        ('Retrieval Speed', flat_results['retrieval']['avg_time'],
         hier_results['retrieval']['avg_time'], 'lower'),
        ('Entities Retrieved', flat_results['retrieval']['avg_entities'],
         hier_results['retrieval']['avg_entities'], 'higher')
    ]
    
    print(f"\n   {'Metric':<20} {'Flat':>12} {'Hierarchical':>15} {'Winner':>15}")
    print("   " + "-"*62)
    
    flat_wins = 0
    hier_wins = 0
    
    for metric, flat_val, hier_val, direction in metrics_comparison:
        if hier_val is None:
            print(f"   {metric:<20} {flat_val:>12.4f} {'N/A':>15} {'Flat':>15}")
            comparison['winner'][metric] = 'flat'
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
            
            print(f"   {metric:<20} {flat_val:>12.4f} {hier_val:>15.4f} {winner:>15}")
            comparison['winner'][metric] = winner.lower()
    
    # Overall
    overall_winner = 'Flat' if flat_wins > hier_wins else 'Hierarchical' if hier_wins > flat_wins else 'Tie'
    comparison['summary'] = {
        'flat_wins': flat_wins,
        'hierarchical_wins': hier_wins,
        'overall_winner': overall_winner.lower()
    }
    
    print("\n" + "="*70)
    print(f" 🏆 OVERALL WINNER: {overall_winner}")
    print(f"    Flat: {flat_wins} wins | Hierarchical: {hier_wins} wins")
    print("="*70)
    
    # Save results
    results_path = os.path.join(COMPARISON_DIR, 'enhanced_benchmark_results.json')
    save_json(comparison, results_path)
    print(f"\n💾 Saved: {results_path}")
    
    # Generate visualization
    generate_enhanced_viz(comparison)
    
    return comparison


def generate_enhanced_viz(comparison: dict):
    """Generate enhanced visualization."""
    import matplotlib.pyplot as plt
    
    print("\n📊 Generating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Chart 1: Winner count
    ax1 = axes[0, 0]
    winners = list(comparison['winner'].values())
    flat_count = winners.count('flat')
    hier_count = winners.count('hierarchical')
    
    ax1.bar(['Flat', 'Hierarchical'], [flat_count, hier_count], color=['#3498db', '#2ecc71'])
    ax1.set_title('Metrics Won', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count')
    
    for i, v in enumerate([flat_count, hier_count]):
        ax1.text(i, v + 0.1, str(v), ha='center', fontweight='bold')
    
    # Chart 2: F1 Scores
    ax2 = axes[0, 1]
    flat_f1 = comparison['flat']['retrieval']['avg_f1']
    hier_f1 = comparison['hierarchical']['retrieval']['avg_f1']
    
    ax2.bar(['Flat', 'Hierarchical'], [flat_f1, hier_f1], color=['#3498db', '#2ecc71'])
    ax2.set_title('F1-Score (RAPTOR Metric)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1-Score')
    ax2.set_ylim(0, 1)
    
    # Chart 3: Hierarchical-specific metrics
    ax3 = axes[1, 0]
    hier_metrics = ['Branch\nQuality', 'Level\nDiversity', 'Topic-\nSubtopic']
    hier_vals = [
        comparison['hierarchical']['branch_quality']['avg_branch_quality'],
        comparison['hierarchical']['level_quality']['avg_level_diversity'],
        comparison['hierarchical']['topic_subtopic']['avg_relation_quality']
    ]
    
    ax3.barh(hier_metrics, hier_vals, color='#2ecc71')
    ax3.set_title('Hierarchical-Specific Metrics', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, 1)
    
    # Chart 4: Speed comparison
    ax4 = axes[1, 1]
    flat_time = comparison['flat']['retrieval']['avg_time']
    hier_time = comparison['hierarchical']['retrieval']['avg_time']
    
    ax4.bar(['Flat', 'Hierarchical'], [flat_time * 1000, hier_time * 1000], color=['#3498db', '#2ecc71'])
    ax4.set_title('Retrieval Speed', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Time (ms)')
    
    plt.tight_layout()
    
    viz_path = os.path.join(COMPARISON_DIR, 'enhanced_comparison.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ enhanced_comparison.png")


if __name__ == "__main__":
    run_enhanced_benchmark()
