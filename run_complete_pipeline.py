"""
Complete Semantic Model Pipeline - All Improvements

This script runs the FULL improved pipeline:
1. Phase 1: Relationship inference + noise rescue + LLM summaries
2. Phase 2: Graph-aware retrieval + multi-hop
3. Phase 3: Typed relationships + final results

Run this to get the complete improved semantic model.
"""
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.graph_improvements import (
    RelationshipInferencer, 
    NoiseRescuer, 
    TopicSummaryGenerator
)
from modules.graph_retrieval import GraphAwareRetriever, ContextGenerator
import config
from utils import get_logger, save_json
import networkx as nx
import numpy as np
import openai

logger = get_logger(__name__)


class TypedRelationshipExtractor:
    """
    Extract typed relationships from the knowledge graph.
    
    Adds semantic types like: uses, extends, part_of, trained_on, etc.
    """
    
    RELATION_TYPES = [
        'uses',           # X uses Y (ViT uses Attention)
        'extends',        # X extends Y (Swin extends ViT)
        'part_of',        # X is part of Y (Encoder part of Transformer)
        'trained_on',     # X trained on Y (GPT trained on WebText)
        'outperforms',    # X outperforms Y
        'authored_by',    # Paper authored by Person
        'compared_to',    # X compared to Y
        'applied_to',     # Method applied to Task
        'evaluated_on',   # Model evaluated on Dataset
        'based_on',       # X based on Y
    ]
    
    def __init__(self):
        logger.info("TypedRelationshipExtractor initialized")
    
    def infer_relationship_types(self, G: nx.DiGraph) -> nx.DiGraph:
        """
        Infer relationship types based on entity types and patterns.
        
        Uses LLM for ambiguous cases.
        """
        logger.info(f"Inferring relationship types for {G.number_of_edges()} edges...")
        
        typed_count = 0
        
        for u, v, data in G.edges(data=True):
            current_rel = data.get('relation', 'related_to')
            
            # Skip if already typed (not generic)
            if current_rel not in ['related_to', 'semantically_related']:
                continue
            
            # Get entity types
            u_type = G.nodes[u].get('type', '').lower()
            v_type = G.nodes[v].get('type', '').lower()
            
            # Infer based on entity type patterns
            inferred = self._infer_from_types(u, v, u_type, v_type)
            
            if inferred:
                G[u][v]['relation'] = inferred
                typed_count += 1
        
        logger.info(f"Typed {typed_count} relationships")
        return G
    
    def _infer_from_types(self, source: str, target: str, 
                          source_type: str, target_type: str) -> str:
        """Infer relationship type from entity types."""
        
        source_lower = source.lower()
        target_lower = target.lower()
        
        # Model → Dataset patterns
        if 'model' in source_type or 'architecture' in source_type:
            if 'dataset' in target_type:
                return 'trained_on'
            if 'method' in target_type or 'technique' in target_type:
                return 'uses'
            if 'task' in target_type:
                return 'applied_to'
        
        # Dataset patterns
        if 'dataset' in target_type:
            return 'evaluated_on'
        
        # Architecture patterns
        if 'architecture' in target_type:
            if 'extends' in source_lower or 'based' in source_lower:
                return 'extends'
            return 'uses'
        
        # Person patterns
        if 'person' in target_type or 'author' in target_type:
            return 'authored_by'
        
        # Component patterns
        if 'component' in target_type:
            return 'part_of'
        
        # Method patterns
        if 'method' in target_type or 'technique' in target_type:
            return 'uses'
        
        # Default inference from names
        if any(word in source_lower for word in ['transformer', 'vit', 'bert', 'gpt']):
            if any(word in target_lower for word in ['attention', 'encoder', 'decoder']):
                return 'uses'
            if any(word in target_lower for word in ['imagenet', 'coco', 'wikipedia']):
                return 'trained_on'
        
        return None
    
    def add_llm_typed_relationships(self, G: nx.DiGraph, sample_size: int = 50) -> nx.DiGraph:
        """
        Use LLM to type ambiguous relationships.
        
        Only processes a sample for cost efficiency.
        """
        # Get edges still with generic relations
        untyped = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get('relation') in ['related_to', 'semantically_related']
        ]
        
        if not untyped:
            logger.info("All relationships already typed!")
            return G
        
        logger.info(f"Using LLM to type {min(sample_size, len(untyped))} remaining edges...")
        
        sample = untyped[:sample_size]
        
        try:
            client = openai.OpenAI()
            
            # Batch for efficiency
            batch_size = 10
            for i in range(0, len(sample), batch_size):
                batch = sample[i:i+batch_size]
                
                # Build prompt
                pairs = [f"{u} → {v}" for u, v in batch]
                pairs_str = '\n'.join(pairs)
                
                types_str = ', '.join(self.RELATION_TYPES)
                
                prompt = f"""For each entity pair, determine the relationship type.

Available types: {types_str}

Entity pairs:
{pairs_str}

Respond with ONLY the relationship type for each pair, one per line.
Use 'related_to' if unsure.
"""
                
                response = client.chat.completions.create(
                    model=config.OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a relationship classifier. Respond only with relationship types."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=100
                )
                
                # Parse response
                lines = response.choices[0].message.content.strip().split('\n')
                for (u, v), rel in zip(batch, lines):
                    rel = rel.strip().lower()
                    if rel in self.RELATION_TYPES:
                        G[u][v]['relation'] = rel
            
            typed_now = len([
                1 for _, _, d in G.edges(data=True)
                if d.get('relation') not in ['related_to', 'semantically_related']
            ])
            logger.info(f"Total typed relationships: {typed_now}")
            
        except Exception as e:
            logger.warning(f"LLM typing failed: {e}")
        
        return G


def run_full_pipeline():
    """Run the complete improved pipeline and collect all results."""
    
    start_time = time.time()
    
    print("\n" + "="*70)
    print(" 🚀 RUNNING COMPLETE IMPROVED SEMANTIC MODEL PIPELINE")
    print("="*70)
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'phases': {},
        'metrics': {}
    }
    
    # Paths
    graph_path = os.path.join(config.DATA_DIR, "best_kg", "enhanced_global_graph.json")
    model_path = os.path.join(config.DATA_DIR, "best_semantic_models", "semantic_model_best.json")
    
    # Load initial data
    print("\n📂 Loading base data...")
    with open(graph_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    with open(model_path, 'r', encoding='utf-8') as f:
        semantic_model = json.load(f)
    
    # Build graph
    G = nx.DiGraph()
    for node_data in graph_data['nodes']:
        node_id = node_data['id']
        G.add_node(node_id, **{k: v for k, v in node_data.items() if k != 'id'})
    for edge_data in graph_data['edges']:
        G.add_edge(edge_data['source'], edge_data['target'],
                  **{k: v for k, v in edge_data.items() if k not in ['source', 'target']})
    
    # Record baseline
    baseline = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
        'topics': len(semantic_model.get('semantic_topics', [])),
        'modularity': semantic_model.get('quality_metrics', {}).get('modularity', 0)
    }
    results['metrics']['baseline'] = baseline
    print(f"   Baseline: {baseline['nodes']} nodes, {baseline['edges']} edges")
    
    # ================================================================
    # PHASE 1: Graph Improvements
    # ================================================================
    print("\n" + "-"*70)
    print(" PHASE 1: Relationship Inference & Noise Rescue")
    print("-"*70)
    
    phase1_start = time.time()
    
    # 1.1 Relationship Inference
    print("\n🔗 1.1 Inferring semantic relationships...")
    inferencer = RelationshipInferencer(similarity_threshold=0.65, max_edges_per_node=5)
    G = inferencer.infer_relationships(G)
    
    # 1.2 Noise Rescue
    topics = semantic_model.get('semantic_topics', [])
    current_clusters = {}
    for i, topic in enumerate(topics):
        for entity in topic.get('key_entities', []):
            entity_name = entity.get('name', entity) if isinstance(entity, dict) else entity
            current_clusters[entity_name] = i
    for node in G.nodes():
        if node not in current_clusters:
            current_clusters[node] = -1
    
    noise_before = len([c for c in current_clusters.values() if c == -1])
    print(f"\n🔧 1.2 Rescuing noise entities ({noise_before} noise nodes)...")
    rescuer = NoiseRescuer(similarity_threshold=0.55, use_llm=True)
    updated_clusters = rescuer.rescue_noise(G, current_clusters, topics)
    noise_after = len([c for c in updated_clusters.values() if c == -1])
    
    # 1.3 LLM Summaries
    print("\n📝 1.3 Generating LLM topic summaries...")
    summarizer = TopicSummaryGenerator()
    topics = summarizer.generate_summaries(G, topics)
    
    phase1_time = time.time() - phase1_start
    results['phases']['phase1'] = {
        'edges_added': G.number_of_edges() - baseline['edges'],
        'noise_rescued': noise_before - noise_after,
        'summaries_generated': len(topics),
        'time_seconds': phase1_time
    }
    print(f"\n✅ Phase 1 complete in {phase1_time:.1f}s")
    
    # ================================================================
    # PHASE 2: Graph-Aware Retrieval (already implemented in modules)
    # ================================================================
    print("\n" + "-"*70)
    print(" PHASE 2: Graph-Aware Retrieval Verification")
    print("-"*70)
    
    phase2_start = time.time()
    
    # Save intermediate graph for retrieval
    improved_graph_path = os.path.join(config.DATA_DIR, "best_kg", "improved_global_graph.json")
    improved_graph_data = {
        'nodes': [{'id': n, **{k: v for k, v in G.nodes[n].items() if k != 'aliases' or not isinstance(v, set)}} for n in G.nodes()],
        'edges': [{'source': u, 'target': v, **G[u][v]} for u, v in G.edges()],
        'statistics': graph_data.get('statistics', {})
    }
    for node_data in improved_graph_data['nodes']:
        if 'aliases' in node_data and isinstance(node_data['aliases'], set):
            node_data['aliases'] = list(node_data['aliases'])
    save_json(improved_graph_data, improved_graph_path)
    
    # Save improved model
    improved_model_path = os.path.join(config.DATA_DIR, "best_semantic_models", "improved_semantic_model.json")
    semantic_model['semantic_topics'] = topics
    save_json(semantic_model, improved_model_path)
    
    # Test retrieval
    print("\n🔍 Testing graph-aware retrieval...")
    retriever = GraphAwareRetriever(improved_graph_path, improved_model_path)
    context_gen = ContextGenerator(retriever)
    
    test_queries = [
        "How does Vision Transformer work?",
        "What is self-attention?",
        "GPT-3 training data"
    ]
    
    retrieval_results = []
    for query in test_queries:
        result = context_gen.generate_context(query, include_llm_summary=True)
        retrieval_results.append({
            'query': query,
            'entities': len(result.get('entities', [])),
            'relationships': len(result.get('relationships', []))
        })
        print(f"   Query: '{query[:30]}...' → {len(result.get('entities', []))} entities, {len(result.get('relationships', []))} relations")
    
    phase2_time = time.time() - phase2_start
    results['phases']['phase2'] = {
        'retrieval_tests': retrieval_results,
        'time_seconds': phase2_time
    }
    print(f"\n✅ Phase 2 complete in {phase2_time:.1f}s")
    
    # ================================================================
    # PHASE 3: Typed Relationships
    # ================================================================
    print("\n" + "-"*70)
    print(" PHASE 3: Typed Relationships")
    print("-"*70)
    
    phase3_start = time.time()
    
    # Count initial generic relations
    generic_before = len([
        1 for _, _, d in G.edges(data=True)
        if d.get('relation', 'related_to') in ['related_to', 'semantically_related']
    ])
    
    print(f"\n🏷️ Typing {generic_before} generic relationships...")
    
    extractor = TypedRelationshipExtractor()
    G = extractor.infer_relationship_types(G)
    G = extractor.add_llm_typed_relationships(G, sample_size=50)
    
    # Count typed relations
    relation_types = {}
    for _, _, d in G.edges(data=True):
        rel = d.get('relation', 'related_to')
        relation_types[rel] = relation_types.get(rel, 0) + 1
    
    generic_after = relation_types.get('related_to', 0) + relation_types.get('semantically_related', 0)
    
    phase3_time = time.time() - phase3_start
    results['phases']['phase3'] = {
        'generic_before': generic_before,
        'generic_after': generic_after,
        'typed_count': generic_before - generic_after,
        'relation_distribution': relation_types,
        'time_seconds': phase3_time
    }
    print(f"\n✅ Phase 3 complete in {phase3_time:.1f}s")
    
    # ================================================================
    # FINAL RESULTS
    # ================================================================
    print("\n" + "="*70)
    print(" 📊 FINAL RESULTS")
    print("="*70)
    
    # Final metrics
    final = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
        'typed_relations': len(relation_types),
        'coverage': (len(updated_clusters) - noise_after) / len(updated_clusters)
    }
    results['metrics']['final'] = final
    
    # Print comparison
    print("\n📈 Metrics Comparison:")
    print(f"\n   {'Metric':<25} {'Before':>15} {'After':>15} {'Change':>15}")
    print("   " + "-"*70)
    print(f"   {'Edges':<25} {baseline['edges']:>15} {final['edges']:>15} {'+' + str(final['edges'] - baseline['edges']):>15}")
    print(f"   {'Avg Degree':<25} {baseline['avg_degree']:>15.2f} {final['avg_degree']:>15.2f} {'+' + f'{final['avg_degree'] - baseline['avg_degree']:.2f}':>15}")
    print(f"   {'Coverage':<25} {'38%':>15} {f'{final['coverage']*100:.1f}%':>15} {'+' + f'{(final['coverage'] - 0.38)*100:.1f}%':>15}")
    print(f"   {'Relation Types':<25} {'1':>15} {final['typed_relations']:>15} {'+' + str(final['typed_relations'] - 1):>15}")
    
    print("\n🏷️ Relationship Type Distribution:")
    for rel, count in sorted(relation_types.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   • {rel}: {count}")
    
    # Save final graph
    final_graph_path = os.path.join(config.DATA_DIR, "best_kg", "final_improved_graph.json")
    final_graph_data = {
        'nodes': [{'id': n, **{k: v for k, v in G.nodes[n].items() if not isinstance(v, set)}} for n in G.nodes()],
        'edges': [{'source': u, 'target': v, **G[u][v]} for u, v in G.edges()],
        'statistics': graph_data.get('statistics', {}),
        'improvements': results
    }
    for node_data in final_graph_data['nodes']:
        if 'aliases' in node_data and isinstance(node_data['aliases'], set):
            node_data['aliases'] = list(node_data['aliases'])
    save_json(final_graph_data, final_graph_path)
    
    # Save final results
    results_path = os.path.join(config.DATA_DIR, "final_improvement_results.json")
    results['total_time_seconds'] = time.time() - start_time
    save_json(results, results_path)
    
    print(f"\n💾 Outputs saved:")
    print(f"   • Final graph: {final_graph_path}")
    print(f"   • Results: {results_path}")
    
    total_time = time.time() - start_time
    print(f"\n⏱️ Total time: {total_time:.1f} seconds")
    print("\n" + "="*70)
    print(" ✅ COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
    print("="*70)
    
    return results


if __name__ == "__main__":
    results = run_full_pipeline()
