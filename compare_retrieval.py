"""
Comprehensive Retrieval Comparison Script

Compares:
1. Old: Embedding-only retrieval
2. New: Graph-aware retrieval with multi-hop

Shows the difference in:
- Entity discovery
- Relationship context
- Context richness
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.hierarchical_retrieval import load_retriever
from modules.graph_retrieval import GraphAwareRetriever, ContextGenerator
import config


def compare_retrieval_methods():
    """Compare old vs new retrieval approaches."""
    print("\n" + "="*70)
    print(" 🔬 RETRIEVAL COMPARISON: OLD (Embedding) vs NEW (Graph-Aware)")
    print("="*70)
    
    # Load old retriever (hierarchical/embedding-based)
    hier_model_path = os.path.join(
        config.DATA_DIR, "best_semantic_models", "hierarchical_semantic_model.json"
    )
    
    old_retriever = None
    if os.path.exists(hier_model_path):
        old_retriever = load_retriever(hier_model_path)
        print("✅ Old retriever loaded (embedding-based)")
    else:
        print("⚠️ Old retriever not available")
    
    # Load new retriever (graph-aware)
    graph_path = os.path.join(config.DATA_DIR, "best_kg", "improved_global_graph.json")
    if not os.path.exists(graph_path):
        graph_path = os.path.join(config.DATA_DIR, "best_kg", "enhanced_global_graph.json")
    
    model_path = os.path.join(config.DATA_DIR, "best_semantic_models", "improved_semantic_model.json")
    if not os.path.exists(model_path):
        model_path = os.path.join(config.DATA_DIR, "best_semantic_models", "semantic_model_best.json")
    
    new_retriever = GraphAwareRetriever(graph_path, model_path)
    context_gen = ContextGenerator(new_retriever)
    print("✅ New retriever loaded (graph-aware)")
    
    # Test queries
    test_queries = [
        "How does Vision Transformer work?",
        "What is self-attention mechanism?",
        "GPT-3 training data and architecture",
    ]
    
    results = []
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"📝 Query: {query}")
        print('='*70)
        
        result = {'query': query}
        
        # Old retrieval
        if old_retriever:
            print("\n🔵 OLD (Embedding-only):")
            old_result = old_retriever.retrieve(query, mode='tree_traversal')
            old_entities = old_result.get('entities', [])[:5]
            print(f"   Entities: {old_entities}")
            print(f"   Count: {len(old_result.get('entities', []))}")
            print(f"   Relationships: ❌ None (not supported)")
            result['old_entities'] = old_entities
            result['old_count'] = len(old_result.get('entities', []))
        
        # New retrieval
        print("\n🟢 NEW (Graph-aware):")
        new_context = context_gen.generate_context(query, include_llm_summary=True)
        new_entities = new_context.get('entities', [])[:5]
        new_rels = new_context.get('relationships', [])[:5]
        
        print(f"   Entities: {new_entities}")
        print(f"   Count: {len(new_context.get('entities', []))}")
        print(f"   Relationships ({len(new_context.get('relationships', []))}):")
        for rel in new_rels:
            print(f"      • {rel['source']} → {rel['relation']} → {rel['target']}")
        
        if 'llm_summary' in new_context:
            print(f"\n   📋 LLM Summary:")
            print(f"      {new_context['llm_summary'][:200]}...")
        
        result['new_entities'] = new_entities
        result['new_count'] = len(new_context.get('entities', []))
        result['new_relationships'] = len(new_context.get('relationships', []))
        results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print(" 📊 COMPARISON SUMMARY")
    print("="*70)
    
    print("\n| Query | Old Entities | New Entities | Relationships |")
    print("|-------|--------------|--------------|---------------|")
    for r in results:
        old_count = r.get('old_count', 'N/A')
        new_count = r.get('new_count', 0)
        rels = r.get('new_relationships', 0)
        print(f"| {r['query'][:25]}... | {old_count} | {new_count} | {rels} |")
    
    print("\n✅ Key Improvements:")
    print("   1. Graph-aware: Uses knowledge graph edges for expansion")
    print("   2. Multi-hop: Discovers connected entities via graph traversal")
    print("   3. Relationships: Returns HOW entities are connected")
    print("   4. Context: LLM-ready context with relationship narrative")
    
    # Save comparison results
    output_path = os.path.join(config.DATA_DIR, "retrieval_comparison.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📁 Comparison saved to: {output_path}")


if __name__ == "__main__":
    compare_retrieval_methods()
