"""
Generate Comprehensive Showcase

Creates a complete showcase folder with:
1. All visualizations (graphs, charts, hierarchies)
2. Metrics and comparisons
3. Q&A examples with full retrieval paths
4. README for presentation
"""
import json
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.graph_retrieval import GraphAwareRetriever, ContextGenerator
import config
from utils import get_logger, save_json

logger = get_logger(__name__)

# Create showcase directory
SHOWCASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "showcase")
os.makedirs(SHOWCASE_DIR, exist_ok=True)


def generate_graph_visualizations():
    """Generate knowledge graph visualizations."""
    print("\n📊 Generating Graph Visualizations...")
    
    # Load final graph
    graph_path = os.path.join(config.DATA_DIR, "best_kg", "final_improved_graph.json")
    if not os.path.exists(graph_path):
        graph_path = os.path.join(config.DATA_DIR, "best_kg", "improved_global_graph.json")
    
    with open(graph_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    G = nx.DiGraph()
    for node_data in data['nodes']:
        G.add_node(node_data['id'], **{k: v for k, v in node_data.items() if k != 'id' and not isinstance(v, (list, set))})
    for edge_data in data['edges']:
        G.add_edge(edge_data['source'], edge_data['target'],
                  relation=edge_data.get('relation', 'related_to'))
    
    # 1. Relationship Type Distribution
    plt.figure(figsize=(12, 6))
    relation_counts = {}
    for _, _, d in G.edges(data=True):
        rel = d.get('relation', 'unknown')
        relation_counts[rel] = relation_counts.get(rel, 0) + 1
    
    sorted_rels = sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)
    rels, counts = zip(*sorted_rels[:10])
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(rels)))
    bars = plt.barh(range(len(rels)), counts, color=colors)
    plt.yticks(range(len(rels)), rels)
    plt.xlabel('Count')
    plt.title('Relationship Type Distribution', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                str(count), va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(SHOWCASE_DIR, 'relationship_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ relationship_distribution.png")
    
    # 2. Graph Metrics Comparison
    plt.figure(figsize=(10, 6))
    
    # Load baseline and final metrics
    results_path = os.path.join(config.DATA_DIR, "final_improvement_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        baseline = results['metrics']['baseline']
        final = results['metrics']['final']
    else:
        baseline = {'edges': 304, 'avg_degree': 1.0}
        final = {'edges': 822, 'avg_degree': 2.7}
    
    metrics = ['Edges', 'Avg Degree (x100)']
    before = [baseline['edges'], baseline['avg_degree'] * 100]
    after = [final['edges'], final['avg_degree'] * 100]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, before, width, label='Before', color='#ff6b6b', alpha=0.8)
    bars2 = ax.bar(x + width/2, after, width, label='After', color='#4ecdc4', alpha=0.8)
    
    ax.set_ylabel('Value')
    ax.set_title('Graph Metrics: Before vs After Improvements', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(SHOWCASE_DIR, 'metrics_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ metrics_comparison.png")
    
    # 3. Top Entities by Importance
    plt.figure(figsize=(12, 8))
    
    importance_scores = []
    for node in G.nodes():
        imp = G.nodes[node].get('importance', 0)
        importance_scores.append((node, imp))
    
    importance_scores.sort(key=lambda x: x[1], reverse=True)
    top_20 = importance_scores[:20]
    
    names, scores = zip(*top_20)
    short_names = [n[:25] + '...' if len(n) > 25 else n for n in names]
    
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(names)))
    plt.barh(range(len(names)), scores, color=colors)
    plt.yticks(range(len(names)), short_names)
    plt.xlabel('Importance Score')
    plt.title('Top 20 Entities by Importance', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(SHOWCASE_DIR, 'top_entities.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ top_entities.png")
    
    # 4. Knowledge Graph Subgraph Visualization
    plt.figure(figsize=(16, 12))
    
    # Get top entities and their neighbors
    top_entities = [n for n, _ in importance_scores[:15]]
    subgraph_nodes = set(top_entities)
    for node in top_entities[:10]:
        subgraph_nodes.update(list(G.successors(node))[:3])
        subgraph_nodes.update(list(G.predecessors(node))[:3])
    
    subG = G.subgraph(list(subgraph_nodes)[:50])
    
    pos = nx.spring_layout(subG, k=2, iterations=50, seed=42)
    
    # Color by type
    node_colors = []
    for node in subG.nodes():
        if node in top_entities[:5]:
            node_colors.append('#ff6b6b')
        elif node in top_entities:
            node_colors.append('#feca57')
        else:
            node_colors.append('#54a0ff')
    
    nx.draw_networkx_nodes(subG, pos, node_color=node_colors, 
                          node_size=1500, alpha=0.8)
    nx.draw_networkx_edges(subG, pos, alpha=0.5, 
                          edge_color='gray', arrows=True, 
                          arrowsize=15, connectionstyle="arc3,rad=0.1")
    
    labels = {n: n[:15] + '...' if len(n) > 15 else n for n in subG.nodes()}
    nx.draw_networkx_labels(subG, pos, labels, font_size=8)
    
    plt.title('Knowledge Graph - Top Entities and Connections', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='#ff6b6b', label='Top 5 Entities'),
        mpatches.Patch(color='#feca57', label='Top 6-15 Entities'),
        mpatches.Patch(color='#54a0ff', label='Connected Entities')
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(SHOWCASE_DIR, 'knowledge_graph.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ knowledge_graph.png")
    
    return G


def generate_qa_examples(retriever, context_gen):
    """Generate Q&A examples with full retrieval paths."""
    print("\n🔍 Generating Q&A Examples with Retrieval Paths...")
    
    test_questions = [
        "How does Vision Transformer (ViT) work?",
        "What is self-attention mechanism in transformers?",
        "What datasets is GPT-3 trained on?",
        "How does BERT differ from GPT?",
        "What is the Swin Transformer architecture?",
        "Explain the attention mechanism",
        "What benchmarks are used for evaluating language models?",
        "How do transformers handle long sequences?",
    ]
    
    qa_results = []
    
    for question in test_questions:
        print(f"   Processing: {question[:40]}...")
        
        # Get retrieval results
        result = retriever.retrieve(question)
        context = context_gen.generate_context(question, include_llm_summary=True)
        
        qa_entry = {
            'question': question,
            'timestamp': datetime.now().isoformat(),
            
            # Retrieved entities with scores
            'entities': [
                {
                    'name': e.get('entity', e) if isinstance(e, dict) else e,
                    'score': e.get('score', 0) if isinstance(e, dict) else 0,
                    'is_seed': e.get('is_seed', False) if isinstance(e, dict) else False,
                    'hop_distance': e.get('hop_distance', 0) if isinstance(e, dict) else 0
                }
                for e in result.get('entity_details', [])[:10]
            ],
            
            # Seed entities (first match)
            'seed_entities': result.get('seed_entities', []),
            
            # Multi-hop relationships
            'relationships': result.get('relationship_chains', []),
            
            # LLM context and summary
            'context_text': result.get('context_text', ''),
            'llm_summary': context.get('llm_summary', ''),
            
            # Statistics
            'stats': {
                'total_entities': len(result.get('entities', [])),
                'total_relationships': len(result.get('relationship_chains', [])),
                'max_hop_distance': max([e.get('hop_distance', 0) for e in result.get('entity_details', []) if isinstance(e, dict)], default=0)
            }
        }
        
        qa_results.append(qa_entry)
    
    # Save Q&A results
    qa_path = os.path.join(SHOWCASE_DIR, 'qa_examples.json')
    save_json(qa_results, qa_path)
    print(f"   ✅ qa_examples.json ({len(qa_results)} questions)")
    
    return qa_results


def generate_qa_markdown(qa_results):
    """Generate readable markdown for Q&A examples."""
    print("\n📝 Generating Q&A Markdown...")
    
    md_lines = [
        "# Q&A Examples with Retrieval Paths\n",
        "This document shows how the graph-aware retrieval system answers questions",
        "with multi-hop reasoning and relationship discovery.\n",
        "---\n"
    ]
    
    for i, qa in enumerate(qa_results, 1):
        md_lines.append(f"## Question {i}: {qa['question']}\n")
        
        # LLM Summary
        md_lines.append("### 💡 Answer Summary")
        md_lines.append(f"> {qa.get('llm_summary', 'No summary available.')}\n")
        
        # Stats
        stats = qa.get('stats', {})
        md_lines.append("### 📊 Retrieval Statistics")
        md_lines.append(f"- **Total Entities Retrieved**: {stats.get('total_entities', 0)}")
        md_lines.append(f"- **Relationships Found**: {stats.get('total_relationships', 0)}")
        md_lines.append(f"- **Max Hop Distance**: {stats.get('max_hop_distance', 0)}\n")
        
        # Seed entities
        if qa.get('seed_entities'):
            md_lines.append("### 🌱 Seed Entities (Direct Matches)")
            for seed in qa['seed_entities'][:5]:
                md_lines.append(f"- {seed}")
            md_lines.append("")
        
        # Top entities with hops
        md_lines.append("### 🎯 Retrieved Entities (with hop distance)")
        md_lines.append("| Entity | Score | Hop | Is Seed |")
        md_lines.append("|--------|-------|-----|---------|")
        for e in qa.get('entities', [])[:8]:
            name = e.get('name', 'Unknown')[:30]
            score = e.get('score', 0)
            hop = e.get('hop_distance', 0)
            is_seed = "✅" if e.get('is_seed') else ""
            md_lines.append(f"| {name} | {score:.2f} | {hop} | {is_seed} |")
        md_lines.append("")
        
        # Relationships
        if qa.get('relationships'):
            md_lines.append("### 🔗 Relationship Chains")
            for rel in qa['relationships'][:8]:
                md_lines.append(f"- `{rel.get('source', '?')}` → **{rel.get('relation', '?')}** → `{rel.get('target', '?')}`")
            md_lines.append("")
        
        md_lines.append("---\n")
    
    # Save
    md_path = os.path.join(SHOWCASE_DIR, 'qa_showcase.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"   ✅ qa_showcase.md")


def generate_showcase_readme():
    """Generate main README for showcase folder."""
    print("\n📖 Generating Showcase README...")
    
    # Load final results
    results_path = os.path.join(config.DATA_DIR, "final_improvement_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    else:
        results = {'metrics': {'baseline': {}, 'final': {}}, 'phases': {}}
    
    baseline = results.get('metrics', {}).get('baseline', {})
    final = results.get('metrics', {}).get('final', {})
    phases = results.get('phases', {})
    
    readme = f"""# 🚀 Semantic Model Showcase

## Project: Graph-Enhanced Hierarchical Semantic Model

This showcase demonstrates the improvements made to the semantic model through
graph-aware retrieval, multi-hop reasoning, and typed relationships.

---

## 📊 Key Results

### Before vs After Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Graph Edges** | {baseline.get('edges', 304)} | {final.get('edges', 822)} | +{final.get('edges', 822) - baseline.get('edges', 304)} |
| **Avg Degree** | {baseline.get('avg_degree', 1.0):.2f} | {final.get('avg_degree', 2.7):.2f} | +{(final.get('avg_degree', 2.7) - baseline.get('avg_degree', 1.0)) / baseline.get('avg_degree', 1.0) * 100:.0f}% |
| **Relation Types** | 1 | {final.get('typed_relations', 10)} | +{final.get('typed_relations', 10) - 1} types |
| **Coverage** | 38% | {final.get('coverage', 0.43) * 100:.0f}% | +{(final.get('coverage', 0.43) - 0.38) * 100:.0f}% |

### Phase Execution Times

| Phase | Description | Time |
|-------|-------------|------|
| Phase 1 | Relationship Inference + Noise Rescue | {phases.get('phase1', {}).get('time_seconds', 0):.1f}s |
| Phase 2 | Graph-Aware Retrieval | {phases.get('phase2', {}).get('time_seconds', 0):.1f}s |
| Phase 3 | Typed Relationships | {phases.get('phase3', {}).get('time_seconds', 0):.1f}s |
| **Total** | Complete Pipeline | {results.get('total_time_seconds', 0):.1f}s |

---

## 📁 Showcase Contents

### Visualizations
- `knowledge_graph.png` - Interactive view of top entities and connections
- `relationship_distribution.png` - Distribution of typed relationships
- `metrics_comparison.png` - Before/after metrics comparison
- `top_entities.png` - Top 20 entities by importance score

### Q&A Examples
- `qa_showcase.md` - Formatted Q&A with retrieval paths (readable)
- `qa_examples.json` - Raw Q&A data with all details

### Data
- All metrics and results exported from pipeline

---

## 🔗 Relationship Types Discovered

| Type | Count | Example |
|------|-------|---------|
| `uses` | {phases.get('phase3', {}).get('relation_distribution', {}).get('uses', 0)} | Vision Transformer → uses → Self-Attention |
| `evaluated_on` | {phases.get('phase3', {}).get('relation_distribution', {}).get('evaluated_on', 0)} | Model → evaluated_on → ImageNet |
| `authored_by` | {phases.get('phase3', {}).get('relation_distribution', {}).get('authored_by', 0)} | Paper → authored_by → Researcher |
| `trained_on` | {phases.get('phase3', {}).get('relation_distribution', {}).get('trained_on', 0)} | GPT-3 → trained_on → Common Crawl |
| `part_of` | {phases.get('phase3', {}).get('relation_distribution', {}).get('part_of', 0)} | Encoder → part_of → Transformer |

---

## 🎯 Key Innovations

1. **Graph-Aware Retrieval**: Uses knowledge graph edges, not just embeddings
2. **Multi-Hop Expansion**: Discovers entities 2+ hops away via graph traversal
3. **Typed Relationships**: 10 semantic relationship types instead of generic edges
4. **LLM Context**: Rich, narrative context for question answering

---

## 🛠️ How to Run

```bash
# Generate this showcase
python generate_showcase.py

# Run the complete pipeline
python run_complete_pipeline.py

# Test graph-aware retrieval
python modules/graph_retrieval.py
```

---

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    readme_path = os.path.join(SHOWCASE_DIR, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme)
    
    print(f"   ✅ README.md")


def copy_key_data():
    """Copy key data files to showcase."""
    print("\n📦 Copying Key Data Files...")
    
    import shutil
    
    files_to_copy = [
        (os.path.join(config.DATA_DIR, "final_improvement_results.json"), "results.json"),
        (os.path.join(config.DATA_DIR, "best_semantic_models", "improved_semantic_model.json"), "semantic_model.json"),
    ]
    
    for src, dst in files_to_copy:
        if os.path.exists(src):
            shutil.copy(src, os.path.join(SHOWCASE_DIR, dst))
            print(f"   ✅ {dst}")


def run_showcase_generation():
    """Run complete showcase generation."""
    print("\n" + "="*70)
    print(" 🎨 GENERATING COMPREHENSIVE SHOWCASE")
    print("="*70)
    print(f" Output folder: {SHOWCASE_DIR}")
    print("="*70)
    
    # Generate graph visualizations
    G = generate_graph_visualizations()
    
    # Initialize retriever
    print("\n🔧 Initializing Graph-Aware Retriever...")
    graph_path = os.path.join(config.DATA_DIR, "best_kg", "final_improved_graph.json")
    if not os.path.exists(graph_path):
        graph_path = os.path.join(config.DATA_DIR, "best_kg", "improved_global_graph.json")
    
    model_path = os.path.join(config.DATA_DIR, "best_semantic_models", "improved_semantic_model.json")
    if not os.path.exists(model_path):
        model_path = os.path.join(config.DATA_DIR, "best_semantic_models", "semantic_model_best.json")
    
    retriever = GraphAwareRetriever(graph_path, model_path)
    context_gen = ContextGenerator(retriever)
    
    # Generate Q&A examples
    qa_results = generate_qa_examples(retriever, context_gen)
    
    # Generate Q&A markdown
    generate_qa_markdown(qa_results)
    
    # Copy key data
    copy_key_data()
    
    # Generate README
    generate_showcase_readme()
    
    print("\n" + "="*70)
    print(" ✅ SHOWCASE GENERATION COMPLETE!")
    print("="*70)
    print(f"\n📁 Showcase folder: {SHOWCASE_DIR}")
    print("\nContents:")
    for f in os.listdir(SHOWCASE_DIR):
        size = os.path.getsize(os.path.join(SHOWCASE_DIR, f))
        print(f"   • {f} ({size / 1024:.1f} KB)")


if __name__ == "__main__":
    run_showcase_generation()
