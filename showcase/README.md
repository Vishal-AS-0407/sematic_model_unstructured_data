# 🚀 Semantic Model Showcase

## Project: Graph-Enhanced Hierarchical Semantic Model

This showcase demonstrates the improvements made to the semantic model through
graph-aware retrieval, multi-hop reasoning, and typed relationships.

---

## 📊 Key Results

### Before vs After Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Graph Edges** | 304 | 822 | +518 |
| **Avg Degree** | 1.00 | 2.70 | +170% |
| **Relation Types** | 1 | 10 | +9 types |
| **Coverage** | 38% | 43% | +5% |

### Phase Execution Times

| Phase | Description | Time |
|-------|-------------|------|
| Phase 1 | Relationship Inference + Noise Rescue | 68.8s |
| Phase 2 | Graph-Aware Retrieval | 18.5s |
| Phase 3 | Typed Relationships | 8.5s |
| **Total** | Complete Pipeline | 95.9s |

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
| `uses` | 222 | Vision Transformer → uses → Self-Attention |
| `evaluated_on` | 100 | Model → evaluated_on → ImageNet |
| `authored_by` | 68 | Paper → authored_by → Researcher |
| `trained_on` | 55 | GPT-3 → trained_on → Common Crawl |
| `part_of` | 13 | Encoder → part_of → Transformer |

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

Generated: 2025-12-11 01:26:53
