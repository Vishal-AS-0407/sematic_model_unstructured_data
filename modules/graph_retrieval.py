"""
Graph-Aware Retrieval Module

PURPOSE:
    This module implements retrieval that ACTUALLY USES the knowledge graph structure,
    not just embedding similarity. This is the fundamental fix for proper KG utilization.

KEY INNOVATIONS:
    1. Multi-hop expansion: Include entities N-hops away in graph
    2. Relationship paths: Return HOW entities connect, not just names
    3. Graph-based scoring: Combine embedding similarity with graph distance
    4. Context chains: Generate relationship chains for LLM context

COMPARISON:
    OLD (embedding-only):
        Query → Embed → Cosine similarity → Top-K entities (disconnected)
    
    NEW (graph-aware):
        Query → Embed → Seed entities → Graph expansion → Path scoring → 
        Relationship chains → Rich context

OUTPUT:
    Instead of: [Entity1, Entity2, Entity3]
    Returns:
        Entity1:
        ├── uses → Entity4
        ├── trained_on → Entity5
        └── extends → Entity2
"""
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
import json
import os
import sys
import openai

# Handle imports
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_logger, save_json
import config

logger = get_logger(__name__)


class GraphAwareRetriever:
    """
    Retrieval engine that uses BOTH embeddings AND graph structure.
    
    This fixes the fundamental issue of not using the knowledge graph
    during retrieval.
    """
    
    def __init__(self, 
                 graph_path: str,
                 semantic_model_path: str,
                 max_hops: int = 2,
                 seed_count: int = 5,
                 max_results: int = 20):
        """
        Initialize graph-aware retriever.
        
        Args:
            graph_path: Path to knowledge graph JSON
            semantic_model_path: Path to semantic model JSON
            max_hops: Maximum hops for graph expansion
            seed_count: Number of seed entities from embedding search
            max_results: Maximum results to return
        """
        self.max_hops = max_hops
        self.seed_count = seed_count
        self.max_results = max_results
        
        logger.info("Loading graph-aware retriever...")
        
        # Load embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load knowledge graph
        logger.info(f"Loading knowledge graph from {graph_path}")
        self.G = self._load_graph(graph_path)
        logger.info(f"Graph loaded: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        
        # Load semantic model for topic context
        with open(semantic_model_path, 'r', encoding='utf-8') as f:
            self.semantic_model = json.load(f)
        
        # Build entity embeddings index
        self._build_entity_index()
        
        logger.info("GraphAwareRetriever initialized successfully")
    
    def _load_graph(self, graph_path: str) -> nx.DiGraph:
        """Load graph from JSON."""
        with open(graph_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        G = nx.DiGraph()
        
        for node_data in data['nodes']:
            node_id = node_data['id']
            G.add_node(node_id, **{k: v for k, v in node_data.items() if k != 'id'})
        
        for edge_data in data['edges']:
            source = edge_data['source']
            target = edge_data['target']
            # Get relationship type if available
            relation = edge_data.get('relation', edge_data.get('relations', ['related_to']))
            if isinstance(relation, list):
                relation = relation[0] if relation else 'related_to'
            G.add_edge(source, target, relation=relation, **{
                k: v for k, v in edge_data.items() 
                if k not in ['source', 'target', 'relation']
            })
        
        return G
    
    def _build_entity_index(self):
        """Build embedding index for all entities."""
        logger.info("Building entity embedding index...")
        
        self.entities = list(self.G.nodes())
        
        # Create text representations for entities
        entity_texts = []
        for entity in self.entities:
            node_data = self.G.nodes[entity]
            entity_type = node_data.get('type', '')
            desc = node_data.get('description', '')
            text = f"{entity}. Type: {entity_type}. {desc}"
            entity_texts.append(text)
        
        # Generate embeddings
        self.entity_embeddings = self.embedding_model.encode(entity_texts, show_progress_bar=True)
        
        logger.info(f"Entity index built: {len(self.entities)} entities")
    
    def retrieve(self, query: str) -> Dict:
        """
        Retrieve entities using BOTH embeddings AND graph structure.
        
        Algorithm:
        1. Embed query
        2. Find seed entities via embedding similarity
        3. Expand via graph edges (multi-hop)
        4. Score by combined metric
        5. Return with relationship context
        
        Args:
            query: User query string
            
        Returns:
            Dict with entities, relationships, paths, and context
        """
        logger.info(f"Retrieving for: '{query[:50]}...'")
        
        # Step 1: Embed query
        query_embedding = self.embedding_model.encode(query)
        
        # Step 2: Find seed entities
        seeds = self._get_seed_entities(query_embedding)
        logger.info(f"Found {len(seeds)} seed entities")
        
        # Step 3: Expand via graph
        expanded = self._expand_via_graph(seeds)
        logger.info(f"Expanded to {len(expanded)} entities")
        
        # Step 4: Score all candidates
        scored = self._score_entities(query_embedding, seeds, expanded)
        
        # Step 5: Build relationship context
        result = self._build_result(query, seeds, scored)
        
        return result
    
    def _get_seed_entities(self, query_embedding: np.ndarray) -> List[Tuple[str, float]]:
        """Get initial seed entities via embedding similarity."""
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            self.entity_embeddings
        )[0]
        
        # Get top-k seeds
        top_indices = np.argsort(similarities)[::-1][:self.seed_count]
        
        seeds = [
            (self.entities[i], float(similarities[i]))
            for i in top_indices
            if similarities[i] > 0.3  # Minimum threshold
        ]
        
        return seeds
    
    def _expand_via_graph(self, seeds: List[Tuple[str, float]]) -> Dict[str, Dict]:
        """
        Expand seed entities via graph edges (multi-hop traversal).
        
        This is the KEY function that uses the knowledge graph structure.
        """
        expanded = {}
        visited = set()
        
        # BFS expansion with hop tracking
        queue = [(entity, similarity, 0, []) for entity, similarity in seeds]
        
        while queue:
            entity, seed_sim, hop, path = queue.pop(0)
            
            if entity in visited:
                continue
            visited.add(entity)
            
            if hop > self.max_hops:
                continue
            
            # Store entity with metadata
            if entity not in expanded:
                expanded[entity] = {
                    'seed_similarity': seed_sim,
                    'hop_distance': hop,
                    'path': path.copy(),
                    'relations_in': [],
                    'relations_out': []
                }
            
            # Get neighbors via edges
            if entity in self.G:
                # Outgoing edges
                for neighbor in self.G.successors(entity):
                    if neighbor not in visited:
                        edge_data = self.G[entity][neighbor]
                        relation = edge_data.get('relation', 'related_to')
                        new_path = path + [(entity, relation, neighbor)]
                        
                        expanded[entity]['relations_out'].append({
                            'target': neighbor,
                            'relation': relation
                        })
                        
                        # Decay similarity with distance
                        decayed_sim = seed_sim * (0.7 ** (hop + 1))
                        queue.append((neighbor, decayed_sim, hop + 1, new_path))
                
                # Incoming edges
                for neighbor in self.G.predecessors(entity):
                    if neighbor not in visited:
                        edge_data = self.G[neighbor][entity]
                        relation = edge_data.get('relation', 'related_to')
                        new_path = path + [(neighbor, relation, entity)]
                        
                        expanded[entity]['relations_in'].append({
                            'source': neighbor,
                            'relation': relation
                        })
                        
                        decayed_sim = seed_sim * (0.7 ** (hop + 1))
                        queue.append((neighbor, decayed_sim, hop + 1, new_path))
        
        return expanded
    
    def _score_entities(self, 
                        query_embedding: np.ndarray,
                        seeds: List[Tuple[str, float]],
                        expanded: Dict[str, Dict]) -> List[Dict]:
        """
        Score entities by combined metric:
        - Embedding similarity (semantic relevance)
        - Graph distance from seeds (structural relevance)
        - Node importance (PageRank/centrality)
        - Relationship richness (connectivity)
        """
        scored = []
        seed_entities = {s[0] for s in seeds}
        
        for entity, data in expanded.items():
            # Get embedding similarity
            if entity in self.entities:
                idx = self.entities.index(entity)
                embed_sim = float(cosine_similarity(
                    query_embedding.reshape(1, -1),
                    self.entity_embeddings[idx].reshape(1, -1)
                )[0][0])
            else:
                embed_sim = 0
            
            # Get node importance from graph
            node_data = self.G.nodes.get(entity, {})
            importance = node_data.get('importance', 0.1)
            pagerank = node_data.get('pagerank', 0.001)
            
            # Hop penalty
            hop_penalty = 1.0 / (1 + data['hop_distance'])
            
            # Is seed bonus
            seed_bonus = 0.3 if entity in seed_entities else 0
            
            # Combined score
            score = (
                0.4 * embed_sim +
                0.2 * importance +
                0.15 * pagerank * 100 +  # Scale pagerank
                0.15 * hop_penalty +
                0.1 * seed_bonus
            )
            
            scored.append({
                'entity': entity,
                'score': score,
                'embedding_similarity': embed_sim,
                'importance': importance,
                'hop_distance': data['hop_distance'],
                'is_seed': entity in seed_entities,
                'relations_out': data['relations_out'][:5],
                'relations_in': data['relations_in'][:5],
                'path': data['path']
            })
        
        # Sort by score
        scored.sort(key=lambda x: x['score'], reverse=True)
        
        return scored[:self.max_results]
    
    def _build_result(self, 
                      query: str,
                      seeds: List[Tuple[str, float]],
                      scored: List[Dict]) -> Dict:
        """Build final result with relationship context."""
        
        # Build relationship chains
        relationship_chains = []
        for item in scored[:10]:
            if item['relations_out']:
                for rel in item['relations_out'][:3]:
                    relationship_chains.append({
                        'source': item['entity'],
                        'relation': rel['relation'],
                        'target': rel['target']
                    })
        
        # Generate context text
        context_text = self._generate_context_text(scored[:10], relationship_chains)
        
        return {
            'query': query,
            'entities': [s['entity'] for s in scored],
            'entity_details': scored,
            'seed_entities': [s[0] for s in seeds],
            'relationship_chains': relationship_chains,
            'context_text': context_text,
            'entity_count': len(scored),
            'relationship_count': len(relationship_chains)
        }
    
    def _generate_context_text(self, 
                               entities: List[Dict],
                               relationships: List[Dict]) -> str:
        """Generate human-readable context for LLM."""
        lines = ["## Retrieved Knowledge Context\n"]
        
        lines.append("### Key Entities (ranked by relevance)")
        for i, ent in enumerate(entities[:5], 1):
            node_data = self.G.nodes.get(ent['entity'], {})
            desc = node_data.get('description', '')[:100]
            lines.append(f"{i}. **{ent['entity']}** (score: {ent['score']:.2f})")
            if desc:
                lines.append(f"   - {desc}")
        
        if relationships:
            lines.append("\n### Relationships")
            for rel in relationships[:10]:
                lines.append(f"- {rel['source']} → *{rel['relation']}* → {rel['target']}")
        
        return '\n'.join(lines)


class ContextGenerator:
    """
    Generate rich LLM-ready context from retrieval results.
    
    Uses the knowledge graph to create informative context
    that includes relationship explanations.
    """
    
    def __init__(self, retriever: GraphAwareRetriever):
        self.retriever = retriever
    
    def generate_context(self, query: str, include_llm_summary: bool = True) -> Dict:
        """
        Generate rich context for a query.
        
        Args:
            query: User query
            include_llm_summary: Whether to generate LLM-enhanced summary
            
        Returns:
            Context dict with entities, relationships, and narrative
        """
        # Get retrieval results
        result = self.retriever.retrieve(query)
        
        # Build structured context
        context = {
            'query': query,
            'entities': result['entities'][:10],
            'relationships': result['relationship_chains'][:15],
            'text_context': result['context_text']
        }
        
        if include_llm_summary:
            context['llm_summary'] = self._generate_llm_summary(query, result)
        
        return context
    
    def _generate_llm_summary(self, query: str, result: Dict) -> str:
        """Generate LLM-enhanced summary of retrieved knowledge."""
        try:
            client = openai.OpenAI()
            
            # Build entity and relationship lists
            entities = result['entities'][:8]
            relationships = result['relationship_chains'][:10]
            
            rel_str = '\n'.join([
                f"- {r['source']} {r['relation']} {r['target']}"
                for r in relationships
            ])
            
            prompt = f"""Given this query about AI/ML research:
Query: "{query}"

Retrieved entities: {', '.join(entities)}

Relationships found:
{rel_str}

Write a 2-3 sentence summary that:
1. Directly addresses the query
2. Mentions the most relevant entities
3. Explains key relationships between them

Be concise and technical."""
            
            response = client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a technical assistant summarizing AI research knowledge."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.warning(f"LLM summary failed: {e}")
            return result['context_text']


def run_graph_retrieval_demo():
    """Run demo comparing old vs new retrieval."""
    logger.info("="*60)
    logger.info(" GRAPH-AWARE RETRIEVAL DEMO")
    logger.info("="*60)
    
    # Paths
    graph_path = os.path.join(config.DATA_DIR, "best_kg", "improved_global_graph.json")
    
    # Fallback to original if improved doesn't exist
    if not os.path.exists(graph_path):
        graph_path = os.path.join(config.DATA_DIR, "best_kg", "enhanced_global_graph.json")
    
    model_path = os.path.join(config.DATA_DIR, "best_semantic_models", "improved_semantic_model.json")
    if not os.path.exists(model_path):
        model_path = os.path.join(config.DATA_DIR, "best_semantic_models", "semantic_model_best.json")
    
    # Initialize retriever
    retriever = GraphAwareRetriever(graph_path, model_path)
    context_gen = ContextGenerator(retriever)
    
    # Test queries
    test_queries = [
        "How does Vision Transformer work?",
        "What datasets is GPT-3 trained on?",
        "Explain attention mechanism in transformers",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        context = context_gen.generate_context(query, include_llm_summary=True)
        
        print(f"\n📊 Entities ({len(context['entities'])}):")
        for ent in context['entities'][:5]:
            print(f"   • {ent}")
        
        print(f"\n🔗 Relationships ({len(context['relationships'])}):")
        for rel in context['relationships'][:5]:
            print(f"   • {rel['source']} → {rel['relation']} → {rel['target']}")
        
        print(f"\n📝 LLM Summary:")
        print(f"   {context.get('llm_summary', 'N/A')}")
    
    print("\n" + "="*60)
    print(" DEMO COMPLETE")
    print("="*60)


if __name__ == "__main__":
    run_graph_retrieval_demo()
