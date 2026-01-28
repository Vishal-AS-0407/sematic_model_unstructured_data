"""
Visualization Script for Hierarchy and Semantic Model

Creates interactive HTML visualizations:
1. Collapsible hierarchy tree
2. Topic relationship network
3. Knowledge graph interactive view
4. Sunburst chart for hierarchical entity view

All outputs saved to: showcase/visualizations/
"""

import json
import os

# Output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "showcase", "visualizations")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_hierarchical_model():
    """Load the hierarchical semantic model."""
    path = os.path.join(DATA_DIR, "best_semantic_models", "hierarchical_semantic_model.json")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_knowledge_graph():
    """Load the knowledge graph."""
    path = os.path.join(DATA_DIR, "best_kg", "final_improved_graph.json")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_semantic_model():
    """Load the flat semantic model."""
    path = os.path.join(SCRIPT_DIR, "showcase", "semantic_model.json")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_hierarchy_tree_html(hierarchy_data):
    """Generate interactive collapsible hierarchy tree using D3.js."""
    
    tree = hierarchy_data.get("hierarchical_tree", {}).get("hierarchy", {})
    metadata = hierarchy_data.get("hierarchical_tree", {}).get("metadata", {})
    
    # Convert tree to D3-compatible format
    def convert_node(node):
        result = {
            "name": node.get("name", "Unknown"),
            "id": node.get("id", ""),
            "type": node.get("type", "unknown"),
            "level": node.get("level", 0),
            "entity_count": node.get("entity_count", 0),
            "summary": node.get("summary", "")[:200] + "..." if len(node.get("summary", "")) > 200 else node.get("summary", ""),
            "key_entities": node.get("key_entities", [])[:5]
        }
        if "children" in node and node["children"]:
            result["children"] = [convert_node(child) for child in node["children"]]
        return result
    
    tree_json = json.dumps(convert_node(tree))
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hierarchy Tree - Semantic Model</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: #e8e8e8;
        }}
        .header {{
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(10px);
            padding: 20px 40px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .header h1 {{
            font-size: 28px;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }}
        .header .stats {{
            display: flex;
            gap: 30px;
            color: #a0a0a0;
            font-size: 14px;
        }}
        .header .stats span {{
            color: #00d4ff;
            font-weight: 600;
        }}
        #tree-container {{
            width: 100%;
            height: calc(100vh - 100px);
            overflow: auto;
        }}
        .node circle {{
            stroke-width: 2px;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        .node circle:hover {{
            stroke-width: 4px;
            filter: brightness(1.3);
        }}
        .node text {{
            font-size: 12px;
            fill: #e8e8e8;
            pointer-events: none;
        }}
        .link {{
            fill: none;
            stroke: rgba(255,255,255,0.2);
            stroke-width: 1.5px;
        }}
        .tooltip {{
            position: absolute;
            background: rgba(20, 20, 40, 0.95);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 12px;
            padding: 15px;
            max-width: 350px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            z-index: 1000;
        }}
        .tooltip h3 {{
            color: #00d4ff;
            margin-bottom: 10px;
            font-size: 16px;
        }}
        .tooltip p {{
            font-size: 13px;
            line-height: 1.5;
            color: #c0c0c0;
            margin-bottom: 10px;
        }}
        .tooltip .entities {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
        }}
        .tooltip .entity-tag {{
            background: rgba(123, 44, 191, 0.3);
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            color: #e0b0ff;
        }}
        .legend {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(20, 20, 40, 0.9);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 5px 0;
            font-size: 12px;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🌳 Hierarchical Semantic Model</h1>
        <div class="stats">
            <div>Total Levels: <span>{metadata.get('total_levels', 5)}</span></div>
            <div>Total Nodes: <span>{metadata.get('total_nodes', 27)}</span></div>
            <div>Leaf Topics: <span>{metadata.get('leaf_count', 14)}</span></div>
        </div>
    </div>
    <div id="tree-container"></div>
    <div class="tooltip" id="tooltip"></div>
    <div class="legend">
        <div class="legend-item"><div class="legend-color" style="background: #ff6b6b;"></div> Root</div>
        <div class="legend-item"><div class="legend-color" style="background: #4ecdc4;"></div> Branch</div>
        <div class="legend-item"><div class="legend-color" style="background: #45b7d1;"></div> Leaf</div>
    </div>
    <script>
        const data = {tree_json};
        
        const container = document.getElementById('tree-container');
        const width = Math.max(1400, container.clientWidth);
        const height = Math.max(800, container.clientHeight);
        
        const svg = d3.select('#tree-container')
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .append('g')
            .attr('transform', 'translate(80, 40)');
        
        const tree = d3.tree().size([height - 100, width - 200]);
        const root = d3.hierarchy(data);
        tree(root);
        
        // Links
        svg.selectAll('.link')
            .data(root.links())
            .join('path')
            .attr('class', 'link')
            .attr('d', d3.linkHorizontal()
                .x(d => d.y)
                .y(d => d.x));
        
        // Nodes
        const node = svg.selectAll('.node')
            .data(root.descendants())
            .join('g')
            .attr('class', 'node')
            .attr('transform', d => `translate(${{d.y}},${{d.x}})`);
        
        const getColor = (type) => {{
            if (type === 'root') return '#ff6b6b';
            if (type === 'branch') return '#4ecdc4';
            return '#45b7d1';
        }};
        
        node.append('circle')
            .attr('r', d => 8 + d.data.entity_count / 5)
            .attr('fill', d => getColor(d.data.type))
            .attr('stroke', d => d3.color(getColor(d.data.type)).darker(0.5))
            .on('mouseover', function(event, d) {{
                const tooltip = document.getElementById('tooltip');
                tooltip.innerHTML = `
                    <h3>${{d.data.name}}</h3>
                    <p><strong>Type:</strong> ${{d.data.type}} | <strong>Level:</strong> ${{d.data.level}} | <strong>Entities:</strong> ${{d.data.entity_count}}</p>
                    <p>${{d.data.summary}}</p>
                    <div class="entities">
                        ${{d.data.key_entities.map(e => `<span class="entity-tag">${{e}}</span>`).join('')}}
                    </div>
                `;
                tooltip.style.opacity = 1;
                tooltip.style.left = (event.pageX + 15) + 'px';
                tooltip.style.top = (event.pageY - 10) + 'px';
            }})
            .on('mouseout', function() {{
                document.getElementById('tooltip').style.opacity = 0;
            }});
        
        node.append('text')
            .attr('dy', 4)
            .attr('x', d => d.children ? -12 : 12)
            .attr('text-anchor', d => d.children ? 'end' : 'start')
            .text(d => d.data.name.length > 20 ? d.data.name.substring(0, 20) + '...' : d.data.name);
    </script>
</body>
</html>'''
    
    output_path = os.path.join(OUTPUT_DIR, "hierarchy_interactive.html")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"   ✅ hierarchy_interactive.html")
    return output_path


def generate_topic_network_html(semantic_model):
    """Generate interactive topic relationship network."""
    
    topics = semantic_model.get("semantic_topics", [])
    
    # Build nodes and links
    nodes = []
    links = []
    
    for topic in topics:
        nodes.append({
            "id": topic.get("topic_id", ""),
            "name": topic.get("topic_name", "Unknown"),
            "score": topic.get("score", 0),
            "entity_count": topic.get("entity_count", 0),
            "keywords": topic.get("keywords", [])[:5],
            "description": topic.get("description", "")[:150] + "..."
        })
        
        for rel in topic.get("related_topics", []):
            links.append({
                "source": topic.get("topic_id", ""),
                "target": rel.get("topic_id", ""),
                "similarity": rel.get("similarity", 0)
            })
    
    nodes_json = json.dumps(nodes)
    links_json = json.dumps(links)
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Topic Relationship Network</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #21262d 100%);
            min-height: 100vh;
            color: #e8e8e8;
        }}
        .header {{
            background: rgba(255,255,255,0.03);
            backdrop-filter: blur(10px);
            padding: 20px 40px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .header h1 {{
            font-size: 28px;
            background: linear-gradient(90deg, #58a6ff, #a371f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        #network-container {{
            width: 100%;
            height: calc(100vh - 80px);
        }}
        .tooltip {{
            position: absolute;
            background: rgba(30, 30, 50, 0.95);
            border: 1px solid rgba(88, 166, 255, 0.3);
            border-radius: 12px;
            padding: 15px;
            max-width: 320px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.6);
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            z-index: 1000;
        }}
        .tooltip h3 {{
            color: #58a6ff;
            margin-bottom: 8px;
        }}
        .tooltip p {{
            font-size: 12px;
            line-height: 1.5;
            color: #a0a0a0;
            margin-bottom: 8px;
        }}
        .tooltip .keywords {{
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
        }}
        .tooltip .keyword {{
            background: rgba(163, 113, 247, 0.2);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 10px;
            color: #c9b1ff;
        }}
        .node {{
            cursor: grab;
        }}
        .node:active {{
            cursor: grabbing;
        }}
        .link {{
            stroke: rgba(88, 166, 255, 0.3);
            stroke-opacity: 0.6;
        }}
        .instructions {{
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: rgba(30, 30, 50, 0.8);
            padding: 12px;
            border-radius: 8px;
            font-size: 12px;
            color: #a0a0a0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔗 Topic Relationship Network</h1>
    </div>
    <div id="network-container"></div>
    <div class="tooltip" id="tooltip"></div>
    <div class="instructions">
        🖱️ Drag nodes to rearrange | Hover for details
    </div>
    <script>
        const nodes = {nodes_json};
        const links = {links_json};
        
        const container = document.getElementById('network-container');
        const width = container.clientWidth;
        const height = container.clientHeight;
        
        const svg = d3.select('#network-container')
            .append('svg')
            .attr('width', width)
            .attr('height', height);
        
        const simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id).distance(150))
            .force('charge', d3.forceManyBody().strength(-400))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(50));
        
        const colorScale = d3.scaleOrdinal(d3.schemeTableau10);
        
        const link = svg.selectAll('.link')
            .data(links)
            .join('line')
            .attr('class', 'link')
            .attr('stroke-width', d => d.similarity * 5 + 1);
        
        const node = svg.selectAll('.node')
            .data(nodes)
            .join('g')
            .attr('class', 'node')
            .call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended));
        
        node.append('circle')
            .attr('r', d => 15 + d.score * 30)
            .attr('fill', (d, i) => colorScale(i))
            .attr('stroke', '#fff')
            .attr('stroke-width', 2)
            .on('mouseover', function(event, d) {{
                const tooltip = document.getElementById('tooltip');
                tooltip.innerHTML = `
                    <h3>${{d.name}}</h3>
                    <p><strong>Score:</strong> ${{d.score.toFixed(3)}} | <strong>Entities:</strong> ${{d.entity_count}}</p>
                    <p>${{d.description}}</p>
                    <div class="keywords">
                        ${{d.keywords.map(k => `<span class="keyword">${{k}}</span>`).join('')}}
                    </div>
                `;
                tooltip.style.opacity = 1;
                tooltip.style.left = (event.pageX + 15) + 'px';
                tooltip.style.top = (event.pageY - 10) + 'px';
            }})
            .on('mouseout', function() {{
                document.getElementById('tooltip').style.opacity = 0;
            }});
        
        node.append('text')
            .text(d => d.name.length > 15 ? d.name.substring(0, 15) + '...' : d.name)
            .attr('text-anchor', 'middle')
            .attr('dy', d => 25 + d.score * 30)
            .attr('fill', '#e0e0e0')
            .attr('font-size', '11px');
        
        simulation.on('tick', () => {{
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            
            node.attr('transform', d => `translate(${{d.x}},${{d.y}})`);
        }});
        
        function dragstarted(event) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }}
        
        function dragged(event) {{
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }}
        
        function dragended(event) {{
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }}
    </script>
</body>
</html>'''
    
    output_path = os.path.join(OUTPUT_DIR, "topic_network.html")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"   ✅ topic_network.html")
    return output_path


def generate_knowledge_graph_html(kg_data):
    """Generate interactive knowledge graph visualization."""
    
    nodes = kg_data.get("nodes", [])[:100]  # Limit for performance
    edges = kg_data.get("edges", [])
    
    # Filter edges to only include existing nodes
    node_ids = {n['id'] for n in nodes}
    filtered_edges = [e for e in edges if e.get('source') in node_ids and e.get('target') in node_ids][:200]
    
    # Prepare nodes data
    kg_nodes = []
    for n in nodes:
        kg_nodes.append({
            "id": n.get("id", ""),
            "type": n.get("type", "unknown"),
            "importance": n.get("importance", 0),
            "description": n.get("description", "")[:100],
            "community": n.get("community", 0)
        })
    
    # Prepare edges data
    kg_links = []
    for e in filtered_edges:
        kg_links.append({
            "source": e.get("source", ""),
            "target": e.get("target", ""),
            "relation": e.get("relation", "related_to")
        })
    
    nodes_json = json.dumps(kg_nodes)
    links_json = json.dumps(kg_links)
    
    # Get unique types for legend
    types = list(set(n['type'] for n in kg_nodes))
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Knowledge Graph</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a0a2e 0%, #2d1b4e 50%, #1a2a4e 100%);
            min-height: 100vh;
            color: #e8e8e8;
        }}
        .header {{
            background: rgba(255,255,255,0.03);
            backdrop-filter: blur(10px);
            padding: 15px 30px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header h1 {{
            font-size: 24px;
            background: linear-gradient(90deg, #f093fb, #f5576c);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .filters {{
            display: flex;
            gap: 10px;
        }}
        .filter-btn {{
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            color: #e8e8e8;
            padding: 6px 12px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s;
        }}
        .filter-btn:hover, .filter-btn.active {{
            background: rgba(240, 147, 251, 0.3);
            border-color: #f093fb;
        }}
        #graph-container {{
            width: 100%;
            height: calc(100vh - 60px);
        }}
        .tooltip {{
            position: absolute;
            background: rgba(30, 20, 50, 0.95);
            border: 1px solid rgba(240, 147, 251, 0.3);
            border-radius: 10px;
            padding: 12px;
            max-width: 280px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.6);
            pointer-events: none;
            opacity: 0;
            z-index: 1000;
        }}
        .tooltip h3 {{ color: #f093fb; margin-bottom: 6px; font-size: 14px; }}
        .tooltip p {{ font-size: 11px; color: #a0a0a0; margin: 4px 0; }}
        .tooltip .type-badge {{
            display: inline-block;
            background: rgba(245, 87, 108, 0.3);
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 10px;
            color: #ffb3c1;
        }}
        .legend {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(30, 20, 50, 0.9);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.1);
            max-height: 300px;
            overflow-y: auto;
        }}
        .legend-title {{ font-size: 14px; margin-bottom: 10px; color: #f093fb; }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 4px 0;
            font-size: 11px;
        }}
        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Knowledge Graph</h1>
        <div class="filters">
            <button class="filter-btn active" data-type="all">All</button>
            {' '.join([f'<button class="filter-btn" data-type="{t}">{t}</button>' for t in types[:8]])}
        </div>
    </div>
    <div id="graph-container"></div>
    <div class="tooltip" id="tooltip"></div>
    <div class="legend">
        <div class="legend-title">Entity Types</div>
        {' '.join([f'<div class="legend-item"><div class="legend-color" style="background: hsl({i*360//len(types)}, 70%, 60%);"></div>{t}</div>' for i, t in enumerate(types)])}
    </div>
    <script>
        const nodes = {nodes_json};
        const links = {links_json};
        const types = {json.dumps(types)};
        
        const container = document.getElementById('graph-container');
        const width = container.clientWidth;
        const height = container.clientHeight;
        
        const svg = d3.select('#graph-container')
            .append('svg')
            .attr('width', width)
            .attr('height', height);
        
        const g = svg.append('g');
        
        // Zoom behavior
        svg.call(d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', (event) => g.attr('transform', event.transform)));
        
        const simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id).distance(80))
            .force('charge', d3.forceManyBody().strength(-200))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(30));
        
        const colorScale = (type) => {{
            const idx = types.indexOf(type);
            return `hsl(${{idx * 360 / types.length}}, 70%, 60%)`;
        }};
        
        const link = g.selectAll('.link')
            .data(links)
            .join('line')
            .attr('stroke', 'rgba(255,255,255,0.15)')
            .attr('stroke-width', 1);
        
        const node = g.selectAll('.node')
            .data(nodes)
            .join('g')
            .attr('class', 'node')
            .call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended));
        
        node.append('circle')
            .attr('r', d => 6 + d.importance * 20)
            .attr('fill', d => colorScale(d.type))
            .attr('stroke', '#fff')
            .attr('stroke-width', 1.5)
            .on('mouseover', function(event, d) {{
                const tooltip = document.getElementById('tooltip');
                tooltip.innerHTML = `
                    <h3>${{d.id}}</h3>
                    <p><span class="type-badge">${{d.type}}</span></p>
                    <p><strong>Importance:</strong> ${{d.importance.toFixed(3)}}</p>
                    <p>${{d.description}}</p>
                `;
                tooltip.style.opacity = 1;
                tooltip.style.left = (event.pageX + 15) + 'px';
                tooltip.style.top = (event.pageY - 10) + 'px';
            }})
            .on('mouseout', function() {{
                document.getElementById('tooltip').style.opacity = 0;
            }});
        
        node.append('text')
            .text(d => d.id.length > 12 ? d.id.substring(0, 12) + '...' : d.id)
            .attr('text-anchor', 'middle')
            .attr('dy', d => 12 + d.importance * 20)
            .attr('fill', '#c0c0c0')
            .attr('font-size', '9px');
        
        simulation.on('tick', () => {{
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            node.attr('transform', d => `translate(${{d.x}},${{d.y}})`);
        }});
        
        // Filter functionality
        document.querySelectorAll('.filter-btn').forEach(btn => {{
            btn.addEventListener('click', function() {{
                document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                const type = this.dataset.type;
                node.style('opacity', d => type === 'all' || d.type === type ? 1 : 0.1);
                link.style('opacity', d => type === 'all' || 
                    (nodes.find(n => n.id === d.source.id)?.type === type) || 
                    (nodes.find(n => n.id === d.target.id)?.type === type) ? 0.6 : 0.05);
            }});
        }});
        
        function dragstarted(event) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }}
        function dragged(event) {{
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }}
        function dragended(event) {{
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }}
    </script>
</body>
</html>'''
    
    output_path = os.path.join(OUTPUT_DIR, "knowledge_graph_interactive.html")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"   ✅ knowledge_graph_interactive.html")
    return output_path


def generate_sunburst_html(hierarchy_data):
    """Generate sunburst chart for hierarchical view."""
    
    tree = hierarchy_data.get("hierarchical_tree", {}).get("hierarchy", {})
    
    def convert_to_sunburst(node):
        result = {
            "name": node.get("name", "Unknown"),
            "value": node.get("entity_count", 1),
            "type": node.get("type", "unknown")
        }
        if "children" in node and node["children"]:
            result["children"] = [convert_to_sunburst(child) for child in node["children"]]
        return result
    
    sunburst_data = json.dumps(convert_to_sunburst(tree))
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Entity Hierarchy Sunburst</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3a 50%, #0a1a2a 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            color: #e8e8e8;
        }}
        .header {{
            width: 100%;
            background: rgba(255,255,255,0.03);
            backdrop-filter: blur(10px);
            padding: 20px 40px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            text-align: center;
        }}
        .header h1 {{
            font-size: 28px;
            background: linear-gradient(90deg, #ffd700, #ff6b6b, #4ecdc4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        #sunburst-container {{
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 30px;
        }}
        .tooltip {{
            position: absolute;
            background: rgba(20, 20, 40, 0.95);
            border: 1px solid rgba(255, 215, 0, 0.3);
            border-radius: 10px;
            padding: 12px;
            max-width: 250px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.6);
            pointer-events: none;
            opacity: 0;
            z-index: 1000;
        }}
        .tooltip h3 {{ color: #ffd700; margin-bottom: 6px; }}
        .tooltip p {{ font-size: 12px; color: #a0a0a0; }}
        .instructions {{
            position: fixed;
            bottom: 20px;
            background: rgba(20, 20, 40, 0.8);
            padding: 12px 20px;
            border-radius: 20px;
            font-size: 13px;
            color: #a0a0a0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>☀️ Entity Hierarchy Sunburst</h1>
    </div>
    <div id="sunburst-container"></div>
    <div class="tooltip" id="tooltip"></div>
    <div class="instructions">
        Click segments to zoom in | Click center to zoom out
    </div>
    <script>
        const data = {sunburst_data};
        
        const width = Math.min(window.innerWidth - 60, 800);
        const height = width;
        const radius = width / 2;
        
        const color = d3.scaleOrdinal()
            .domain(['root', 'branch', 'leaf'])
            .range(['#ff6b6b', '#4ecdc4', '#45b7d1']);
        
        const partition = d3.partition()
            .size([2 * Math.PI, radius]);
        
        const arc = d3.arc()
            .startAngle(d => d.x0)
            .endAngle(d => d.x1)
            .padAngle(d => Math.min((d.x1 - d.x0) / 2, 0.005))
            .padRadius(radius / 2)
            .innerRadius(d => d.y0)
            .outerRadius(d => d.y1 - 1);
        
        const root = d3.hierarchy(data)
            .sum(d => d.value)
            .sort((a, b) => b.value - a.value);
        
        partition(root);
        
        const svg = d3.select('#sunburst-container')
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .append('g')
            .attr('transform', `translate(${{width / 2}},${{height / 2}})`);
        
        const path = svg.selectAll('path')
            .data(root.descendants())
            .join('path')
            .attr('fill', d => {{
                let current = d;
                while (current.depth > 1) current = current.parent;
                return d3.interpolateRainbow(current.x0 / (2 * Math.PI));
            }})
            .attr('fill-opacity', d => 0.9 - d.depth * 0.15)
            .attr('d', arc)
            .style('cursor', 'pointer')
            .on('mouseover', function(event, d) {{
                d3.select(this).attr('fill-opacity', 1);
                const tooltip = document.getElementById('tooltip');
                tooltip.innerHTML = `
                    <h3>${{d.data.name}}</h3>
                    <p><strong>Type:</strong> ${{d.data.type}}</p>
                    <p><strong>Entities:</strong> ${{d.value}}</p>
                `;
                tooltip.style.opacity = 1;
                tooltip.style.left = (event.pageX + 15) + 'px';
                tooltip.style.top = (event.pageY - 10) + 'px';
            }})
            .on('mouseout', function(event, d) {{
                d3.select(this).attr('fill-opacity', 0.9 - d.depth * 0.15);
                document.getElementById('tooltip').style.opacity = 0;
            }})
            .on('click', clicked);
        
        const label = svg.selectAll('text')
            .data(root.descendants().filter(d => d.depth && (d.y0 + d.y1) / 2 * (d.x1 - d.x0) > 10))
            .join('text')
            .attr('transform', d => {{
                const x = (d.x0 + d.x1) / 2 * 180 / Math.PI;
                const y = (d.y0 + d.y1) / 2;
                return `rotate(${{x - 90}}) translate(${{y}},0) rotate(${{x < 180 ? 0 : 180}})`;
            }})
            .attr('dy', '0.35em')
            .attr('text-anchor', 'middle')
            .attr('fill', '#fff')
            .attr('font-size', '10px')
            .text(d => d.data.name.length > 10 ? d.data.name.substring(0, 10) + '...' : d.data.name);
        
        let currentFocus = root;
        
        function clicked(event, p) {{
            currentFocus = currentFocus === p ? p.parent || root : p;
            
            root.each(d => {{
                d.target = {{
                    x0: Math.max(0, Math.min(1, (d.x0 - currentFocus.x0) / (currentFocus.x1 - currentFocus.x0))) * 2 * Math.PI,
                    x1: Math.max(0, Math.min(1, (d.x1 - currentFocus.x0) / (currentFocus.x1 - currentFocus.x0))) * 2 * Math.PI,
                    y0: Math.max(0, d.y0 - currentFocus.y0),
                    y1: Math.max(0, d.y1 - currentFocus.y0)
                }};
            }});
            
            const t = svg.transition().duration(750);
            
            path.transition(t)
                .tween('data', d => {{
                    const i = d3.interpolate(d.current || d, d.target);
                    return t => d.current = i(t);
                }})
                .attrTween('d', d => () => arc(d.current));
        }}
        
        path.each(d => d.current = d);
    </script>
</body>
</html>'''
    
    output_path = os.path.join(OUTPUT_DIR, "sunburst_chart.html")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"   ✅ sunburst_chart.html")
    return output_path


def main():
    """Generate all visualizations."""
    print("\n🎨 Generating Interactive Visualizations...")
    print("=" * 50)
    
    # Load data
    print("\n📂 Loading data files...")
    hierarchy_data = load_hierarchical_model()
    print("   ✅ Loaded hierarchical_semantic_model.json")
    
    kg_data = load_knowledge_graph()
    print("   ✅ Loaded final_improved_graph.json")
    
    semantic_model = load_semantic_model()
    print("   ✅ Loaded semantic_model.json")
    
    # Generate visualizations
    print("\n🔧 Generating HTML visualizations...")
    
    generate_hierarchy_tree_html(hierarchy_data)
    generate_topic_network_html(semantic_model)
    generate_knowledge_graph_html(kg_data)
    generate_sunburst_html(hierarchy_data)
    
    print("\n" + "=" * 50)
    print("✨ All visualizations generated successfully!")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print("\n📋 Generated files:")
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith('.html'):
            print(f"   • {f}")


if __name__ == "__main__":
    main()
