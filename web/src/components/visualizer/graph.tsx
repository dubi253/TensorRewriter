import { component$, useSignal, useTask$ } from '@builder.io/qwik';
import { select, zoom, curveBasis, line } from 'd3';
import dagre from 'dagre';

export interface Tensor {
  name: string;
  shape: number[];
}

export interface Operator {
  type: string;
  inputs: string[];
  output: string;
  output_shape?: number[];
  params?: any;
}

export interface GraphData {
  inputs: Tensor[];
  outputs: Tensor[];
  operators: Operator[];
}

export const Graph = component$((props: { data: GraphData }) => {
  const svgRef = useSignal<Element>();
  const containerRef = useSignal<Element>();

  useTask$(({ track }) => {
    const data = track(() => props.data);
    const svgEl = track(() => svgRef.value);
    const containerEl = track(() => containerRef.value);
    
    if (typeof window === 'undefined' || !svgEl || !containerEl) return;

    // Clear previous
    select(svgEl).selectAll("*").remove();

    const width = (containerEl as HTMLElement).offsetWidth || 600;
    const height = (containerEl as HTMLElement).offsetHeight || 300;

    const svg = select(svgEl)
      .attr("width", width)
      .attr("height", height)
      .attr("viewBox", [0, 0, width, height])
      .attr("style", "max-width: 100%; height: auto;");

    // Arrow marker
    svg.append("defs").append("marker")
      .attr("id", "arrowhead")
      .attr("viewBox", "0 0 10 10")
      .attr("refX", 10) // Tip of the arrow at the end of the line
      .attr("refY", 5)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M 0 0 L 10 5 L 0 10 z")
      .attr("fill", "#333");

    // Initialize Dagre Graph
    const g = new dagre.graphlib.Graph();
    g.setGraph({ rankdir: 'TB', nodesep: 50, ranksep: 50 });
    g.setDefaultEdgeLabel(() => ({}));

    const tensorMap = new Map<string, any>();

    // Input Tensors
    data.inputs.forEach(t => {
      const nodeId = t.name;
      g.setNode(nodeId, { 
        label: t.name, 
        width: 60, 
        height: 60,
        type: 'tensor',
        shape: JSON.stringify(t.shape)
      });
      tensorMap.set(t.name, nodeId);
    });

    // Operators and Intermediate/Output Tensors
    data.operators.forEach(op => {
      // Operator Node
      const opNodeId = `op_${op.output}_${Math.random().toString(36).substr(2, 9)}`;
      g.setNode(opNodeId, { 
        label: op.type, 
        width: 100, 
        height: 40,
        type: 'operator',
        params: op.params
      });

      // Links from inputs to Op
      op.inputs.forEach(inputName => {
        if (tensorMap.has(inputName)) {
          g.setEdge(tensorMap.get(inputName), opNodeId);
        }
      });

      // Output Tensor Node
      let outNodeId = tensorMap.get(op.output);
      if (!outNodeId) {
        outNodeId = op.output;
        g.setNode(outNodeId, { 
          label: op.output, 
          width: 60, 
          height: 60,
          type: 'tensor',
          shape: op.output_shape ? JSON.stringify(op.output_shape) : '?'
        });
        tensorMap.set(op.output, outNodeId);
      } else {
          // Update shape if needed
          const node = g.node(outNodeId);
          if (op.output_shape) node.shape = JSON.stringify(op.output_shape);
      }

      // Link Op to Output
      g.setEdge(opNodeId, outNodeId);
    });

    // Calculate Layout
    dagre.layout(g);

    // Helper functions for intersection
    const intersectRect = (node: any, point: any) => {
        const x = node.x;
        const y = node.y;
        const dx = point.x - x;
        const dy = point.y - y;
        const w = node.width / 2;
        const h = node.height / 2;
        
        if (dx === 0 && dy === 0) return {x, y};
        
        if (Math.abs(dy) * w > Math.abs(dx) * h) {
            // Intersection with top/bottom
            if (dy > 0) {
                return { x: x + dx * h / Math.abs(dy), y: y + h };
            } else {
                return { x: x + dx * h / Math.abs(dy), y: y - h };
            }
        } else {
            // Intersection with left/right
            if (dx > 0) {
                return { x: x + w, y: y + dy * w / Math.abs(dx) };
            } else {
                return { x: x - w, y: y + dy * w / Math.abs(dx) };
            }
        }
    };

    const intersectCircle = (node: any, point: any, r: number) => {
        const dx = point.x - node.x;
        const dy = point.y - node.y;
        const dist = Math.sqrt(dx*dx + dy*dy);
        if (dist < 1e-6) return {x: node.x, y: node.y};
        
        return {
            x: node.x + (dx / dist) * r,
            y: node.y + (dy / dist) * r
        };
    };

    // Extract nodes and edges
    const nodes = g.nodes().map(v => {
        const node = g.node(v);
        return {
            id: v,
            ...node,
            x: node.x,
            y: node.y
        };
    });

    const edges = g.edges().map(e => {
        const edge = g.edge(e);
        const sourceNode = g.node(e.v) as any;
        const targetNode = g.node(e.w) as any;
        
        const points = edge.points.slice();
        
        // Clip start (Source) and end (Target) to node boundaries
        if (points.length > 1) {
            // Start
            const p2 = points[1];
            let startPoint;
            if (sourceNode.type === 'tensor') {
                startPoint = intersectCircle(sourceNode, p2, 30);
            } else {
                startPoint = intersectRect(sourceNode, p2);
            }
            points[0] = startPoint;
            
            // End
            const n = points.length;
            const pn_1 = points[n-2];
            let endPoint;
            if (targetNode.type === 'tensor') {
                endPoint = intersectCircle(targetNode, pn_1, 30);
            } else {
                endPoint = intersectRect(targetNode, pn_1);
            }
            points[n-1] = endPoint;
        }

        return {
            source: { x: sourceNode.x, y: sourceNode.y, id: e.v },
            target: { x: targetNode.x, y: targetNode.y, id: e.w },
            points: points
        };
    });

    // Center the graph
    const graphWidth = g.graph().width || width;
    const graphHeight = g.graph().height || height;
    const xOffset = (width - graphWidth) / 2;
    const yOffset = (height - graphHeight) / 2;

    // Zoom behavior
    const zoomG = svg.append("g");
    
    const zoomBehavior = zoom()
        .scaleExtent([0.1, 4])
        .on("zoom", (event) => {
            zoomG.attr("transform", event.transform);
        });

    // Initial transform to center
    // const initialScale = Math.min(width / (graphWidth + 100), height / (graphHeight + 100), 1);
    
    svg.call(zoomBehavior as any);
    
    // Apply initial offset to nodes/edges? No, better to use the group transform.
    // But d3 zoom controls the transform.
    // Let's just center the layout coordinates themselves if we want it centered by default.
    // Or just rely on the user.
    // Let's shift the nodes.
    
    nodes.forEach(n => {
        n.x += xOffset > 0 ? xOffset : 50;
        n.y += yOffset > 0 ? yOffset : 50;
    });
    
    edges.forEach(e => {
        e.points.forEach((p: any) => {
            p.x += xOffset > 0 ? xOffset : 50;
            p.y += yOffset > 0 ? yOffset : 50;
        });
    });

    // Draw Edges
    // Dagre gives points for edges, we can use a line generator
    const lineGenerator = line()
        .x((d: any) => d.x)
        .y((d: any) => d.y)
        .curve(curveBasis);

    zoomG.append("g")
      .attr("stroke", "#333")
      .attr("stroke-opacity", 0.8)
      .selectAll("path")
      .data(edges)
      .join("path")
      .attr("d", (d: any) => lineGenerator(d.points))
      .attr("fill", "none")
      .attr("stroke-width", 2.5)
      .attr("marker-end", "url(#arrowhead)");

    const node = zoomG.append("g")
      .attr("stroke", "#fff")
      .attr("stroke-width", 1.5)
      .selectAll("g")
      .data(nodes)
      .join("g")
      .attr("transform", (d: any) => `translate(${d.x},${d.y})`);
      // .call(drag()
      //   .on("start", dragstarted)
      //   .on("drag", dragged)
      //   .on("end", dragended) as any);

    // Draw Tensors (Circles)
    node.filter((d: any) => d.type === 'tensor')
      .append("circle")
      .attr("r", 30)
      .attr("fill", "#e0f7fa")
      .attr("stroke", "#006064");

    // Draw Operators (Rects)
    node.filter((d: any) => d.type === 'operator')
      .append("rect")
      .attr("width", 100)
      .attr("height", 40)
      .attr("x", -50)
      .attr("y", -20)
      .attr("rx", 10)
      .attr("ry", 10)
      .attr("fill", "#fff3e0")
      .attr("stroke", "#e65100");

    // Labels
    node.append("text")
      .attr("dy", (d: any) => d.type === 'tensor' ? "-0.2em" : "0.3em")
      .attr("text-anchor", "middle")
      .text((d: any) => d.label)
      .attr("fill", "#333")
      .attr("stroke", "none")
      .attr("font-size", "12px")
      .style("pointer-events", "none");

    // Params Labels for Operators
    node.filter((d: any) => d.type === 'operator' && d.params && Object.keys(d.params).length > 0)
      .append("text")
      .attr("dy", "1.5em")
      .attr("text-anchor", "middle")
      .text((d: any) => Object.entries(d.params).map(([k, v]) => `${k}:${v}`).join(', '))
      .attr("fill", "#555")
      .attr("stroke", "none")
      .attr("font-size", "10px")
      .style("pointer-events", "none");

    // Shape Labels for Tensors
    node.filter((d: any) => d.type === 'tensor')
      .append("text")
      .attr("dy", "1.2em")
      .attr("text-anchor", "middle")
      .text((d: any) => d.shape)
      .attr("fill", "#666")
      .attr("stroke", "none")
      .attr("font-size", "10px")
      .style("pointer-events", "none");
    
  });

  return (
    <div ref={containerRef} style={{ width: '100%', height: '300px', border: '1px solid #ccc', borderRadius: '8px', background: '#f9f9f9', overflow: 'hidden' }}>
      <svg ref={svgRef}></svg>
    </div>
  );
});

