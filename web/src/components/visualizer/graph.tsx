import { component$, useSignal, useTask$ } from '@builder.io/qwik';
import { select, drag, zoom, forceSimulation, forceLink, forceManyBody, forceCenter, forceCollide, forceY, forceX } from 'd3';

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
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 32) // Adjust based on node radius
      .attr("refY", 0)
      .attr("markerWidth", 8)
      .attr("markerHeight", 8)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("fill", "#333");

    // Process data into nodes and links
    const nodes: any[] = [];
    const links: any[] = [];
    const tensorMap = new Map<string, any>();

    // Input Tensors
    data.inputs.forEach(t => {
      const node = {
        id: t.name,
        type: 'tensor',
        label: t.name,
        shape: JSON.stringify(t.shape),
        r: 30
      };
      nodes.push(node);
      tensorMap.set(t.name, node);
    });

    // Operators and Intermediate/Output Tensors
    data.operators.forEach(op => {
      // Operator Node
      const opNode = {
        id: `op_${op.output}_${Math.random().toString(36).substr(2, 9)}`,
        type: 'operator',
        label: op.type,
        params: op.params,
        width: 100,
        height: 40
      };
      nodes.push(opNode);

      // Links from inputs to Op
      op.inputs.forEach(inputName => {
        // If input tensor doesn't exist (maybe it's an intermediate not in inputs list?), find or create?
        // In this system, inputs should be in 'inputs' or produced by previous ops.
        // If produced by previous op, it should be in tensorMap.
        if (tensorMap.has(inputName)) {
          links.push({ source: tensorMap.get(inputName).id, target: opNode.id });
        }
      });

      // Output Tensor Node
      // Check if it's already created (shouldn't be, SSA usually)
      let outNode = tensorMap.get(op.output);
      if (!outNode) {
        outNode = {
          id: op.output,
          type: 'tensor',
          label: op.output,
          shape: op.output_shape ? JSON.stringify(op.output_shape) : '?',
          r: 30
        };
        nodes.push(outNode);
        tensorMap.set(op.output, outNode);
      } else {
          // Update shape if we didn't have it (e.g. if it was in inputs but now we know it's output of something? Unlikely in SSA)
          if (op.output_shape) outNode.shape = JSON.stringify(op.output_shape);
      }

      // Link Op to Output
      links.push({ source: opNode.id, target: outNode.id });
    });

    // Better approach: Assign levels directly to node objects
    const nodeMap = new Map<string, any>();
    nodes.forEach(n => nodeMap.set(n.id, n));
    
    // Initialize levels
    nodes.forEach(n => n.level = -1);
    data.inputs.forEach(t => {
        const n = tensorMap.get(t.name);
        if (n) n.level = 0;
    });

    let changed = true;
    while (changed) {
        changed = false;
        // Propagate levels
        links.forEach(link => {
            const source = typeof link.source === 'object' ? link.source : nodeMap.get(link.source);
            const target = typeof link.target === 'object' ? link.target : nodeMap.get(link.target);
            
            if (source && target && source.level !== -1) {
                if (target.level < source.level + 1) {
                    target.level = source.level + 1;
                    changed = true;
                }
            }
        });
    }
    
    // Fallback for disconnected or cycles (shouldn't happen in DAG)
    nodes.forEach(n => { if (n.level === -1) n.level = 0; });

    // Initial positioning to minimize crossings (Barycenter heuristic)
    // Group nodes by level
    const levels = new Map<number, any[]>();
    nodes.forEach(n => {
        if (!levels.has(n.level)) levels.set(n.level, []);
        levels.get(n.level)!.push(n);
    });

    // Sort levels
    const sortedLevels = Array.from(levels.keys()).sort((a, b) => a - b);
    
    // Assign initial X positions
    // const levelWidth = width / (sortedLevels.length + 1);
    
    sortedLevels.forEach(level => {
        const levelNodes = levels.get(level)!;
        // If level 0, just spread them
        if (level === 0) {
            levelNodes.forEach((n, i) => {
                n.x = (i + 1) * (width / (levelNodes.length + 1));
                n.y = 50;
            });
        } else {
            // For other levels, try to place near parents
            levelNodes.forEach(n => {
                // Find parents
                const parents = links
                    .filter(l => (typeof l.target === 'object' ? l.target.id : l.target) === n.id)
                    .map(l => typeof l.source === 'object' ? l.source : nodeMap.get(l.source));
                
                if (parents.length > 0) {
                    const avgX = parents.reduce((sum, p) => sum + (p.x || width/2), 0) / parents.length;
                    n.x = avgX;
                } else {
                    n.x = width / 2;
                }
                n.y = level * 120 + 50;
            });
            
            // Spread overlapping nodes in the same level
            levelNodes.sort((a, b) => a.x - b.x);
            // levelNodes.forEach((n, i) => {
            //     // Simple spread if too close
            //     // This is just a hint for the force layout
            // });
        }
    });

    // Simulation
    const simulation = forceSimulation(nodes)
      .force("link", forceLink(links).id((d: any) => d.id).distance(80))
      .force("charge", forceManyBody().strength(-300))
      .force("center", forceCenter(width / 2, height / 2).strength(0.05))
      .force("collide", forceCollide().radius((d: any) => d.type === 'tensor' ? 45 : 65))
      .force("y", forceY((d: any) => d.level * 120 + 50).strength(2)) // Stronger vertical force
      .force("x", forceX((d: any) => d.x).strength(0.5)) // Use calculated X as target
      .stop();

    simulation.tick(300);

    // Zoom behavior
    const g = svg.append("g");
    
    const zoomBehavior = zoom()
        .scaleExtent([0.1, 4])
        .on("zoom", (event) => {
            g.attr("transform", event.transform);
        });

    svg.call(zoomBehavior as any);

    const link = g.append("g")
      .attr("stroke", "#333")
      .attr("stroke-opacity", 0.8)
      .selectAll("line")
      .data(links)
      .join("line")
      .attr("stroke-width", 2.5)
      .attr("marker-end", "url(#arrowhead)");

    const node = g.append("g")
      .attr("stroke", "#fff")
      .attr("stroke-width", 1.5)
      .selectAll("g")
      .data(nodes)
      .join("g")
      .call(drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended) as any);

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

    const ticked = () => {
      link
        .attr("x1", (d: any) => d.source.x)
        .attr("y1", (d: any) => d.source.y)
        .attr("x2", (d: any) => d.target.x)
        .attr("y2", (d: any) => d.target.y);

      node
        .attr("transform", (d: any) => `translate(${d.x},${d.y})`);
    };

    simulation.on("tick", ticked);
    ticked();

    function dragstarted(event: any) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }

    function dragged(event: any) {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    }

    function dragended(event: any) {
      if (!event.active) simulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    }
    
    // Cleanup
    return () => simulation.stop();
  });

  return (
    <div ref={containerRef} style={{ width: '100%', height: '300px', border: '1px solid #ccc', borderRadius: '8px', background: '#f9f9f9', overflow: 'hidden' }}>
      <svg ref={svgRef}></svg>
    </div>
  );
});

