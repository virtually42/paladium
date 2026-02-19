import * as d3 from 'd3';
import mermaid from 'mermaid';

// Initialize mermaid
mermaid.initialize({
  startOnLoad: false,
  theme: 'dark',
  securityLevel: 'loose'
});

// Import Scala.js module
let PaladiumFrontend = null;

async function loadScalaJS() {
  try {
    // Try to load from Mill's output directory
    const module = await import('scalajs/main.js');
    PaladiumFrontend = module.PaladiumFrontend || window.PaladiumFrontend;
    console.log('Scala.js module loaded successfully');
  } catch (e) {
    console.warn('Could not load Scala.js module directly, falling back to API:', e);
  }
}

// Extract variable names from expression
function extractVariables(expr) {
  const varPattern = /[a-zA-Z_][a-zA-Z0-9_]*/g;
  const reserved = new Set(['log', 'sin', 'cos', 'tan', 'exp', 'sqrt']);
  const matches = expr.match(varPattern) || [];
  return [...new Set(matches.filter(v => !reserved.has(v)))];
}

// Update variable inputs based on expression
function updateVariableInputs() {
  const expr = document.getElementById('expression').value;
  const variables = extractVariables(expr);
  const container = document.getElementById('variables');

  // Get existing values
  const existingValues = {};
  container.querySelectorAll('.variable-input').forEach(div => {
    const name = div.querySelector('label').textContent.replace(':', '');
    const value = div.querySelector('input').value;
    existingValues[name] = value;
  });

  // Rebuild inputs
  container.innerHTML = '';
  variables.forEach(varName => {
    const div = document.createElement('div');
    div.className = 'variable-input';

    const label = document.createElement('label');
    label.textContent = varName + ':';

    const input = document.createElement('input');
    input.type = 'text';
    input.value = existingValues[varName] || '1';
    input.dataset.variable = varName;

    div.appendChild(label);
    div.appendChild(input);
    container.appendChild(div);
  });
}

// Get variable values from inputs
function getVariableValues() {
  const values = {};
  document.querySelectorAll('.variable-input input').forEach(input => {
    values[input.dataset.variable] = parseFloat(input.value) || 0;
  });
  return values;
}

// Evaluate using Scala.js directly
async function evaluateScalaJS(expr, variables) {
  if (!PaladiumFrontend) {
    throw new Error('Scala.js not loaded');
  }

  const jsVars = {};
  for (const [k, v] of Object.entries(variables)) {
    jsVars[k] = v;
  }

  const result = PaladiumFrontend.evaluateExpression(expr, jsVars);
  const gradients = PaladiumFrontend.getGradients(expr, jsVars);
  const symbolicGradients = PaladiumFrontend.getSymbolicGradients(expr);
  const mermaidGraph = PaladiumFrontend.toMermaidGraph(expr);
  const d3Graph = JSON.parse(PaladiumFrontend.toD3Graph(expr));

  return { result, gradients, symbolicGradients, mermaidGraph, d3Graph };
}

// Evaluate using API fallback
async function evaluateAPI(expr, variables) {
  const [evalRes, gradRes, symbolicRes] = await Promise.all([
    fetch('/api/evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ expression: expr, variables })
    }).then(r => r.json()),

    fetch('/api/gradient', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ expression: expr, variables })
    }).then(r => r.json()),

    fetch('/api/symbolic-gradient', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ expression: expr })
    }).then(r => r.json())
  ]);

  return {
    result: evalRes.result,
    gradients: { value: gradRes.value, gradients: gradRes.gradients },
    symbolicGradients: symbolicRes.gradients,
    mermaidGraph: null,
    d3Graph: null
  };
}

// Render Mermaid graph
async function renderMermaid(graphDef) {
  if (!graphDef) return;

  const container = document.getElementById('mermaid-graph');
  try {
    const { svg } = await mermaid.render('mermaid-svg', graphDef);
    container.innerHTML = svg;
  } catch (e) {
    container.textContent = 'Error rendering graph: ' + e.message;
  }
}

// Render D3 force-directed graph
function renderD3Graph(graphData) {
  if (!graphData) return;

  const container = document.getElementById('graph-container');
  const width = container.clientWidth;
  const height = 400;

  // Clear previous graph
  d3.select('#d3-graph').selectAll('*').remove();

  const svg = d3.select('#d3-graph')
    .attr('width', width)
    .attr('height', height);

  // Create a copy of nodes and links for D3
  const nodes = graphData.nodes.map(n => ({ ...n }));
  const links = graphData.links.map(l => ({ ...l }));

  // Color scale for node types
  const color = d3.scaleOrdinal()
    .domain(['variable', 'literal', 'constant', 'operation', 'unary', 'function'])
    .range(['#e94560', '#0f3460', '#16213e', '#ff9a3c', '#ff6b6b', '#4ecdc4']);

  // Create force simulation
  const simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d => d.id).distance(80))
    .force('charge', d3.forceManyBody().strength(-300))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('y', d3.forceY(height / 2).strength(0.1));

  // Add arrow marker
  svg.append('defs').append('marker')
    .attr('id', 'arrow')
    .attr('viewBox', '0 -5 10 10')
    .attr('refX', 25)
    .attr('refY', 0)
    .attr('markerWidth', 6)
    .attr('markerHeight', 6)
    .attr('orient', 'auto')
    .append('path')
    .attr('fill', '#999')
    .attr('d', 'M0,-5L10,0L0,5');

  // Create links
  const link = svg.append('g')
    .selectAll('line')
    .data(links)
    .join('line')
    .attr('stroke', '#999')
    .attr('stroke-opacity', 0.6)
    .attr('stroke-width', 2)
    .attr('marker-end', 'url(#arrow)');

  // Create node groups
  const node = svg.append('g')
    .selectAll('g')
    .data(nodes)
    .join('g')
    .call(d3.drag()
      .on('start', dragstarted)
      .on('drag', dragged)
      .on('end', dragended));

  // Add circles to nodes
  node.append('circle')
    .attr('r', 20)
    .attr('fill', d => color(d.type))
    .attr('stroke', '#fff')
    .attr('stroke-width', 2);

  // Add labels to nodes
  node.append('text')
    .text(d => d.label)
    .attr('text-anchor', 'middle')
    .attr('dy', '0.35em')
    .attr('fill', '#fff')
    .attr('font-size', '12px')
    .attr('font-weight', 'bold');

  // Update positions on each tick
  simulation.on('tick', () => {
    link
      .attr('x1', d => d.source.x)
      .attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x)
      .attr('y2', d => d.target.y);

    node.attr('transform', d => `translate(${d.x},${d.y})`);
  });

  // Drag functions
  function dragstarted(event) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    event.subject.fx = event.subject.x;
    event.subject.fy = event.subject.y;
  }

  function dragged(event) {
    event.subject.fx = event.x;
    event.subject.fy = event.y;
  }

  function dragended(event) {
    if (!event.active) simulation.alphaTarget(0);
    event.subject.fx = null;
    event.subject.fy = null;
  }
}

// Main evaluation function
async function evaluate() {
  const expr = document.getElementById('expression').value;
  const variables = getVariableValues();

  try {
    let data;
    if (PaladiumFrontend) {
      data = await evaluateScalaJS(expr, variables);
    } else {
      data = await evaluateAPI(expr, variables);
    }

    // Display result
    document.getElementById('result').textContent = data.result.toFixed(6);

    // Display numerical gradients
    const gradientsList = document.getElementById('gradients');
    gradientsList.innerHTML = '';
    const grads = data.gradients.gradients || data.gradients;
    for (const [varName, value] of Object.entries(grads)) {
      const li = document.createElement('li');
      li.textContent = `d/d${varName} = ${value.toFixed(6)}`;
      gradientsList.appendChild(li);
    }

    // Display symbolic gradients
    const symbolicList = document.getElementById('symbolic-gradients');
    symbolicList.innerHTML = '';
    for (const [varName, expr] of Object.entries(data.symbolicGradients)) {
      const li = document.createElement('li');
      li.textContent = `d/d${varName} = ${expr}`;
      symbolicList.appendChild(li);
    }

    // Render graphs
    if (data.mermaidGraph) {
      await renderMermaid(data.mermaidGraph);
    }
    if (data.d3Graph) {
      renderD3Graph(data.d3Graph);
    }

  } catch (e) {
    console.error('Evaluation error:', e);
    document.getElementById('result').textContent = 'Error: ' + e.message;
  }
}

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
  await loadScalaJS();

  // Set up event listeners
  document.getElementById('expression').addEventListener('input', updateVariableInputs);
  document.getElementById('evaluate').addEventListener('click', evaluate);

  // Initial setup
  updateVariableInputs();
});
