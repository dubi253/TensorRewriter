# TensorRewrite: Automatic Tensor Rule Synthesizer

TensorRewriter is a Python-based system that automatically discovers graph rewrite rules (optimizations) for deep learning compilers. It works by systematically enumerating Directed Acyclic Graphs (DAGs) of tensor operations and verifying numerical equivalence between them using random inputs.

See our [web visualizer](https://dubi253.github.io/TensorRewriter/) to explore the discovered rules!

## Features

*   **Automatic Rule Discovery**: Synthesizes rewrite rules by enumerating graphs and checking for equivalence.
*   **Extensible Operator Set**: Easily add new tensor operations.
*   **Parallel Synthesis**: Uses multiprocessing to speed up the search space exploration.
*   **Exportable Rules**: Saves rules in JSON format for integration with other tools.
*   **Web Visualizer (Preview)**: Interactive web interface to visualize discovered rules.

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

**Install dependencies**:
```bash
uv sync --extra dev
```

## Usage

### 1. Run Synthesis

To start the rule discovery process:

```bash
uv run scripts/run_synthesis.py
```

This will:
*   Enumerate tensor graphs up to a specified depth (default: 4).
*   Verify equivalence using random inputs.
*   Export discovered rules to `web/src/rules.json`.

### 2. Visualize Rules

The project includes a modern web interface to explore the synthesized rules.

1.  **Navigate to the web directory**:
    ```bash
    cd web
    ```

2.  **Install dependencies** (requires [Bun](https://bun.sh/)):
    ```bash
    bun install
    ```

3.  **Start the server**:
    ```bash
    bun build && bun start
    ```

4.  Open your browser at `http://localhost:5173/`.

The web interface features:
*   **Interactive Graph Visualization**: View source and target graphs using D3.js.
*   **Search**: Filter rules by operator types (e.g., "matmul", "relu").
*   **Pagination**: Browse through discovered rules with adjustable page size.
*   **Responsive Design**: Built with Tailwind CSS and DaisyUI for a modern look.

## Project Structure

*   `tensor_rewrite/`: Core Python package.
    *   `core.py`: Graph and Tensor definitions.
    *   `ops.py`: Operator implementations and shape inference.
    *   `synthesizer.py`: Main synthesis logic (enumeration, equivalence checking).
    *   `evaluator.py`: Graph execution engine.
*   `scripts/`: Utility scripts (e.g., `run_synthesis.py`).
*   `web/`: Web-based visualizer (HTML/JS).
*   `tests/`: Unit tests.

## Supported Operators (Can be extended)

*   **MatMul**: Matrix Multiplication
*   **ElementWiseAdd**: Element-wise addition
*   **ElementWiseSub**: Element-wise subtraction
*   **Transpose**: Matrix transposition
*   **Relu**: Rectified Linear Unit
*   **Reshape**: Tensor reshaping (automatically finds valid target shapes)
*   **Concat**: Tensor concatenation (automatically finds valid axes)
*   **Slice**: Tensor slicing
*   **Sum**: Reduction sum
*   **ExpandDims**: Add dimension
*   **ElementWiseMul**: Element-wise multiplication
*   **Mean**: Reduction mean
*   **Max**: Reduction max
*   **Min**: Reduction min
*   **Prod**: Reduction product

## Development

### Python Core

To run the test suite:

```bash
uv run pytest
```

### Web Interface

The web visualizer is built with **Qwik**, **Vite**, **Tailwind CSS**, and **DaisyUI**.

*   **Linting**: `bun run lint`
*   **Building**: `bun run build`
*   **Project Structure**:
    *   `web/src/routes/`: Main application routes.
    *   `web/src/components/visualizer/`: D3.js graph components.
