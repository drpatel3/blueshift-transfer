import logging
import base64
import json
from pathlib import Path
import os
from collections import defaultdict, deque
try:
    import tiktoken
except ImportError:
    tiktoken = None

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM client setup — AWS Bedrock only (OpenAI paths removed)
# ---------------------------------------------------------------------------

import boto3
_bedrock = boto3.client("bedrock-runtime", region_name=config.AWS_REGION)


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def count_text_tokens(text: str) -> int:
    # tiktoken is legacy — it used OpenAI's encoder. For Bedrock we fall back to
    # the char-per-token heuristic, which is close enough for cost accounting
    # and budget warnings (the model's reported usage is authoritative anyway).
    if tiktoken is None:
        return len(text) // 4
    encoding = tiktoken.encoding_for_model("gpt-4")
    return len(encoding.encode(text))


# Pricing per 1M tokens — Bedrock Claude only.
MODEL_PRICING = {
    "us.anthropic.claude-opus-4-6-v1": {"input": 15.00, "output": 75.00},
    "us.anthropic.claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
}


def _call_llm(prompt: str, base64_image: str, model: str,
              temperature: float = 0.0) -> dict:
    """Unified LLM call. Returns {"content": str, "input_tokens": int, "output_tokens": int}."""
    return _call_bedrock(prompt, base64_image, model, temperature)


def _call_bedrock(prompt: str, base64_image: str, model: str,
                  temperature: float) -> dict:
    image_bytes = base64.b64decode(base64_image)
    response = _bedrock.converse(
        modelId=model,
        messages=[{
            "role": "user",
            "content": [
                {"image": {"format": "png", "source": {"bytes": image_bytes}}},
                {"text": prompt},
            ],
        }],
        inferenceConfig={"temperature": temperature, "maxTokens": 4096},
    )
    output = response["output"]["message"]["content"][0]["text"]
    usage = response["usage"]
    return {
        "content": output,
        "input_tokens": usage["inputTokens"],
        "output_tokens": usage["outputTokens"],
    }


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in dollars for a given model and token counts."""
    pricing = MODEL_PRICING.get(model, {"input": 0, "output": 0})
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def clean_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def load_terms_reference(terms_path: str = None) -> str:
    """Load the TERMS.md reference file."""
    if terms_path is None:
        terms_path = config.TERMS_PATH
    with open(terms_path, 'r', encoding='utf-8') as f:
        return f.read()


def extract_with_terms(image_path: str, max_iterations: int = 3) -> dict:
    """Extract flowsheet data using TERMS.md as stage reference."""
    base64_image = encode_image(image_path)
    terms_reference = load_terms_reference()
    total_tokens = 0
    total_cost = 0.0
    costs_by_model = defaultdict(float)

    # Initial extraction
    initial_prompt = f"""You are an expert process engineer analyzing an ore-to-metal process flowsheet.

    Use this reference to identify and categorize stages. Match flowsheet terms to standard IDs using the synonyms:

    <stage_reference>
    {terms_reference}
    </stage_reference>

    Extract ALL information from this flowsheet and return JSON with:
    1. "stages": array of process stages, each with:
       - id: lowercase identifier (e.g., "primary_crushing")
       - name: descriptive stage name
       - order: sequence number from input (1 = first stage)

    2. "units": array of equipment, each with:
       - id: lowercase identifier
       - label: exact text from diagram
       - type: equipment type
       - stage_id: which stage this belongs to
       - cells: number of cells if shown, otherwise 1

    3. "connections": array of flows, each with:
       - parent_id: source stage id (use "INPUT" for ore feed)
       - child_id: destination stage id (use "OUTPUT" for final product)

    Return only valid JSON."""
    llm_model = config.BEDROCK_MODEL
    response = _call_llm(initial_prompt, base64_image, llm_model)

    current_response = clean_json(response["content"])
    prompt_tokens = response["input_tokens"]
    completion_tokens = response["output_tokens"]
    total_tokens += prompt_tokens + completion_tokens
    initial_cost = calculate_cost(llm_model, prompt_tokens, completion_tokens)
    total_cost += initial_cost
    costs_by_model[llm_model] += initial_cost
    text_tokens = count_text_tokens(initial_prompt)
    image_tokens = prompt_tokens - text_tokens
    logger.info(f"Text tokens: {text_tokens}, Image tokens: {image_tokens}, Output tokens: {completion_tokens}")
    logger.info(f"Initial call cost ({llm_model}): ${initial_cost:.4f}")

    logger.info(f"Total tokens used: {total_tokens}")
    logger.info(f"Total cost: ${total_cost:.4f}")
    logger.info(f"Cost by model: " + ", ".join(f"{model}: ${cost:.4f}" for model, cost in costs_by_model.items()))
    try:
        data = json.loads(current_response)
        return {"data": data, "total_tokens": total_tokens, "total_cost": total_cost, "costs_by_model": dict(costs_by_model)}
    except json.JSONDecodeError as e:
        return {"error": str(e), "raw": current_response, "total_tokens": total_tokens, "total_cost": total_cost, "costs_by_model": dict(costs_by_model)}


def extract_with_refinement(image_path: str, max_iterations: int = 3, return_history: bool = False, on_iteration: callable = None) -> dict:
    """Extract flowsheet data with self-refinement loop, Bedrock-only.

    Uses BEDROCK_MODEL for initial extraction and BEDROCK_REFINEMENT_MODEL for
    refinement. If return_history=True, returns list of
    {"iteration": n, "tokens": t, "data": parsed_json} for each step.
    """
    base64_image = encode_image(image_path)
    terms_reference = load_terms_reference()
    total_tokens = 0
    total_cost = 0.0
    costs_by_model = defaultdict(float)
    history = []

    initial_model = config.BEDROCK_MODEL
    refinement_model = config.BEDROCK_REFINEMENT_MODEL

    # Initial extraction
    initial_prompt = f"""You are an expert process engineer analyzing an ore-to-metal process flowsheet.

    Use this reference to identify and categorize stages. Match flowsheet terms to standard IDs using the synonyms:

    <stage_reference>
    {terms_reference}
    </stage_reference>

    Extract ALL information from this flowsheet and return JSON with:
    1. "stages": array of process stages, each with:
       - id: lowercase identifier (e.g., "primary_crushing")
       - name: descriptive stage name
       - order: sequence number from input (1 = first stage)

    2. "units": array of equipment, each with:
       - id: lowercase identifier
       - label: exact text from diagram
       - type: equipment type
       - stage_id: which stage this belongs to
       - cells: number of cells if shown, otherwise 1

    3. "connections": array of flows, each with:
       - parent_id: source stage id (use "INPUT" for ore feed)
       - child_id: destination stage id (use "OUTPUT" for final product)

    Return only valid JSON."""

    response = _call_llm(initial_prompt, base64_image, initial_model)

    current_response = clean_json(response["content"])
    prompt_tokens = response["input_tokens"]
    completion_tokens = response["output_tokens"]
    total_tokens += prompt_tokens + completion_tokens
    initial_cost = calculate_cost(initial_model, prompt_tokens, completion_tokens)
    total_cost += initial_cost
    costs_by_model[initial_model] += initial_cost
    logger.info(f"Initial call cost ({initial_model}): ${initial_cost:.4f}")

    # Store initial result
    try:
        data = json.loads(current_response)
        history.append({"iteration": 0, "tokens": total_tokens, "data": data})
        if on_iteration:
            on_iteration(0, data)
    except json.JSONDecodeError:
        history.append({"iteration": 0, "tokens": total_tokens, "data": None})

    # Scale iterations by complexity: <1000 = 0, 1000-1499 = 1, +1 per 500 tokens after
    if prompt_tokens < 1000:
        iterations_needed = 0
    else:
        iterations_needed = 1
    logger.info(f"Input tokens: {prompt_tokens} -> {iterations_needed} refinement round(s)")

    # Refinement loop
    for i in range(iterations_needed):
        refinement_prompt = f"""Evaluate the current response: {current_response}

        Compare the response against the flowsheet image. Be extremely rigorous.

        Look for and add:
          - any missing equipment
          - incorrect connections
          - wrong stage assignments

          Return ONLY the corrected JSON with no explanation.
        """

        response = _call_llm(refinement_prompt, base64_image, refinement_model)

        current_response = clean_json(response["content"])
        ref_prompt_tokens = response["input_tokens"]
        ref_completion_tokens = response["output_tokens"]
        total_tokens += ref_prompt_tokens + ref_completion_tokens
        ref_cost = calculate_cost(refinement_model, ref_prompt_tokens, ref_completion_tokens)
        total_cost += ref_cost
        costs_by_model[refinement_model] += ref_cost
        logger.info(f"=== Refinement {i+1}/{iterations_needed} (${ref_cost:.4f}) ===\n{current_response[:500]}...\n")

        # Store refinement result
        try:
            data = json.loads(current_response)
            history.append({"iteration": i + 1, "tokens": total_tokens, "data": data})
            if on_iteration:
                on_iteration(i + 1, data)
        except json.JSONDecodeError:
            history.append({"iteration": i + 1, "tokens": total_tokens, "data": None})

    logger.info(f"Total tokens used: {total_tokens}")
    logger.info(f"Total cost: ${total_cost:.4f}")
    logger.info(f"Cost by model: " + ", ".join(f"{model}: ${cost:.4f}" for model, cost in costs_by_model.items()))

    if return_history:
        return {"history": history, "total_tokens": total_tokens, "total_cost": total_cost, "costs_by_model": dict(costs_by_model)}
    try:
        data = json.loads(current_response)
        return {"data": data, "total_tokens": total_tokens, "total_cost": total_cost, "costs_by_model": dict(costs_by_model)}
    except json.JSONDecodeError as e:
        return {"error": str(e), "raw": current_response, "total_tokens": total_tokens, "total_cost": total_cost, "costs_by_model": dict(costs_by_model)}


def build_graph(data: dict):
    """Build NetworkX graph from extracted flowsheet data."""
    import networkx as nx
    G = nx.DiGraph()

    # Add stage nodes
    for stage in data.get("stages", []):
        G.add_node(stage["id"], label=stage.get("name", stage["id"]), node_type="stage")

    # Add connections between stages
    for conn in data.get("connections", []):
        src = conn.get("parent_id") or conn.get("from_type") or conn.get("from_stage")
        dst = conn.get("child_id") or conn.get("to_type") or conn.get("to_stage")
        if src and dst:
            # Add INPUT/OUTPUT nodes if they don't exist
            if src == "INPUT" and src not in G:
                G.add_node("INPUT", label="INPUT", node_type="terminal")
            if dst == "OUTPUT" and dst not in G:
                G.add_node("OUTPUT", label="OUTPUT", node_type="terminal")
            G.add_edge(src, dst)

    # Remove orphaned nodes (no parent and not INPUT)
    if "INPUT" in G:
        reachable = set(nx.descendants(G, "INPUT")) | {"INPUT"}
        orphans = [n for n in G.nodes if n not in reachable]
        if orphans:
            logger.info(f"Removing {len(orphans)} orphaned nodes: {orphans}")
            G.remove_nodes_from(orphans)

    return G


def save_result(results_dict: dict, image_path: str, data: dict, output_file: str = "results.json") -> dict:
    """Save extraction result with image filename as key."""
    key = os.path.splitext(os.path.basename(image_path))[0]
    results_dict[key] = data

    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    return results_dict


def draw_graph(G, output_file: str = "refinement_graph.png"):
    """Draw the flowsheet graph with hierarchical layout."""
    import networkx as nx
    import matplotlib.pyplot as plt
    # Compute layers via BFS from roots
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    if not roots:
        roots = list(G.nodes)[:1]

    layer = {}
    visited = set()
    queue = deque([(r, 0) for r in roots])

    while queue:
        node, depth = queue.popleft()
        if node in visited:
            layer[node] = max(layer.get(node, 0), depth)
            continue
        visited.add(node)
        layer[node] = depth
        for child in G.successors(node):
            queue.append((child, depth + 1))

    for n in G.nodes:
        if n not in layer:
            layer[n] = 0

    # Group by layer
    layers = defaultdict(list)
    for node, l in layer.items():
        layers[l].append(node)

    # Compute positions
    pos = {}
    max_layer = max(layers.keys()) if layers else 0
    for l, nodes in layers.items():
        y = max_layer - l
        for i, node in enumerate(nodes):
            x = (i - (len(nodes) - 1) / 2) * 3.0
            pos[node] = (x, y * 2.0)

    # Draw
    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=15, edge_color="gray", ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color="lightblue", ax=ax)
    labels = {n: G.nodes[n].get("label", n) for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

    ax.set_title("Refined Flowsheet Graph", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.show()
    logger.info(f"Graph saved to {output_file}")

def calculate_f1(results_path, ground_truth_path):
    """Calculate F1 score from results and ground truth"""
    with open(results_path, 'r') as f:
        results = json.load(f)
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)

    tp, fp, fn = 0, 0, 0

    for key, gt_data in ground_truth.items():
        res_data = results.get(key, {})
        gt_stages = {stage['id'] for stage in gt_data.get('stages', [])}
        res_stages = {stage['id'] for stage in res_data.get('stages', [])}

        tp += len(gt_stages & res_stages)
        fp += len(res_stages - gt_stages)
        fn += len(gt_stages - res_stages)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    return f1

if __name__ == "__main__":
    folder = Path("extracted_images")
    image_paths = list(folder.glob("*.png"))
    results = {}
    counter = 0
    overall_cost = 0.0
    overall_tokens = 0
    overall_costs_by_model = defaultdict(float)
    test_image_paths_short = [folder / "7_pdf_flowsheet_1_page148.png"]
    test_image_paths = [folder / "7_pdf_flowsheet_1_page148.png", folder / "test_casino_flowsheet_1_page212.png", folder / "15_pdf_flowsheet_1_page171.png", folder / "26_pdf_flowsheet_1_page193.png"]
    logging.basicConfig(level=logging.INFO)
    for image_path in test_image_paths:
        logger.info(f"Processing: {image_path}")
        logger.info(f"{'='*60}\n")
        result = extract_with_terms(str(image_path), max_iterations=3)

        # Store result with image filename as key
        key = os.path.splitext(os.path.basename(image_path))[0]
        if "error" in result:
            logger.error(f"Error: {result['error']}")
            results[key] = {"error": result['error'], "raw": result.get('raw', '')}
        else:
            logger.info(f"Image total tokens: {result['total_tokens']}")
            logger.info(f"Image total cost: ${result['total_cost']:.4f}")
            results[key] = result.get('data', {})

        overall_cost += result.get('total_cost', 0)
        overall_tokens += result.get('total_tokens', 0)
        for model, cost in result.get('costs_by_model', {}).items():
            overall_costs_by_model[model] += cost
        counter += 1
        logger.info(f"\n--- Running totals: {counter} images, {overall_tokens} tokens, ${overall_cost:.4f} ---")
        if counter > 5:
            break

    # Save results to JSON
    with open("results_two.json", 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL SUMMARY: {counter} images processed")
    logger.info(f"Total tokens: {overall_tokens}")
    logger.info(f"Total cost: ${overall_cost:.4f}")
    logger.info(f"Cost by model:")
    for model, cost in overall_costs_by_model.items():
        logger.info(f"  {model}: ${cost:.4f}")
    logger.info(f"Results saved to results_two.json")
    logger.info(f"{'='*60}")